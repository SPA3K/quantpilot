#!/usr/bin/env python3
"""
QuantPilot ML Prediction Tracker
记录每日ML预测信号 + 次日实际涨跌，计算假设跟随ML操作的收益。

用法:
  # 记录今日预测（收盘前调用）
  python3 ml_tracker.py predict --date 2026-08-26 --signals '{"芯片/半导体": 0.62, ...}'
  
  # 记录实际结果（次日开盘后调用）
  python3 ml_tracker.py result --date 2026-08-26 --actuals '{"芯片/半导体": 1.2, ...}'
  
  # 查看统计
  python3 ml_tracker.py stats
  
  # 导出宣发数据
  python3 ml_tracker.py export

数据文件: $QUANTPILOT_DATA_DIR/ml_tracker_data.json (默认 ~/.quantpilot/industry/)
"""

import json, sys, argparse, datetime
import os
from pathlib import Path
STATE_DIR = Path(os.environ.get("QUANTPILOT_DATA_DIR")
             or (Path.home() / ".quantpilot" / "industry"))


DATA_PATH = STATE_DIR / "ml_tracker_data.json"

SECTORS = ["芯片/半导体", "CPO/光通信", "具身/机器人", "科技宽基", "新能源"]

def load_data():
    if DATA_PATH.exists():
        return json.loads(DATA_PATH.read_text())
    return {"records": [], "start_date": datetime.date.today().isoformat(), "config": {
        "threshold_buy": 0.55,   # ML score > 此值 → 买入信号
        "threshold_sell": 0.45,  # ML score < 此值 → 卖出/观望
        "position_size": 0.2,   # 每个方向仓位比例
    }}

def save_data(data):
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def cmd_predict(args):
    """记录今日ML预测"""
    data = load_data()
    signals = json.loads(args.signals)
    
    record = {
        "date": args.date,
        "predictions": signals,
        "actuals": {},  # 次日填充
        "action": {},   # 根据threshold生成操作建议
        "pnl": {}       # 次日计算
    }
    
    # 生成操作建议
    threshold_buy = data["config"]["threshold_buy"]
    threshold_sell = data["config"]["threshold_sell"]
    for sector, score in signals.items():
        if score >= threshold_buy:
            record["action"][sector] = "买入"
        elif score <= threshold_sell:
            record["action"][sector] = "观望/卖出"
        else:
            record["action"][sector] = "观望"
    
    # 检查是否已有该日期的记录
    existing = [i for i, r in enumerate(data["records"]) if r["date"] == args.date]
    if existing:
        data["records"][existing[0]] = record
        print(f"更新 {args.date} 的预测记录")
    else:
        data["records"].append(record)
        print(f"新增 {args.date} 的预测记录")
    
    save_data(data)
    
    # 输出操作建议
    print(f"\n📊 {args.date} ML操作建议:")
    for sector, action in record["action"].items():
        score = signals[sector]
        emoji = "🟢" if action == "买入" else "⚪"
        print(f"  {emoji} {sector}: {score:.3f} → {action}")

def cmd_result(args):
    """记录实际涨跌结果"""
    data = load_data()
    actuals = json.loads(args.actuals)
    
    # 找到对应日期的记录
    record = None
    for r in data["records"]:
        if r["date"] == args.date:
            record = r
            break
    
    if not record:
        print(f"❌ 未找到 {args.date} 的预测记录，请先运行 predict")
        return
    
    record["actuals"] = actuals
    
    # 计算假设跟随ML的收益
    threshold_buy = data["config"]["threshold_buy"]
    total_pnl = 0
    count = 0
    
    for sector, actual_return in actuals.items():
        pred_score = record["predictions"].get(sector, 0.5)
        action = record["action"].get(sector, "观望")
        
        if action == "买入":
            # 买入后实际收益
            pnl = actual_return / 100  # 百分比转小数
            record["pnl"][sector] = pnl
            total_pnl += pnl
            count += 1
        else:
            record["pnl"][sector] = 0  # 观望不操作，收益为0
    
    record["daily_pnl"] = total_pnl
    record["trades_count"] = count
    
    save_data(data)
    
    # 输出结果
    print(f"\n📈 {args.date} 实际结果:")
    for sector in SECTORS:
        pred = record["predictions"].get(sector, 0.5)
        actual = actuals.get(sector, 0)
        action = record["action"].get(sector, "观望")
        pnl = record["pnl"].get(sector, 0)
        
        # 判断方向是否正确
        pred_direction = "涨" if pred > 0.5 else "跌"
        actual_direction = "涨" if actual > 0 else "跌" if actual < 0 else "平"
        correct = "✅" if (pred_direction == actual_direction) or (abs(pred - 0.5) < 0.05) else "❌"
        
        print(f"  {correct} {sector}: 预测{pred:.3f}({pred_direction}) → 实际{actual:+.2f}% | 操作:{action} | 盈亏:{pnl:+.4f}")
    
    print(f"\n  当日总盈亏: {total_pnl:+.4f} ({count}笔交易)")

def cmd_stats(args):
    """查看统计"""
    data = load_data()
    records = data["records"]
    
    if not records:
        print("暂无记录")
        return
    
    # 过滤有实际结果的记录
    completed = [r for r in records if r.get("actuals")]
    
    if not completed:
        print(f"共 {len(records)} 天预测记录，暂无实际结果")
        return
    
    # 计算统计
    total_pnl = sum(r.get("daily_pnl", 0) for r in completed)
    total_trades = sum(r.get("trades_count", 0) for r in completed)
    
    # 方向准确率
    correct = 0
    total_preds = 0
    for r in completed:
        for sector in SECTORS:
            pred = r["predictions"].get(sector, 0.5)
            actual = r["actuals"].get(sector, 0)
            if abs(pred - 0.5) >= 0.05:  # 只算有明确方向的预测
                total_preds += 1
                pred_up = pred > 0.5
                actual_up = actual > 0
                if pred_up == actual_up:
                    correct += 1
    
    accuracy = correct / total_preds * 100 if total_preds > 0 else 0
    
    # 最大单日盈亏
    daily_pnls = [r.get("daily_pnl", 0) for r in completed]
    max_gain = max(daily_pnls) if daily_pnls else 0
    max_loss = min(daily_pnls) if daily_pnls else 0
    
    print(f"═══ QuantPilot ML预测追踪统计 ═══")
    print(f"追踪天数: {len(completed)}/{len(records)}")
    print(f"总交易笔数: {total_trades}")
    print(f"方向准确率: {accuracy:.1f}% ({correct}/{total_preds})")
    print(f"累计收益: {total_pnl:+.4f} ({total_pnl*100:+.2f}%)")
    print(f"日均收益: {total_pnl/len(completed)*100:+.3f}%")
    print(f"最大单日盈利: {max_gain*100:+.2f}%")
    print(f"最大单日亏损: {max_loss*100:+.2f}%")
    print(f"═══════════════════════════════")

def cmd_export(args):
    """导出宣发数据"""
    data = load_data()
    records = data["records"]
    completed = [r for r in records if r.get("actuals")]
    
    if not completed:
        print("暂无足够数据导出")
        return
    
    total_pnl = sum(r.get("daily_pnl", 0) for r in completed)
    total_trades = sum(r.get("trades_count", 0) for r in completed)
    
    correct = 0
    total_preds = 0
    for r in completed:
        for sector in SECTORS:
            pred = r["predictions"].get(sector, 0.5)
            actual = r["actuals"].get(sector, 0)
            if abs(pred - 0.5) >= 0.05:
                total_preds += 1
                if (pred > 0.5) == (actual > 0):
                    correct += 1
    
    accuracy = correct / total_preds * 100 if total_preds > 0 else 0
    
    export = {
        "period": f"{completed[0]['date']} ~ {completed[-1]['date']}",
        "trading_days": len(completed),
        "total_trades": total_trades,
        "direction_accuracy": f"{accuracy:.1f}%",
        "cumulative_return": f"{total_pnl*100:+.2f}%",
        "daily_avg_return": f"{total_pnl/len(completed)*100:+.3f}%",
        "records": completed
    }
    
    export_path = STATE_DIR / "ml_tracker_export.json"
    export_path.write_text(json.dumps(export, ensure_ascii=False, indent=2))
    print(f"导出至 {export_path}")
    print(json.dumps(export, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    
    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--date", required=True)
    p_pred.add_argument("--signals", required=True, help='JSON: {"芯片/半导体": 0.62, ...}')
    
    p_result = sub.add_parser("result")
    p_result.add_argument("--date", required=True)
    p_result.add_argument("--actuals", required=True, help='JSON: {"芯片/半导体": 1.2, ...}')
    
    p_stats = sub.add_parser("stats")
    p_export = sub.add_parser("export")
    
    args = parser.parse_args()
    if args.cmd == "predict": cmd_predict(args)
    elif args.cmd == "result": cmd_result(args)
    elif args.cmd == "stats": cmd_stats(args)
    elif args.cmd == "export": cmd_export(args)
    else: parser.print_help()
