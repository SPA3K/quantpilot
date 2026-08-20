"""
QuantPilot 多Level联合评测
L0(因子) + L1(树模型) + L2(深度) + L3(情感) → 融合预测 → 对比实际股价

用法: python scripts/multilevel_eval.py
"""

import sys, os, json, time, pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/workspace/quantpilot/src'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════

STOCKS = {
    '300750': {'name': '宁德时代', 'report_date': '2026-07-25', 'result': '利润432亿同比大增'},
    '600519': {'name': '贵州茅台', 'report_date': '2026-08-15', 'result': '利润下降1.95%'},
    '688256': {'name': '寒武纪',   'report_date': '2026-08-07', 'result': '利润23亿同比+122%'},
    '600900': {'name': '长江电力', 'report_date': '2026-07-16', 'result': '发电量+4.81%'},
}

# 股价变动（已获取）
PRICE_MOVES = {
    '300750': {'chg_1d': +4.44, 'chg_5d': +3.21},
    '600519': {'chg_1d': -3.64, 'chg_5d': None},
    '688256': {'chg_1d': -6.33, 'chg_5d': -8.91},
    '600900': {'chg_1d': -1.53, 'chg_5d': +4.35},
}

DATA_DIR = Path.home() / 'workspace' / 'quantpilot' / 'data' / 'ml' / 'daily'


# ═══════════════════════════════════════════
# L0: 因子模型评分
# ═══════════════════════════════════════════

def load_stock_data(code: str, start: str, end: str) -> pd.DataFrame:
    """加载单只股票的日线数据，本地没有则从baostock拉"""
    # 尝试本地文件
    for fname in [f'{code}.parquet', f'sh_{code}.parquet', f'sz_{code}.parquet']:
        fpath = DATA_DIR / fname
        if fpath.exists():
            df = pd.read_parquet(fpath)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                mask = (df['date'] >= start) & (df['date'] <= end)
                return df[mask].copy()
            return df
    
    # 本地没有，从baostock拉
    try:
        import baostock as bs
        bs.login()
        bs_code = f"sh.{code}" if code.startswith('6') else f"sz.{code}"
        rs = bs.query_history_k_data_plus(
            bs_code, "date,open,high,low,close,volume,amount,pe,pb,ps",
            start_date=start, end_date=end, frequency="d", adjustflag="2"
        )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        bs.logout()
        
        if not rows:
            return pd.DataFrame()
        
        cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pe', 'pb', 'ps']
        df = pd.DataFrame(rows, columns=cols)
        for c in cols[1:]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        logger.warning(f"baostock fetch failed for {code}: {e}")
        return pd.DataFrame()


def compute_factors(df: pd.DataFrame) -> dict:
    """计算单只股票的技术因子"""
    if df.empty or len(df) < 30:
        return {}
    
    close = df['close'].values
    volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
    
    factors = {}
    
    # 动量
    factors['mom_5'] = (close[-1] / close[-6] - 1) if len(close) > 5 else 0
    factors['mom_10'] = (close[-1] / close[-11] - 1) if len(close) > 10 else 0
    factors['mom_20'] = (close[-1] / close[-21] - 1) if len(close) > 20 else 0
    
    # 波动率
    ret = np.diff(np.log(close))
    factors['vol_20'] = np.std(ret[-20:]) * np.sqrt(252) if len(ret) > 20 else 0
    factors['vol_60'] = np.std(ret[-60:]) * np.sqrt(252) if len(ret) > 60 else 0
    
    # RSI
    delta = np.diff(close)
    gain = np.mean(np.maximum(delta[-14:], 0))
    loss = np.mean(np.abs(np.minimum(delta[-14:], 0)))
    factors['rsi_14'] = 100 - 100 / (1 + gain / (loss + 1e-10)) if loss > 0 else 50
    
    # 均线偏离
    factors['ma5_dev'] = (close[-1] / np.mean(close[-5:]) - 1) if len(close) > 5 else 0
    factors['ma20_dev'] = (close[-1] / np.mean(close[-20:]) - 1) if len(close) > 20 else 0
    
    # 成交量变化
    if len(volume) > 20:
        vol_ratio = np.mean(volume[-5:]) / (np.mean(volume[-20:]) + 1e-10)
        factors['vol_ratio'] = vol_ratio
    
    # 布林带位置
    if len(close) > 20:
        ma20 = np.mean(close[-20:])
        std20 = np.std(close[-20:])
        factors['boll_pct'] = (close[-1] - (ma20 - 2*std20)) / (4*std20 + 1e-10)
    
    # 估值（如果有）
    if 'pe' in df.columns:
        pe = df['pe'].values[-1]
        factors['ep'] = 1.0 / (abs(pe) + 1e-10) if pe > 0 else 0
    if 'pb' in df.columns:
        pb = df['pb'].values[-1]
        factors['bp'] = 1.0 / (abs(pb) + 1e-10) if pb > 0 else 0
    
    return factors


def l0_score(factors: dict) -> float:
    """
    L0因子打分：综合技术面评分
    返回: -1(极度看空) 到 +1(极度看多)
    """
    scores = []
    
    # 动量信号
    mom_20 = factors.get('mom_20', 0)
    scores.append(np.tanh(mom_20 * 5))  # 20日动量
    
    # RSI信号 (超买超卖)
    rsi = factors.get('rsi_14', 50)
    rsi_signal = (50 - rsi) / 50  # RSI>50为负面(超买), <50为正面(超卖)
    scores.append(rsi_signal * 0.5)
    
    # 均线偏离
    ma20_dev = factors.get('ma20_dev', 0)
    scores.append(np.tanh(ma20_dev * 10))
    
    # 波动率 (低波动为正面)
    vol = factors.get('vol_20', 0)
    scores.append(-np.tanh(vol - 0.3))  # 年化30%为基准
    
    # 估值
    ep = factors.get('ep', 0)
    if ep > 0:
        scores.append(np.tanh(ep * 20))
    
    return float(np.mean(scores))


# ═══════════════════════════════════════════
# L1: 树模型（用因子截面排名近似）
# ═══════════════════════════════════════════

def l1_score(factors: dict, all_factors: dict) -> float:
    """
    L1树模型近似：用因子截面排名模拟LightGBM
    在4只股票中排名，转化为标准化分数
    """
    # 关键因子
    key_factors = ['mom_20', 'vol_20', 'rsi_14', 'ma20_dev', 'vol_ratio']
    
    ranks = []
    for f in key_factors:
        vals = {code: af.get(f, 0) for code, af in all_factors.items()}
        if not vals:
            continue
        sorted_codes = sorted(vals.keys(), key=lambda c: vals[c])
        rank = sorted_codes.index(list(vals.keys())[0]) if list(vals.keys())[0] in sorted_codes else 2
        # 对于波动率，低的更好
        if f == 'vol_20':
            rank = len(sorted_codes) - 1 - rank
        ranks.append(rank / (len(sorted_codes) - 1 + 1e-10))
    
    # 转化为 -1 到 +1
    avg_rank = np.mean(ranks) if ranks else 0.5
    return float((avg_rank - 0.5) * 2)


# ═══════════════════════════════════════════
# L2: 深度模型
# ═══════════════════════════════════════════

def l2_score_batch(codes: list, pred_date: str, panel: pd.DataFrame) -> dict:
    """L2 GRU/LSTM预训练模型 — 批量推理，cross-stock归一化"""
    results = {code: {'gru': 0.0, 'lstm': 0.0} for code in codes}
    
    try:
        import torch
        from quantpilot.ml.deep_models import PretrainedDeepModel
        
        for model_type in ['gru', 'lstm']:
            try:
                model = PretrainedDeepModel(model_type=model_type)
                model.load_model()
                
                # 用模型自带的predict方法（内部做cross-stock归一化）
                pred = model.predict(panel, pred_date)
                
                if pred.empty:
                    logger.warning(f"L2 {model_type}: empty predictions for date {pred_date}")
                    continue
                
                for code in codes:
                    score_row = pred[pred['code'] == code]
                    if not score_row.empty:
                        raw_score = float(score_row['score'].values[0])
                        # 模型已归一化到0-1，转为 -1 到 +1
                        results[code][model_type] = (raw_score - 0.5) * 2
                    else:
                        logger.warning(f"L2 {model_type}: {code} not in predictions")
            except Exception as e:
                logger.warning(f"L2 {model_type} batch failed: {e}")
    except Exception as e:
        logger.warning(f"L2 import failed: {e}")
    
    return results


# ═══════════════════════════════════════════
# L3: 情感因子（已计算）
# ═══════════════════════════════════════════

# 从之前的评测结果读取
L3_SCORES = {
    '300750': +0.597,  # 宁德时代：财报前看涨
    '600519': -0.023,  # 茅台：财报前中性
    '688256': +0.277,  # 寒武纪：财报前看涨
    '600900': +0.374,  # 长江电力：财报前看涨
}


# ═══════════════════════════════════════════
# 融合决策
# ═══════════════════════════════════════════

def fusion_predict(l0, l1_gru, l1_lstm, l3, weights=None):
    """
    多Level加权融合
    weights: [L0, L1_gru, L1_lstm, L3] 默认等权
    """
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]
    
    scores = [l0, l1_gru, l1_lstm, l3]
    fused = sum(w * s for w, s in zip(weights, scores))
    
    # 判定方向
    if fused > 0.1:
        direction = "看涨 ↑"
    elif fused < -0.1:
        direction = "看跌 ↓"
    else:
        direction = "中性 →"
    
    return fused, direction


# ═══════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════

def main():
    print("="*70)
    print(f"  QuantPilot 多Level联合评测")
    print(f"  L0(因子) + L2(GRU/LSTM) + L3(情感) → 融合预测")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)
    
    # ── Step 1: 加载数据 & 计算因子 ──
    logger.info("Step 1: 加载数据 & 计算因子")
    all_factors = {}
    all_panels = {}
    
    for code, info in STOCKS.items():
        report_date = info['report_date']
        # 用财报前365天的数据（确保60天序列+充足余量）
        start = (datetime.strptime(report_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        end = (datetime.strptime(report_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        
        df = load_stock_data(code, start, end)
        if df.empty:
            logger.warning(f"No data for {code}")
            continue
        
        # 计算vwap_dev（L2模型需要）
        if 'amount' in df.columns and 'volume' in df.columns:
            df['vwap_dev'] = (df['amount'] / (df['volume'] + 1e-10)) / (df['close'] + 1e-10)
        elif 'vwap_dev' not in df.columns:
            df['vwap_dev'] = 1.0  # fallback
        
        factors = compute_factors(df)
        all_factors[code] = factors
        
        # 构建panel用于L2
        df['code'] = code
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        all_panels[code] = df
        
        print(f"  {info['name']}({code}): {len(df)}天数据, {len(factors)}个因子")
    
    # ── Step 2: L0因子评分 ──
    logger.info("Step 2: L0 因子评分")
    l0_scores = {}
    for code, factors in all_factors.items():
        l0_scores[code] = l0_score(factors)
    
    # ── Step 3: L2深度模型（逐只推理，各自用最后交易日）──
    logger.info("Step 3: L2 GRU/LSTM推理")
    
    l2_scores = {}
    for code in STOCKS:
        if code not in all_panels:
            l2_scores[code] = {'gru': 0.0, 'lstm': 0.0}
            continue
        
        # 每只股票用自己的最后交易日
        stock_panel = all_panels[code]
        stock_last_date = str(stock_panel['date'].max().date())
        
        # 单只股票推理（模型会归一化到0.5，需要手动处理）
        stock_result = {'gru': 0.0, 'lstm': 0.0}
        try:
            from quantpilot.ml.deep_models import PretrainedDeepModel
            import torch
            
            for model_type in ['gru', 'lstm']:
                model = PretrainedDeepModel(model_type=model_type)
                model.load_model()
                
                # 手动准备序列并推理（绕过min-max归一化）
                dates = stock_panel[stock_panel['date'] <= pd.Timestamp(stock_last_date)]['date'].unique()
                dates = sorted(dates)[-model.seq_len:]
                
                if len(dates) < model.seq_len:
                    continue
                
                stock_data = stock_panel.sort_values('date')
                stock_data = stock_data[stock_data['date'].isin(dates)]
                
                if len(stock_data) < model.seq_len:
                    continue
                
                features = stock_data[model.feature_cols].values[-model.seq_len:]
                features = np.nan_to_num(features, nan=0.0)
                
                current_close = features[-1, 3]
                current_volume = features[-1, 4]
                
                if current_close > 0 and current_volume > 0:
                    for i in [0, 1, 2, 3, 5]:
                        features[:, i] = features[:, i] / current_close
                    features[:, 4] = features[:, 4] / current_volume
                
                x = torch.FloatTensor(features).unsqueeze(0).to(model.device)
                with torch.no_grad():
                    raw_score = model.model(x).cpu().item()
                
                # 用tanh压缩到[-1, +1]
                stock_result[model_type] = float(np.tanh(raw_score / 20))
        except Exception as e:
            logger.warning(f"L2 failed for {code}: {e}")
        
        l2_scores[code] = stock_result
    
    # ── Step 4: 融合 & 输出 ──
    logger.info("Step 4: 多Level融合")
    
    print(f"\n{'='*70}")
    print(f"  多Level评分详情")
    print(f"{'='*70}")
    print(f"  {'股票':<8} {'L0因子':>8} {'L2-GRU':>8} {'L2-LSTM':>8} {'L3情感':>8} {'融合':>8} {'预测':>8} {'实际1日':>8} {'判定':>4}")
    print(f"  {'─'*72}")
    
    results = {}
    for code in STOCKS:
        info = STOCKS[code]
        l0 = l0_scores.get(code, 0)
        l2g = l2_scores.get(code, {}).get('gru', 0)
        l2l = l2_scores.get(code, {}).get('lstm', 0)
        l3 = L3_SCORES.get(code, 0)
        
        fused, direction = fusion_predict(l0, l2g, l2l, l3)
        
        chg = PRICE_MOVES[code]['chg_1d']
        actual = f"{chg:+.2f}%"
        
        # 判定
        pred_up = fused > 0.05
        actual_up = chg > 0.5
        correct = "✅" if pred_up == actual_up else "❌"
        
        print(f"  {info['name']:<6} {l0:>+8.3f} {l2g:>+8.3f} {l2l:>+8.3f} {l3:>+8.3f} {fused:>+8.3f} {direction:>8} {actual:>8} {correct:>4}")
        
        results[code] = {
            'name': info['name'],
            'L0_factor': round(l0, 3),
            'L2_gru': round(l2g, 3),
            'L2_lstm': round(l2l, 3),
            'L3_sentiment': round(l3, 3),
            'fusion': round(fused, 3),
            'direction': direction,
            'actual_1d': chg,
            'correct': correct == '✅',
        }
    
    # ── 对比：单L3 vs 多Level ──
    print(f"\n{'='*70}")
    print(f"  对比: 单L3 vs 多Level融合")
    print(f"{'='*70}")
    
    l3_correct = 0
    fusion_correct = 0
    total = 0
    
    for code, r in results.items():
        chg = PRICE_MOVES[code]['chg_1d']
        actual_up = chg > 0.5
        
        l3_pred = L3_SCORES[code] > 0.05
        fusion_pred = r['fusion'] > 0.05
        
        l3_ok = l3_pred == actual_up
        fusion_ok = fusion_pred == actual_up
        
        l3_correct += l3_ok
        fusion_correct += fusion_ok
        total += 1
        
        print(f"  {r['name']:<6} L3:{'✅' if l3_ok else '❌'}  融合:{'✅' if fusion_ok else '❌'}  实际:{chg:+.2f}%")
    
    print(f"\n  单L3准确率: {l3_correct}/{total} ({l3_correct/total*100:.0f}%)")
    print(f"  多Level准确率: {fusion_correct}/{total} ({fusion_correct/total*100:.0f}%)")
    
    # ── 保存 ──
    out_dir = Path.home() / 'workspace' / 'quantpilot' / 'eval_results'
    out_dir.mkdir(exist_ok=True)
    
    report = {
        'meta': {
            'date': datetime.now().isoformat(),
            'type': 'multi_level_earnings_prediction',
            'models': ['L0_factor', 'L2_gru', 'L2_lstm', 'L3_sentiment'],
            'fusion_weights': [0.25, 0.25, 0.25, 0.25],
        },
        'results': results,
        'accuracy': {
            'L3_only': f"{l3_correct}/{total}",
            'fusion': f"{fusion_correct}/{total}",
        }
    }
    
    out_path = out_dir / f'multilevel_eval_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n  报告已保存: {out_path}")


if __name__ == '__main__':
    main()
