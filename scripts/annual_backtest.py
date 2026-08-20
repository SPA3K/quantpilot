"""
QuantPilot 多Level年报期回测
预测: 年报发布前(1月初) → 验证: 年报发布后1个月(4月底)
样本: 30只沪深300 × 4年(2023-2026) = 120个预测点
模型: L0因子 + L2-GRU + L2-LSTM + L3情感
"""

import sys, os, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import baostock as bs
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = Path.home() / 'workspace' / 'quantpilot' / 'models' / 'trained'

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════

# 30只沪深300代表性股票（覆盖金融/消费/科技/制造/能源）
STOCKS = [
    '600519', '000858', '601318', '300750', '002594', '600900',
    '600036', '601166', '600276', '000333', '002415', '601888',
    '600030', '000651', '601398', '600000', '000001', '600887',
    '002714', '601012', '300059', '002304', '600309', '000725',
    '601899', '600585', '002352', '601088', '000568', '600104',
]

NAMES = {
    '600519': '贵州茅台', '000858': '五粮液', '601318': '中国平安',
    '300750': '宁德时代', '002594': '比亚迪', '600900': '长江电力',
    '600036': '招商银行', '601166': '兴业银行', '600276': '恒瑞医药',
    '000333': '美的集团', '002415': '海康威视', '601888': '中国中免',
    '600030': '中信证券', '000651': '格力电器', '601398': '工商银行',
    '600000': '浦发银行', '000001': '平安银行', '600887': '伊利股份',
    '002714': '牧原股份', '601012': '隆基绿能', '300059': '东方财富',
    '002304': '洋河股份', '600309': '万华化学', '000725': '京东方A',
    '601899': '紫金矿业', '600585': '海螺水泥', '002352': '顺丰控股',
    '601088': '中国神华', '000568': '泸州老窖', '600104': '上汽集团',
}

# 年报回测年份：预测1月初，验证到4月底
YEARS = [2023, 2024, 2025, 2026]

# ═══════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════

def load_stock(code: str, start: str, end: str) -> pd.DataFrame:
    """从baostock加载日线数据"""
    bs_code = f"sh.{code}" if code.startswith('6') else f"sz.{code}"
    rs = bs.query_history_k_data_plus(
        bs_code, "date,open,high,low,close,volume,amount",
        start_date=start, end_date=end, frequency="d", adjustflag="2"
    )
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
    df = pd.DataFrame(rows, columns=cols)
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    return df


# ═══════════════════════════════════════════
# L0 因子模型
# ═══════════════════════════════════════════

def compute_l0_score(df: pd.DataFrame) -> float:
    """L0技术因子评分"""
    if df.empty or len(df) < 30:
        return 0.0
    close = df['close'].values
    ret = np.diff(np.log(close))
    
    scores = []
    # 20日动量
    mom_20 = close[-1] / close[-21] - 1 if len(close) > 20 else 0
    scores.append(np.tanh(mom_20 * 5))
    # RSI
    delta = np.diff(close[-15:])
    gain = np.mean(np.maximum(delta, 0))
    loss = np.mean(np.abs(np.minimum(delta, 0)))
    rsi = 100 - 100 / (1 + gain / (loss + 1e-10)) if loss > 0 else 50
    scores.append((50 - rsi) / 50 * 0.5)
    # 波动率（低波为正）
    vol = np.std(ret[-20:]) * np.sqrt(252) if len(ret) > 20 else 0
    scores.append(-np.tanh(vol - 0.3))
    
    return float(np.tanh(np.mean(scores)))


# ═══════════════════════════════════════════
# L2 深度模型（Qlib Alpha360格式）
# ═══════════════════════════════════════════

_gru_model = None
_lstm_model = None

def get_gru():
    global _gru_model
    if _gru_model is None:
        import torch
        from quantpilot.ml.deep_models import StreamingDeepModel
        _gru_model = StreamingDeepModel(model_type='gru')
        _gru_model.load_model()
    return _gru_model

# ═══════════════ L1 LightGBM模型 ═══════════════

_lgb_model = None
_lgb_features = None

def get_l1():
    global _lgb_model, _lgb_features
    if _lgb_model is None:
        import pickle
        with open(MODEL_DIR / 'lgb_cs_2008_2022.pkl', 'rb') as f:
            data = pickle.load(f)
        _lgb_model = data['model']
        _lgb_features = data['feature_cols']
    return _lgb_model, _lgb_features

def compute_l1_score(df: pd.DataFrame) -> float:
    """L1 LightGBM因子评分"""
    if df.empty or len(df) < 60:
        return 0.0
    try:
        model, feature_cols = get_l1()
        df = df.sort_values('date').copy()
        c = df['close']; o = df['open']; h = df['high']; l = df['low']
        v = df['volume']; amt = df['amount'] if 'amount' in df.columns else c * v
        
        # 计算和训练时完全一样的因子
        feat = {}
        feat['ret_1'] = c.pct_change(1).iloc[-1]
        feat['ret_5'] = c.pct_change(5).iloc[-1]
        feat['ret_10'] = c.pct_change(10).iloc[-1]
        feat['ret_20'] = c.pct_change(20).iloc[-1]
        feat['mom_5'] = (c.iloc[-1] / c.iloc[-6] - 1) if len(c) > 5 else 0
        feat['mom_10'] = (c.iloc[-1] / c.iloc[-11] - 1) if len(c) > 10 else 0
        feat['mom_20'] = (c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 20 else 0
        feat['mom_60'] = (c.iloc[-1] / c.iloc[-61] - 1) if len(c) > 60 else 0
        ret1 = c.pct_change(1)
        feat['vol_5'] = ret1.rolling(5).std().iloc[-1] * np.sqrt(252) if len(ret1) > 5 else 0
        feat['vol_20'] = ret1.rolling(20).std().iloc[-1] * np.sqrt(252) if len(ret1) > 20 else 0
        feat['vol_60'] = ret1.rolling(60).std().iloc[-1] * np.sqrt(252) if len(ret1) > 60 else 0
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - 100 / (1 + gain / (loss + 1e-10))
        feat['rsi_14'] = rsi.iloc[-1] if len(rsi) > 14 else 50
        feat['ma5_dev'] = (c.iloc[-1] / c.rolling(5).mean().iloc[-1] - 1) if len(c) > 5 else 0
        feat['ma10_dev'] = (c.iloc[-1] / c.rolling(10).mean().iloc[-1] - 1) if len(c) > 10 else 0
        feat['ma20_dev'] = (c.iloc[-1] / c.rolling(20).mean().iloc[-1] - 1) if len(c) > 20 else 0
        feat['ma60_dev'] = (c.iloc[-1] / c.rolling(60).mean().iloc[-1] - 1) if len(c) > 60 else 0
        ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
        feat['boll_pct'] = ((c.iloc[-1] - (ma20.iloc[-1] - 2*std20.iloc[-1])) / (4*std20.iloc[-1] + 1e-10)) if len(c) > 20 else 0.5
        feat['vol_ratio_5_20'] = (v.rolling(5).mean().iloc[-1] / (v.rolling(20).mean().iloc[-1] + 1e-10)) if len(v) > 20 else 1
        feat['vol_ratio_5_60'] = (v.rolling(5).mean().iloc[-1] / (v.rolling(60).mean().iloc[-1] + 1e-10)) if len(v) > 60 else 1
        if 'turn' in df.columns:
            turn = df['turn']
            feat['turn_5'] = turn.rolling(5).mean().iloc[-1] if len(turn) > 5 else 0
            feat['turn_20'] = turn.rolling(20).mean().iloc[-1] if len(turn) > 20 else 0
        else:
            feat['turn_5'] = 0; feat['turn_20'] = 0
        vwap = amt / (v + 1e-10)
        feat['vwap_dev'] = (vwap.iloc[-1] / c.iloc[-1] - 1) if c.iloc[-1] > 0 else 0
        
        X = pd.DataFrame([[feat.get(fc, 0) for fc in feature_cols]], columns=feature_cols)
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        score = float(model.predict(X)[0])
        return score
    except Exception as e:
        logger.warning(f"L1 failed: {e}")
        return 0.0


def get_lstm():
    global _lstm_model
    if _lstm_model is None:
        from quantpilot.ml.deep_models import StreamingDeepModel
        _lstm_model = StreamingDeepModel(model_type='lstm')
        _lstm_model.load_model()
    return _lstm_model

def compute_l2_scores(panel: pd.DataFrame, date: str) -> dict:
    """批量计算L2 GRU/LSTM分数（直接用原始logit）"""
    results = {}
    for model_type, getter in [('gru', get_gru), ('lstm', get_lstm)]:
        try:
            model = getter()
            pred = model.predict(panel, date)
            for _, row in pred.iterrows():
                code = row['code']
                if code not in results:
                    results[code] = {}
                # streaming模型返回原始logit，不做(0.5)*2映射
                results[code][model_type] = float(row['score'])
        except Exception as e:
            logger.warning(f"L2 {model_type} failed: {e}")
    return results


# ═══════════════════════════════════════════
# L3 情感因子（用技术面动量近似，避免API调用）
# ═══════════════════════════════════════════

def compute_l3_proxy(df: pd.DataFrame) -> float:
    """
    L3情感代理：用近期价格动量+成交量变化近似
    正式版应接入东方财富搜索API + StructBERT
    """
    if df.empty or len(df) < 20:
        return 0.0
    close = df['close'].values
    volume = df['volume'].values
    
    # 5日动量（短期情绪）
    mom_5 = close[-1] / close[-6] - 1 if len(close) > 5 else 0
    # 成交量变化（放量=关注度高）
    vol_chg = np.mean(volume[-5:]) / (np.mean(volume[-20:]) + 1e-10) - 1
    
    return float(np.tanh((mom_5 * 3 + vol_chg) / 2))


# ═══════════════════════════════════════════
# 主回测流程
# ═══════════════════════════════════════════

def main():
    print("=" * 70)
    print("  QuantPilot 多Level年报期回测")
    print("  预测: 1月初 | 验证: 4月底 | 预测窗口: ~4个月")
    print(f"  股票: {len(STOCKS)}只沪深300 | 年份: {YEARS}")
    print("=" * 70)
    
    # ── 预加载所有数据（一次登录，批量拉取）──
    logger.info("预加载所有股票数据...")
    bs.login()
    all_stock_data = {}  # {code: DataFrame(全量日线)}
    data_start = f"{min(YEARS)-1}-01-01"
    data_end = f"{max(YEARS)}-04-30"
    
    for i, code in enumerate(STOCKS):
        df = load_stock(code, data_start, data_end)
        if not df.empty and len(df) > 60:
            all_stock_data[code] = df
        if (i + 1) % 10 == 0:
            logger.info(f"  已加载 {i+1}/{len(STOCKS)}")
    bs.logout()
    logger.info(f"预加载完成: {len(all_stock_data)}/{len(STOCKS)} 只有效")
    
    all_results = []
    
    for year in YEARS:
        pred_start = f"{year}-01-02"       # 预测日（1月初）
        eval_end = f"{year}-04-30"         # 验证截止日
        
        print(f"\n{'═'*70}")
        print(f"  {year}年报回测 | 预测日: {pred_start} | 验证至: {eval_end}")
        print(f"{'═'*70}")
        
        # ── Step 1: 用预加载数据 ──
        stock_data = {}
        for code, df in all_stock_data.items():
            pred_df = df[df['date'] <= pred_start]
            if len(pred_df) > 60:
                stock_data[code] = df  # 传全量，各步骤自己截
        
        logger.info(f"有效股票: {len(stock_data)}/{len(STOCKS)}")
        
        # ── Step 2: L0因子评分 ──
        logger.info("L0 因子评分...")
        l0_scores = {}
        for code, df in stock_data.items():
            pred_df = df[df['date'] <= pred_start]
            if len(pred_df) > 30:
                l0_scores[code] = compute_l0_score(pred_df)
        
        # ── Step 2.5: L1 LightGBM评分 ──
        logger.info("L1 LightGBM评分...")
        l1_scores = {}
        for code, df in stock_data.items():
            pred_df = df[df['date'] <= pred_start]
            if len(pred_df) > 60:
                l1_scores[code] = compute_l1_score(pred_df)
        
        # ── Step 3: L2深度模型（跳过）──
        logger.info("L2 跳过（不靠谱）")
        l2_scores = {}
        
        # ── Step 4: L3情感代理 ──
        logger.info("L3 情感评分...")
        l3_scores = {}
        for code, df in stock_data.items():
            pred_df = df[df['date'] <= pred_start]
            l3_scores[code] = compute_l3_proxy(pred_df)
        
        # ── Step 5: 融合预测 ──
        logger.info("融合预测...")
        predictions = {}
        for code in stock_data:
            l0 = l0_scores.get(code, 0)
            l1 = l1_scores.get(code, 0)
            l3 = l3_scores.get(code, 0)
            
            # 融合: L1为主(70%) + L0为辅(20%) + L3微调(10%)
            fusion = 0.7 * l1 + 0.2 * l0 + 0.1 * l3
            
            # 计算实际收益（预测日→验证截止日）
            pred_df = stock_data[code]
            pred_close = pred_df[pred_df['date'] <= pred_start]['close'].iloc[-1] if not pred_df[pred_df['date'] <= pred_start].empty else None
            eval_close = pred_df[pred_df['date'] <= eval_end]['close'].iloc[-1] if not pred_df[pred_df['date'] <= eval_end].empty else None
            
            if pred_close and eval_close:
                actual_return = (eval_close / pred_close - 1) * 100
            else:
                actual_return = None
            
            predictions[code] = {
                'name': NAMES.get(code, code),
                'l0': round(l0, 3),
                'l1': round(l1, 3),
                'l3': round(l3, 3),
                'fusion': round(fusion, 3),
                'actual_return': round(actual_return, 2) if actual_return else None,
            }
        
        # ── Step 6: 评估 ──
        # 按融合分数排序，取TOP5做多 vs BOTTOM5做空
        valid = {k: v for k, v in predictions.items() if v['actual_return'] is not None}
        sorted_by_fusion = sorted(valid.items(), key=lambda x: x[1]['fusion'], reverse=True)
        
        top5 = sorted_by_fusion[:5]
        bottom5 = sorted_by_fusion[-5:]
        
        top5_ret = np.mean([v['actual_return'] for _, v in top5])
        bottom5_ret = np.mean([v['actual_return'] for _, v in bottom5])
        long_short = top5_ret - bottom5_ret
        all_ret = np.mean([v['actual_return'] for _, v in sorted_by_fusion])
        
        print(f"\n  {'─'*60}")
        print(f"  TOP5 (看多)")
        print(f"  {'─'*60}")
        for code, v in top5:
            print(f"    {v['name']:<8} fusion={v['fusion']:+.3f}  L0={v['l0']:+.3f}  L1={v['l1']:+.3f}  L3={v['l3']:+.3f}  实际={v['actual_return']:+.1f}%")
        
        print(f"\n  {'─'*60}")
        print(f"  BOTTOM5 (看空)")
        print(f"  {'─'*60}")
        for code, v in bottom5:
            print(f"    {v['name']:<8} fusion={v['fusion']:+.3f}  L0={v['l0']:+.3f}  L1={v['l1']:+.3f}  L3={v['l3']:+.3f}  实际={v['actual_return']:+.1f}%")
        
        print(f"\n  📊 {year}年报期汇总:")
        print(f"     TOP5平均收益:    {top5_ret:+.2f}%")
        print(f"     BOTTOM5平均收益: {bottom5_ret:+.2f}%")
        print(f"     多空收益(Long-Short): {long_short:+.2f}%")
        print(f"     全样本平均收益:  {all_ret:+.2f}%")
        
        all_results.append({
            'year': year,
            'top5_ret': round(top5_ret, 2),
            'bottom5_ret': round(bottom5_ret, 2),
            'long_short': round(long_short, 2),
            'all_ret': round(all_ret, 2),
            'predictions': predictions,
        })
    
    # ═══════════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  多年回测汇总")
    print(f"{'='*70}")
    print(f"  {'年份':<6} {'TOP5':>8} {'BOTTOM5':>10} {'Long-Short':>12} {'全样本':>8}")
    print(f"  {'─'*50}")
    
    for r in all_results:
        print(f"  {r['year']:<6} {r['top5_ret']:>+7.2f}% {r['bottom5_ret']:>+9.2f}% {r['long_short']:>+11.2f}% {r['all_ret']:>+7.2f}%")
    
    avg_ls = np.mean([r['long_short'] for r in all_results])
    avg_top = np.mean([r['top5_ret'] for r in all_results])
    print(f"\n  平均Long-Short: {avg_ls:+.2f}%")
    print(f"  平均TOP5收益:   {avg_top:+.2f}%")
    
    # 保存
    out_dir = Path.home() / 'workspace' / 'quantpilot' / 'eval_results'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'annual_backtest_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已保存: {out_path}")


if __name__ == '__main__':
    main()
