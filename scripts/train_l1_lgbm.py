"""
QuantPilot L1 LightGBM 训练脚本
数据: 2008-2022年A股日线 → 因子面板 → 截面预测
"""

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path.home() / 'workspace' / 'quantpilot' / 'data' / 'ml' / 'train_2008_2022'
MODEL_DIR = Path.home() / 'workspace' / 'quantpilot' / 'models' / 'trained'
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════ 因子计算 ═══════════════

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算全部因子（和技术因子）"""
    df = df.sort_values('date').copy()
    c = df['close']; o = df['open']; h = df['high']; l = df['low']
    v = df['volume']; amt = df['amount']
    
    # 收益率
    df['ret_1'] = c.pct_change(1)
    df['ret_5'] = c.pct_change(5)
    df['ret_10'] = c.pct_change(10)
    df['ret_20'] = c.pct_change(20)
    
    # 动量
    df['mom_5'] = c / c.shift(5) - 1
    df['mom_10'] = c / c.shift(10) - 1
    df['mom_20'] = c / c.shift(20) - 1
    df['mom_60'] = c / c.shift(60) - 1
    
    # 波动率
    df['vol_5'] = df['ret_1'].rolling(5).std() * np.sqrt(252)
    df['vol_20'] = df['ret_1'].rolling(20).std() * np.sqrt(252)
    df['vol_60'] = df['ret_1'].rolling(60).std() * np.sqrt(252)
    
    # RSI
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi_14'] = 100 - 100 / (1 + gain / (loss + 1e-10))
    
    # 均线偏离
    df['ma5_dev'] = c / c.rolling(5).mean() - 1
    df['ma10_dev'] = c / c.rolling(10).mean() - 1
    df['ma20_dev'] = c / c.rolling(20).mean() - 1
    df['ma60_dev'] = c / c.rolling(60).mean() - 1
    
    # 布林带
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df['boll_pct'] = (c - (ma20 - 2*std20)) / (4*std20 + 1e-10)
    
    # 成交量
    df['vol_ratio_5_20'] = v.rolling(5).mean() / (v.rolling(20).mean() + 1e-10)
    df['vol_ratio_5_60'] = v.rolling(5).mean() / (v.rolling(60).mean() + 1e-10)
    
    # 换手率
    if 'turn' in df.columns:
        df['turn_5'] = df['turn'].rolling(5).mean()
        df['turn_20'] = df['turn'].rolling(20).mean()
    
    # 估值
    if 'pe' in df.columns:
        df['ep'] = 1.0 / (df['pe'].abs() + 1e-10)
        df['ep'] = df['ep'].where(df['pe'] > 0, 0)
    if 'pb' in df.columns:
        df['bp'] = 1.0 / (df['pb'].abs() + 1e-10)
        df['bp'] = df['bp'].where(df['pb'] > 0, 0)
    
    # VWAP偏离
    if 'amount' in df.columns:
        vwap = amt / (v + 1e-10)
        df['vwap_dev'] = vwap / c - 1
    
    # 标签: 未来20日收益
    df['fwd_ret_20'] = c.shift(-20) / c - 1
    
    return df


def get_feature_cols(df):
    """获取特征列"""
    exclude = {'date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount',
               'turn', 'pe', 'pb', 'ps', 'fwd_ret_20'}
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64']]


# ═══════════════ 数据加载 ═══════════════

def load_all_data():
    """加载全部训练数据"""
    logger.info(f"加载数据: {DATA_DIR}")
    files = sorted(DATA_DIR.glob('*.parquet'))
    logger.info(f"  文件数: {len(files)}")
    
    all_dfs = []
    for i, f in enumerate(files):
        try:
            df = pd.read_parquet(f)
            if len(df) < 200:  # 至少200天数据
                continue
            df = compute_features(df)
            all_dfs.append(df)
        except Exception as e:
            pass
        
        if (i+1) % 500 == 0:
            logger.info(f"  已处理 {i+1}/{len(files)}")
    
    panel = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"  总数据: {len(panel)} 行, {panel['code'].nunique()} 只股票")
    return panel


# ═══════════════ 训练 ═══════════════

def train_lightgbm(panel: pd.DataFrame):
    """训练LightGBM截面模型"""
    import lightgbm as lgb
    
    feature_cols = get_feature_cols(panel)
    logger.info(f"特征数: {len(feature_cols)}")
    logger.info(f"特征列: {feature_cols}")
    
    # 清理
    panel = panel.dropna(subset=['fwd_ret_20'])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    
    # 截面排名标签（每月截面内排名）
    panel['ym'] = panel['date'].dt.to_period('M')
    panel['label'] = panel.groupby('ym')['fwd_ret_20'].transform(
        lambda x: x.rank(pct=True)
    )
    
    # 时间分割
    train_mask = panel['date'] < '2019-01-01'
    valid_mask = (panel['date'] >= '2019-01-01') & (panel['date'] < '2020-01-01')
    test_mask = panel['date'] >= '2020-01-01'
    
    X_train = panel.loc[train_mask, feature_cols].fillna(0)
    y_train = panel.loc[train_mask, 'label']
    X_valid = panel.loc[valid_mask, feature_cols].fillna(0)
    y_valid = panel.loc[valid_mask, 'label']
    X_test = panel.loc[test_mask, feature_cols].fillna(0)
    y_test = panel.loc[test_mask, 'label']
    
    logger.info(f"训练集: {len(X_train)} | 验证集: {len(X_valid)} | 测试集: {len(X_test)}")
    
    # 训练
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    
    # 评估
    from scipy.stats import spearmanr
    
    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)
    
    # Rank-IC
    train_ic = spearmanr(pred_train, y_train)[0]
    valid_ic = spearmanr(pred_valid, y_valid)[0]
    test_ic = spearmanr(pred_test, y_test)[0]
    
    logger.info(f"训练IC: {train_ic:.4f}")
    logger.info(f"验证IC: {valid_ic:.4f}")
    logger.info(f"测试IC: {test_ic:.4f}")
    
    # 特征重要性
    importance = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    logger.info("TOP10特征:")
    for name, imp in feat_imp[:10]:
        logger.info(f"  {name}: {imp:.0f}")
    
    # 保存
    model_path = MODEL_DIR / 'lgb_cs_2008_2022.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'params': params,
            'metrics': {
                'train_ic': train_ic,
                'valid_ic': valid_ic,
                'test_ic': test_ic,
            }
        }, f)
    logger.info(f"模型已保存: {model_path}")
    
    return model, feature_cols


# ═══════════════ 主流程 ═══════════════

if __name__ == '__main__':
    panel = load_all_data()
    model, features = train_lightgbm(panel)
    print("L1 LightGBM 训练完成!")
