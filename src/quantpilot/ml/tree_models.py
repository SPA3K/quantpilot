"""
QuantPilot Tree Models — LightGBM/XGBoost/CatBoost截面模型
用A股数据训练，月度截面预测
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional
from quantpilot.ml.model_zoo import BaseModel, register_model

logger = logging.getLogger(__name__)


class LightGBMCrossSection(BaseModel):
    """
    LightGBM截面模型
    每月用历史数据训练，预测下月收益排名
    """
    name = "lightgbm_cs"
    category = "tree"
    description = "LightGBM截面模型，月度滚动训练"
    requires_training = True

    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, min_child_samples=50):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_samples = min_child_samples
        self.model = None
        self.feature_cols = None

    def _get_features(self, df: pd.DataFrame) -> list:
        """获取特征列（排除date, code, close, open, high, low, volume, fwd_ret等）"""
        exclude = {'date', 'code', 'close', 'open', 'high', 'low', 'volume',
                   'fwd_ret', 'ym', 'isST', 'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d'}
        return [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    def _prepare_data(self, panel: pd.DataFrame, train_end: str, lookback_months: int = 12):
        """准备训练数据：用过去N个月的数据"""
        train_end_dt = pd.Timestamp(train_end)
        train_start_dt = train_end_dt - pd.DateOffset(months=lookback_months)

        mask = (panel['date'] >= train_start_dt) & (panel['date'] < train_end_dt)
        train_data = panel[mask].copy()

        # 标签：未来20日收益的截面排名（百分位）
        if 'fwd_ret' not in train_data.columns:
            train_data['fwd_ret'] = train_data.groupby('code')['close'].transform(
                lambda x: x.shift(-20) / x - 1
            )

        train_data = train_data.dropna(subset=['fwd_ret'])
        return train_data

    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        预测：先训练，再预测
        """
        import lightgbm as lgb

        date = pd.Timestamp(date)
        df = factor_panel[factor_panel['date'] == date].copy()

        if df.empty:
            return pd.DataFrame(columns=['date', 'code', 'score'])

        feature_cols = self._get_features(factor_panel)
        self.feature_cols = feature_cols

        # 准备训练数据
        train_data = self._prepare_data(factor_panel, date, lookback_months=12)

        if len(train_data) < 1000:
            logger.warning(f"Insufficient training data: {len(train_data)}")
            df['score'] = 0.5
            return df[['date', 'code', 'score']]

        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data['fwd_ret']

        # 训练
        train_dataset = lgb.Dataset(X_train, label=y_train)

        params = {
            'objective': 'regression',
            'metric': 'mse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_samples': self.min_child_samples,
            'n_jobs': -1,
        }

        self.model = lgb.train(params, train_dataset, num_boost_round=self.n_estimators)

        # 预测
        X_pred = df[feature_cols].fillna(0)
        df['score'] = self.model.predict(X_pred)

        # 归一化到0-1
        score_min = df['score'].min()
        score_max = df['score'].max()
        if score_max > score_min:
            df['score'] = (df['score'] - score_min) / (score_max - score_min)
        else:
            df['score'] = 0.5

        return df[['date', 'code', 'score']].reset_index(drop=True)


class XGBoostCrossSection(BaseModel):
    """
    XGBoost截面模型
    """
    name = "xgboost_cs"
    category = "tree"
    description = "XGBoost截面模型，月度滚动训练"
    requires_training = True

    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.feature_cols = None

    def _get_features(self, df: pd.DataFrame) -> list:
        exclude = {'date', 'code', 'close', 'open', 'high', 'low', 'volume',
                   'fwd_ret', 'ym', 'isST', 'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d'}
        return [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        import xgboost as xgb

        date = pd.Timestamp(date)
        df = factor_panel[factor_panel['date'] == date].copy()

        if df.empty:
            return pd.DataFrame(columns=['date', 'code', 'score'])

        feature_cols = self._get_features(factor_panel)

        # 训练数据
        train_end_dt = date
        train_start_dt = train_end_dt - pd.DateOffset(months=12)
        mask = (factor_panel['date'] >= train_start_dt) & (factor_panel['date'] < train_end_dt)
        train_data = factor_panel[mask].copy()

        if 'fwd_ret' not in train_data.columns:
            train_data['fwd_ret'] = train_data.groupby('code')['close'].transform(
                lambda x: x.shift(-20) / x - 1
            )
        train_data = train_data.dropna(subset=['fwd_ret'])

        if len(train_data) < 1000:
            df['score'] = 0.5
            return df[['date', 'code', 'score']]

        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data['fwd_ret']

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbosity=0,
        )
        self.model.fit(X_train, y_train)

        X_pred = df[feature_cols].fillna(0)
        df['score'] = self.model.predict(X_pred)

        score_min = df['score'].min()
        score_max = df['score'].max()
        if score_max > score_min:
            df['score'] = (df['score'] - score_min) / (score_max - score_min)
        else:
            df['score'] = 0.5

        return df[['date', 'code', 'score']].reset_index(drop=True)


# ── 注册 ──────────────────────────────────────────────

register_model(LightGBMCrossSection())
register_model(XGBoostCrossSection())
