"""
QuantPilot Model Zoo
统一接口：所有模型输入相同数据，输出统一格式的预测分数
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """所有模型的基类"""

    name: str = "base"
    category: str = "unknown"  # factor / tree / deep / llm
    description: str = ""
    requires_training: bool = False

    @abstractmethod
    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        输入: 因子面板 (含 date, code, 及所有因子列)
        输出: DataFrame(date, code, score) — score越高越推荐买入
        """
        pass

    def get_info(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "requires_training": self.requires_training,
        }


# ── L0: 纯因子打分（不需要训练）────────────────────────────

class ValueMomentumModel(BaseModel):
    """
    经典价值+动量因子模型
    - EP (1/PE) + BP (1/PB) + 动量20日 + 低波动
    - 等权打分
    """
    name = "value_momentum"
    category = "factor"
    description = "EP + BP + 20日动量 + 低波动 等权打分"

    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        df = factor_panel[factor_panel["date"] == date].copy()
        if df.empty:
            return pd.DataFrame(columns=["date", "code", "score"])

        # 各因子截面标准化 (z-score)
        factors = ["ep", "bp", "mom_20", "vol_20"]
        for f in factors:
            if f in df.columns:
                df[f"{f}_z"] = (df[f] - df[f].mean()) / (df[f].std() + 1e-10)

        # 动量和波动取反（低波动更好）
        df["score_raw"] = (
            df.get("ep_z", 0) +
            df.get("bp_z", 0) +
            df.get("mom_20_z", 0) -
            df.get("vol_20_z", 0)
        )

        # 归一化到0-1
        s = df["score_raw"]
        df["score"] = (s - s.min()) / (s.max() - s.min() + 1e-10)

        return df[["date", "code", "score"]].reset_index(drop=True)


class QualityModel(BaseModel):
    """
    质量因子模型
    - 高ROE + 高毛利率 + 低负债 + 稳定盈利
    """
    name = "quality"
    category = "factor"
    description = "ROE + 毛利率 + 盈利稳定性 质量因子"

    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        df = factor_panel[factor_panel["date"] == date].copy()
        if df.empty:
            return pd.DataFrame(columns=["date", "code", "score"])

        # 用已有因子近似
        factors = ["ep", "bp", "reversal_20d"]
        for f in factors:
            if f in df.columns:
                df[f"{f}_z"] = (df[f] - df[f].mean()) / (df[f].std() + 1e-10)

        df["score_raw"] = (
            df.get("ep_z", 0) +
            df.get("bp_z", 0) +
            df.get("reversal_20d_z", 0)
        )

        s = df["score_raw"]
        df["score"] = (s - s.min()) / (s.max() - s.min() + 1e-10)
        return df[["date", "code", "score"]].reset_index(drop=True)


class LowVolatilityModel(BaseModel):
    """
    低波动异象 (Low Volatility Anomaly)
    - 学术界发现：低波动股票长期跑赢高波动股票
    """
    name = "low_volatility"
    category = "factor"
    description = "低波动异象：选波动率最低的股票"

    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        df = factor_panel[factor_panel["date"] == date].copy()
        if df.empty:
            return pd.DataFrame(columns=["date", "code", "score"])

        # 波动率越低越好，取反
        if "vol_20" in df.columns:
            df["score"] = 1 - (df["vol_20"] - df["vol_20"].min()) / (df["vol_20"].max() - df["vol_20"].min() + 1e-10)
        else:
            df["score"] = 0.5

        return df[["date", "code", "score"]].reset_index(drop=True)


class MomentumModel(BaseModel):
    """
    纯动量模型
    - 过去20日涨得多的股票继续涨 (趋势跟随)
    """
    name = "momentum_20d"
    category = "factor"
    description = "20日动量因子：追涨"

    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        df = factor_panel[factor_panel["date"] == date].copy()
        if df.empty:
            return pd.DataFrame(columns=["date", "code", "score"])

        if "mom_20" in df.columns:
            s = df["mom_20"]
            df["score"] = (s - s.min()) / (s.max() - s.min() + 1e-10)
        else:
            df["score"] = 0.5

        return df[["date", "code", "score"]].reset_index(drop=True)


class ReversalModel(BaseModel):
    """
    反转模型
    - 过去20日跌得多的股票反弹 (逆向投资)
    """
    name = "reversal_20d"
    category = "factor"
    description = "20日反转因子：抄底"

    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        df = factor_panel[factor_panel["date"] == date].copy()
        if df.empty:
            return pd.DataFrame(columns=["date", "code", "score"])

        if "reversal_20d" in df.columns:
            s = df["reversal_20d"]
            df["score"] = (s - s.min()) / (s.max() - s.min() + 1e-10)
        else:
            df["score"] = 0.5

        return df[["date", "code", "score"]].reset_index(drop=True)


# ── L1: 需要训练的树模型 ─────────────────────────────────

class LightGBMModel(BaseModel):
    """
    LightGBM截面选股模型
    - 需要先训练，然后滚动预测
    """
    name = "lightgbm_cross_section"
    category = "tree"
    description = "LightGBM截面多因子选股"
    requires_training = True

    def __init__(self):
        self.model = None
        self.feature_cols = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict = None):
        import lightgbm as lgb
        if params is None:
            params = {
                "objective": "binary", "metric": "auc",
                "num_leaves": 63, "learning_rate": 0.05,
                "feature_fraction": 0.7, "bagging_fraction": 0.7,
                "verbose": -1, "seed": 42,
            }
        dtrain = lgb.Dataset(X_train, label=y_train)
        self.model = lgb.train(params, dtrain, num_boost_round=300)
        self.feature_cols = list(X_train.columns)

    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        df = factor_panel[factor_panel["date"] == date].copy()
        if df.empty:
            return pd.DataFrame(columns=["date", "code", "score"])

        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = df[self.feature_cols].fillna(0)
        df["score"] = self.model.predict(X)

        return df[["date", "code", "score"]].reset_index(drop=True)


# ── 模型注册表 ──────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, BaseModel] = {}


def register_model(model: BaseModel):
    MODEL_REGISTRY[model.name] = model
    logger.info(f"Registered model: {model.name}")


def get_model(name: str) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]


def list_models() -> List[dict]:
    return [m.get_info() for m in MODEL_REGISTRY.values()]


# 注册所有内置模型
register_model(ValueMomentumModel())
register_model(QualityModel())
register_model(LowVolatilityModel())
register_model(MomentumModel())
register_model(ReversalModel())


if __name__ == "__main__":
    print("Registered models:")
    for info in list_models():
        print(f"  [{info['category']}] {info['name']}: {info['description']}")
