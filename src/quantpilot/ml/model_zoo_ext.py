"""
QuantPilot Model Zoo — 扩展模型
收集网上公认效果好的ML4F模型，统一接口
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging

from quantpilot.ml.model_zoo import BaseModel, register_model

logger = logging.getLogger(__name__)


# ── L0: Alpha101 系列（WorldQuant开源因子）────────────────

class Alpha101Model(BaseModel):
    """
    WorldQuant Alpha101 经典因子
    从101个公式化alpha中选效果最好的几个组合
    参考: https://github.com/Parsnip77/Multi-factor-Model-for-Stock-Selection
    """
    name = "alpha101"
    category = "factor"
    description = "WorldQuant Alpha101 精选因子组合"

    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        df = factor_panel[factor_panel["date"] == date].copy()
        if df.empty:
            return pd.DataFrame(columns=["date", "code", "score"])

        # Alpha#1: rank(Ts_ArgMax(SignedPower(close/delay(close,1)-1, 2), 5))
        # Alpha#6: -correlation(open, volume, 10)
        # Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))
        # Alpha#41: power(high * low, 0.5) - vwap
        # Alpha#53: -1 * delta((((close - low) - (high - close)) / (close - low)), 9)

        signals = []

        # 用已有因子近似Alpha101
        if "mom_5" in df.columns:
            signals.append(("alpha_mom5", df["mom_5"]))
        if "mom_20" in df.columns:
            signals.append(("alpha_mom20", df["mom_20"]))
        if "volume_ratio_5" in df.columns:
            signals.append(("alpha_vol_ratio", -df["volume_ratio_5"]))  # 缩量更好
        if "vwap_dev" in df.columns:
            signals.append(("alpha_vwap", -df["vwap_dev"]))  # 低于VWAP更好
        if "reversal_5d" in df.columns:
            signals.append(("alpha_rev5", df["reversal_5d"]))
        if "bollinger_pct" in df.columns:
            signals.append(("alpha_boll", -df["bollinger_pct"]))  # 接近下轨更好

        if not signals:
            df["score"] = 0.5
        else:
            # 截面标准化后等权
            total = pd.Series(0.0, index=df.index)
            for name, sig in signals:
                z = (sig - sig.mean()) / (sig.std() + 1e-10)
                total += z
            total /= len(signals)
            s = total
            df["score"] = (s - s.min()) / (s.max() - s.min() + 1e-10)

        return df[["date", "code", "score"]].reset_index(drop=True)


# ── L0: 技术指标组合（TA-Lib风格）─────────────────────────

class TechnicalComboModel(BaseModel):
    """
    经典技术指标组合
    - RSI超卖 + MACD金叉 + 布林带下轨
    - 三个信号共振时推荐
    """
    name = "technical_combo"
    category = "factor"
    description = "RSI+MACD+布林带 三信号共振"

    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        df = factor_panel[factor_panel["date"] == date].copy()
        if df.empty:
            return pd.DataFrame(columns=["date", "code", "score"])

        signals = []

        # RSI < 30 超卖 → 买入信号
        if "rsi_14" in df.columns:
            rsi_signal = (30 - df["rsi_14"]).clip(0, 30) / 30
            signals.append(rsi_signal)

        # MACD histogram > 0 → 金叉
        if "macd_hist" in df.columns:
            macd_signal = df["macd_hist"].clip(lower=0)
            macd_signal = macd_signal / (macd_signal.max() + 1e-10)
            signals.append(macd_signal)

        # 布林带位置 < 0.3 → 接近下轨
        if "bollinger_pct" in df.columns:
            boll_signal = (0.3 - df["bollinger_pct"]).clip(0, 0.3) / 0.3
            signals.append(boll_signal)

        if signals:
            total = sum(signals) / len(signals)
            s = total
            df["score"] = (s - s.min()) / (s.max() - s.min() + 1e-10)
        else:
            df["score"] = 0.5

        return df[["date", "code", "score"]].reset_index(drop=True)


# ── L0: 行业中性模型 ──────────────────────────────────────

class SectorNeutralModel(BaseModel):
    """
    行业中性选股
    - 每个行业内分别排名，消除行业beta
    - 选出每个行业内因子最强的股票
    """
    name = "sector_neutral"
    category = "factor"
    description = "行业内排名选股，消除行业beta"

    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        df = factor_panel[factor_panel["date"] == date].copy()
        if df.empty:
            return pd.DataFrame(columns=["date", "code", "score"])

        # 用代码前缀做粗略行业分类
        df["sector"] = df["code"].str[:5]  # sh.6, sz.0, sz.3

        # 每个行业内用EP+动量排名
        score_cols = []
        for col in ["ep", "mom_20"]:
            if col in df.columns:
                df[f"{col}_rank"] = df.groupby("sector")[col].rank(pct=True)
                score_cols.append(f"{col}_rank")

        if score_cols:
            df["score"] = df[score_cols].mean(axis=1)
        else:
            df["score"] = 0.5

        return df[["date", "code", "score"]].reset_index(drop=True)


# ── L0: 等权基准 ──────────────────────────────────────────

class EqualWeightModel(BaseModel):
    """
    等权基准：所有股票得分相同
    用于对比：任何模型必须跑赢这个才有价值
    """
    name = "equal_weight"
    category = "benchmark"
    description = "等权基准（所有股票得分相同）"

    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        df = factor_panel[factor_panel["date"] == date].copy()
        if df.empty:
            return pd.DataFrame(columns=["date", "code", "score"])
        df["score"] = 0.5
        return df[["date", "code", "score"]].reset_index(drop=True)


class RandomModel(BaseModel):
    """
    随机选股：随机打分
    用于对比：任何模型必须显著跑赢随机才有意义
    """
    name = "random"
    category = "benchmark"
    description = "随机选股基准"

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def predict(self, factor_panel: pd.DataFrame, date: str) -> pd.DataFrame:
        df = factor_panel[factor_panel["date"] == date].copy()
        if df.empty:
            return pd.DataFrame(columns=["date", "code", "score"])
        df["score"] = self.rng.random(len(df))
        return df[["date", "code", "score"]].reset_index(drop=True)


# ── 注册所有扩展模型 ──────────────────────────────────────

register_model(Alpha101Model())
register_model(TechnicalComboModel())
register_model(SectorNeutralModel())
register_model(EqualWeightModel())
register_model(RandomModel())


# ── 模型分组 ──────────────────────────────────────────────

MODEL_GROUPS = {
    "benchmark": ["equal_weight", "random"],
    "classic_factor": ["value_momentum", "quality", "low_volatility", "alpha101"],
    "momentum": ["momentum_20d", "reversal_20d", "technical_combo"],
    "sector_neutral": ["sector_neutral"],
    "tree": ["lightgbm_cross_section"],
}
