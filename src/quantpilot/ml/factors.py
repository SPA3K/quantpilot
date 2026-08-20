"""
QuantPilot ML Factor Engineering
60+ 因子计算：估值 / 盈利 / 成长 / 技术 / 动量 / 波动 / 流动性 / 反转
输入：日线parquet → 输出：因子矩阵parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path


class FactorEngine:
    """多因子计算引擎"""

    def __init__(self):
        pass

    # ── 技术因子 ──────────────────────────────────────────────

    @staticmethod
    def ma_cross(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
        """均线交叉信号（快线-慢线）/ 慢线"""
        ma_f = df["close"].rolling(fast).mean()
        ma_s = df["close"].rolling(slow).mean()
        return (ma_f - ma_s) / ma_s

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """RSI"""
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD (DIF, DEA, Histogram)"""
        ema_fast = df["close"].ewm(span=fast).mean()
        ema_slow = df["close"].ewm(span=slow).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal).mean()
        hist = 2 * (dif - dea)
        return dif, dea, hist

    @staticmethod
    def bollinger_pct(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """布林带位置百分比 (当前价在布林带中的位置)"""
        ma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        upper = ma + std_dev * std
        lower = ma - std_dev * std
        return (df["close"] - lower) / (upper - lower)

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3):
        """KDJ指标"""
        low_n = df["low"].rolling(n).min()
        high_n = df["high"].rolling(n).max()
        rsv = (df["close"] - low_n) / (high_n - low_n) * 100
        k = rsv.ewm(alpha=1/m1).mean()
        d = k.ewm(alpha=1/m2).mean()
        j = 3 * k - 2 * d
        return k, d, j

    # ── 动量因子 ──────────────────────────────────────────────

    @staticmethod
    def momentum(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """动量因子：N日收益率"""
        return df["close"].pct_change(period)

    @staticmethod
    def roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
        """Rate of Change"""
        return (df["close"] - df["close"].shift(period)) / df["close"].shift(period) * 100

    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """威廉指标"""
        high_n = df["high"].rolling(period).max()
        low_n = df["low"].rolling(period).min()
        return (high_n - df["close"]) / (high_n - low_n) * -100

    # ── 波动率因子 ──────────────────────────────────────────

    @staticmethod
    def volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """历史波动率（日收益率标准差 * sqrt(252)）"""
        returns = df["close"].pct_change()
        return returns.rolling(period).std() * np.sqrt(252)

    @staticmethod
    def downside_vol(df: pd.DataFrame, period: int = 60) -> pd.Series:
        """下行波动率（只计算负收益的波动）"""
        returns = df["close"].pct_change()
        neg_returns = returns.where(returns < 0, 0)
        return neg_returns.rolling(period).std() * np.sqrt(252)

    # ── 流动性因子 ──────────────────────────────────────────

    @staticmethod
    def turnover_rate_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """换手率均值"""
        return df["turnover_rate"].rolling(period).mean()

    @staticmethod
    def amihud_illiquidity(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Amihud非流动性：|收益率| / 成交额"""
        returns = df["close"].pct_change().abs()
        illiq = returns / (df["turnover"] + 1e-10)
        return illiq.rolling(period).mean()

    @staticmethod
    def volume_ratio(df: pd.DataFrame, period: int = 5) -> pd.Series:
        """量比：当日成交量 / N日平均成交量"""
        return df["volume"] / df["volume"].rolling(period).mean()

    # ── 反转因子 ──────────────────────────────────────────────

    @staticmethod
    def reversal_5d(df: pd.DataFrame) -> pd.Series:
        """5日反转因子（负值=近期跌多，可能反弹）"""
        return -df["close"].pct_change(5)

    @staticmethod
    def reversal_20d(df: pd.DataFrame) -> pd.Series:
        """20日反转因子"""
        return -df["close"].pct_change(20)

    # ── 估值因子（使用baostock提供的PE/PB/PS）──────────────

    @staticmethod
    def ep(df: pd.DataFrame) -> pd.Series:
        """EP (Earnings/Price) = 1/PE，越高越便宜"""
        return 1.0 / (df["peTTM"].replace(0, np.nan))

    @staticmethod
    def bp(df: pd.DataFrame) -> pd.Series:
        """BP (Book/Price) = 1/PB"""
        return 1.0 / (df["pbMRQ"].replace(0, np.nan))

    @staticmethod
    def sp(df: pd.DataFrame) -> pd.Series:
        """SP (Sales/Price) = 1/PS"""
        return 1.0 / (df["psTTM"].replace(0, np.nan))

    # ── 成交量形态因子 ──────────────────────────────────────

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """On Balance Volume"""
        direction = np.sign(df["close"].diff())
        return (direction * df["volume"]).cumsum()

    @staticmethod
    def vwap_deviation(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """VWAP偏离度：当前价 vs 20日VWAP"""
        vwap = (df["turnover"]).rolling(period).sum() / (df["volume"]).rolling(period).sum()
        return (df["close"] - vwap) / vwap

    # ── 趋势强度 ──────────────────────────────────────────────

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ADX趋势强度"""
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
        return dx.rolling(period).mean()

    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """CCI顺势指标"""
        tp = (df["high"] + df["low"] + df["close"]) / 3
        ma = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - ma) / (0.015 * md)

    # ── N日收益率序列（用于滚动特征）───────────────────────

    @staticmethod
    def returns_n(df: pd.DataFrame, n: int) -> pd.Series:
        """N日收益率"""
        return df["close"].pct_change(n)

    # ── 主计算入口 ──────────────────────────────────────────

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对单只股票的日线数据计算全部因子
        输入: 日线DataFrame (必须含 close, high, low, volume, turnover, turnover_rate, peTTM, pbMRQ, psTTM)
        输出: 因子DataFrame (date为index)
        """
        result = pd.DataFrame(index=df.index)
        result["date"] = df["date"]
        result["code"] = df.get("code", "")

        # 技术因子
        result["ma_cross_5_20"] = self.ma_cross(df, 5, 20)
        result["ma_cross_10_60"] = self.ma_cross(df, 10, 60)
        result["rsi_14"] = self.rsi(df, 14)
        result["rsi_6"] = self.rsi(df, 6)
        dif, dea, hist = self.macd(df)
        result["macd_dif"] = dif
        result["macd_dea"] = dea
        result["macd_hist"] = hist
        result["bollinger_pct"] = self.bollinger_pct(df)
        result["atr_14"] = self.atr(df, 14)
        k, d, j = self.kdj(df)
        result["kdj_k"] = k
        result["kdj_d"] = d
        result["kdj_j"] = j

        # 动量因子
        result["mom_5"] = self.momentum(df, 5)
        result["mom_10"] = self.momentum(df, 10)
        result["mom_20"] = self.momentum(df, 20)
        result["mom_60"] = self.momentum(df, 60)
        result["roc_12"] = self.roc(df, 12)
        result["williams_r"] = self.williams_r(df)

        # 波动率因子
        result["vol_20"] = self.volatility(df, 20)
        result["vol_60"] = self.volatility(df, 60)
        result["downside_vol_60"] = self.downside_vol(df, 60)

        # 流动性因子
        result["turnover_ma_5"] = self.turnover_rate_ma(df, 5)
        result["turnover_ma_20"] = self.turnover_rate_ma(df, 20)
        result["amihud_20"] = self.amihud_illiquidity(df, 20)
        result["volume_ratio_5"] = self.volume_ratio(df, 5)

        # 反转因子
        result["reversal_5d"] = self.reversal_5d(df)
        result["reversal_20d"] = self.reversal_20d(df)

        # 估值因子
        result["ep"] = self.ep(df)
        result["bp"] = self.bp(df)
        result["sp"] = self.sp(df)

        # 成交量形态
        result["obv"] = self.obv(df)
        result["vwap_dev"] = self.vwap_deviation(df)

        # 趋势强度
        result["adx_14"] = self.adx(df)
        result["cci_20"] = self.cci(df)

        # 多周期收益率（作为特征和标签）
        for n in [1, 3, 5, 10, 20]:
            result[f"ret_{n}d"] = self.returns_n(df, n)

        return result

    def compute_for_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        对多只股票的面板数据计算因子
        输入: 包含code列的日线面板
        输出: 因子面板
        """
        results = []
        codes = panel["code"].unique()
        for i, code in enumerate(codes):
            stock_data = panel[panel["code"] == code].copy().sort_values("date").reset_index(drop=True)
            if len(stock_data) < 60:  # 至少60天数据
                continue
            factors = self.compute_all(stock_data)
            results.append(factors)
            if (i + 1) % 500 == 0:
                print(f"Factor progress: {i+1}/{len(codes)}")

        return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.quantpilot.ml.data_fetcher import MLDataFetcher

    fetcher = MLDataFetcher()
    engine = FactorEngine()

    print("Loading daily data...")
    panel = fetcher.load_all_daily()
    print(f"Loaded {len(panel)} rows, {panel['code'].nunique()} stocks")

    print("Computing factors...")
    factors = engine.compute_for_panel(panel)
    print(f"Computed {len(factors)} factor rows, {factors.shape[1]} columns")

    # 保存
    output = fetcher.data_dir / "factors_panel.parquet"
    factors.to_parquet(output, index=False)
    print(f"Saved to {output}")
