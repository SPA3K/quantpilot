"""MACD策略 — Gerald Appel (1979)"""

import numpy as np
from quantpilot.algorithms import Algorithm, ParamDef
from quantpilot.core import Bar, Signal


class MACDAlgorithm(Algorithm):
    """
    MACD: DIF上穿DEA买入，下穿卖出。

    来源: Gerald Appel, "Systems and Forecasts" (1979)
    逻辑: DIF=EMA12-EMA26，DEA=DIF的9日EMA。
          金叉(DIF>DEA)=动量转多，死叉=动量转空。
          MACD柱=2*(DIF-DEA)，柱状体放大=趋势加速。
    """

    name = "MACD"
    description = "DIF上穿DEA买入，下穿卖出"
    category = "buy"
    params = [
        ParamDef("fast_period", "int", 12, 5, 26, "快线EMA周期"),
        ParamDef("slow_period", "int", 26, 12, 52, "慢线EMA周期"),
        ParamDef("signal_period", "int", 9, 5, 20, "信号线周期"),
    ]

    def _ema(self, data: list[float], period: int) -> list[float]:
        """计算EMA"""
        if len(data) < period:
            return []
        k = 2 / (period + 1)
        ema = [sum(data[:period]) / period]
        for price in data[period:]:
            ema.append(price * k + ema[-1] * (1 - k))
        return ema

    def compute(self, bars: list[Bar], params: dict) -> Signal | None:
        fast_p = params["fast_period"]
        slow_p = params["slow_period"]
        signal_p = params["signal_period"]

        min_bars = slow_p + signal_p + 1
        if len(bars) < min_bars:
            return None

        closes = [b.close for b in bars]

        # 计算EMA
        ema_fast = self._ema(closes, fast_p)
        ema_slow = self._ema(closes, slow_p)

        # 对齐: EMA_fast从第fast_p个开始，EMA_slow从第slow_p个开始
        # 取公共部分
        offset = slow_p - fast_p
        n = min(len(ema_fast) - offset, len(ema_slow))
        dif_line = [ema_fast[offset + i] - ema_slow[i] for i in range(n)]

        if len(dif_line) < signal_p + 1:
            return None

        # DEA = DIF的signal_period日EMA
        dea_line = self._ema(dif_line, signal_p)

        if len(dea_line) < 2:
            return None

        dif = dif_line[-1]
        dea = dea_line[-1]
        dif_prev = dif_line[-signal_p - 1]
        dea_prev = dea_line[-2]

        ticker = bars[-1].ticker
        ts = bars[-1].timestamp

        # 金叉
        if dif_prev <= dea_prev and dif > dea:
            macd_bar = 2 * (dif - dea)
            strength = min(1.0, abs(macd_bar) / bars[-1].close * 100)
            return Signal(
                timestamp=ts, ticker=ticker, action="buy",
                strength=strength,
                reason=f"MACD金叉 DIF={dif:.3f} DEA={dea:.3f}",
                algorithm=self.name,
            )

        # 死叉
        if dif_prev >= dea_prev and dif < dea:
            macd_bar = 2 * (dif - dea)
            strength = min(1.0, abs(macd_bar) / bars[-1].close * 100)
            return Signal(
                timestamp=ts, ticker=ticker, action="sell",
                strength=strength,
                reason=f"MACD死叉 DIF={dif:.3f} DEA={dea:.3f}",
                algorithm=self.name,
            )

        return None

    def get_warmup_days(self) -> int:
        return 60
