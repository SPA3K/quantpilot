"""ATR追踪止损策略 — J. Welles Wilder (1978)"""

import numpy as np
from quantpilot.algorithms import Algorithm, ParamDef
from quantpilot.core import Bar, Signal


class ATRTrailingStop(Algorithm):
    """
    ATR追踪止损: 价格跌破ATR×N倍追踪线卖出。

    来源: J. Welles Wilder, "New Concepts in Technical Trading Systems" (1978)
    逻辑: ATR=真实波幅的N日均值，反映股票的波动性。
          追踪止损线=最高价-ATR×倍数。
          价格跌破追踪线=趋势反转（卖出）。
          优势：止损位随价格上涨自动上移，锁定利润。
    """

    name = "ATR追踪止损"
    description = "跌破ATR追踪线时卖出"
    category = "sell"
    params = [
        ParamDef("atr_period", "int", 14, 7, 28, "ATR计算周期"),
        ParamDef("atr_multiplier", "float", 2.0, 1.0, 4.0, "ATR倍数"),
    ]

    def _calc_atr(self, bars: list[Bar], period: int) -> float:
        """计算ATR"""
        if len(bars) < period + 1:
            return 0.0

        true_ranges = []
        for i in range(-period, 0):
            high = bars[i].high
            low = bars[i].low
            prev_close = bars[i - 1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        return np.mean(true_ranges)

    def compute(self, bars: list[Bar], params: dict) -> Signal | None:
        period = params["atr_period"]
        multiplier = params["atr_multiplier"]

        if len(bars) < period + 2:
            return None

        atr = self._calc_atr(bars, period)
        if atr <= 0:
            return None

        # 计算追踪止损线
        recent_highs = [b.high for b in bars[-period:]]
        highest = max(recent_highs)
        trailing_stop = highest - atr * multiplier

        current = bars[-1].close
        prev = bars[-2].close

        ticker = bars[-1].ticker
        ts = bars[-1].timestamp

        # 价格跌破追踪线
        if current < trailing_stop and prev >= trailing_stop:
            distance = (trailing_stop - current) / atr if atr > 0 else 0
            strength = min(1.0, 0.5 + distance * 0.2)
            return Signal(
                timestamp=ts, ticker=ticker, action="sell",
                strength=strength,
                reason=f"跌破ATR追踪线 {current:.2f}<{trailing_stop:.2f} (ATR={atr:.2f})",
                algorithm=self.name,
            )

        return None

    def get_warmup_days(self) -> int:
        return 30
