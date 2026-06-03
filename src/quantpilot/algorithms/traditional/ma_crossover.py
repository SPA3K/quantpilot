"""双均线交叉策略 — Granville (1960)"""

import numpy as np
from quantpilot.algorithms import Algorithm, ParamDef
from quantpilot.core import Bar, Signal


class MACrossover(Algorithm):
    """
    双均线交叉: 快线上穿慢线买入，下穿卖出。

    来源: Joseph Granville, "Granville's New Key to Stock Market Profits" (1960)
    逻辑: 短期均线代表近期趋势，长期均线代表中期趋势。
          金叉=趋势转多，死叉=趋势转空。
    """

    name = "双均线交叉"
    description = "快线上穿慢线买入，下穿卖出"
    category = "buy"
    params = [
        ParamDef("fast_period", "int", 5, 3, 60, "快线周期"),
        ParamDef("slow_period", "int", 20, 10, 250, "慢线周期"),
    ]

    def compute(self, bars: list[Bar], params: dict) -> Signal | None:
        fast = params["fast_period"]
        slow = params["slow_period"]

        if len(bars) < slow + 1:
            return None

        closes = [b.close for b in bars]

        # 当前和前一根的均线值
        ma_fast = np.mean(closes[-fast:])
        ma_fast_prev = np.mean(closes[-fast-1:-1])
        ma_slow = np.mean(closes[-slow:])
        ma_slow_prev = np.mean(closes[-slow-1:-1])

        ticker = bars[-1].ticker
        ts = bars[-1].timestamp

        # 金叉：快线从下穿上
        if ma_fast_prev <= ma_slow_prev and ma_fast > ma_slow:
            strength = min(1.0, abs(ma_fast - ma_slow) / ma_slow * 10)
            return Signal(
                timestamp=ts, ticker=ticker, action="buy",
                strength=strength,
                reason=f"MA{fast}上穿MA{slow} ({ma_fast:.2f}>{ma_slow:.2f})",
                algorithm=self.name,
            )

        # 死叉：快线从上穿下
        if ma_fast_prev >= ma_slow_prev and ma_fast < ma_slow:
            strength = min(1.0, abs(ma_slow - ma_fast) / ma_slow * 10)
            return Signal(
                timestamp=ts, ticker=ticker, action="sell",
                strength=strength,
                reason=f"MA{fast}下穿MA{slow} ({ma_fast:.2f}<{ma_slow:.2f})",
                algorithm=self.name,
            )

        return None

    def get_warmup_days(self) -> int:
        return 60
