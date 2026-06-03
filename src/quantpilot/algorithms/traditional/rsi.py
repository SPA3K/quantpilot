"""RSI超买超卖策略 — Wilder (1978)"""

import numpy as np
from quantpilot.algorithms import Algorithm, ParamDef
from quantpilot.core import Bar, Signal


class RSIAlgorithm(Algorithm):
    """
    RSI超买超卖: RSI<30买入，RSI>70卖出。

    来源: J. Welles Wilder, "New Concepts in Technical Trading Systems" (1978)
    逻辑: RSI衡量价格变动的速度和幅度。
          低于30=超卖（可能反弹），高于70=超买（可能回调）。
    """

    name = "RSI超买超卖"
    description = "RSI<30买入，RSI>70卖出"
    category = "buy"
    params = [
        ParamDef("period", "int", 14, 6, 28, "RSI计算周期"),
        ParamDef("oversold", "float", 30, 15, 40, "超卖阈值"),
        ParamDef("overbought", "float", 70, 60, 85, "超买阈值"),
    ]

    def compute(self, bars: list[Bar], params: dict) -> Signal | None:
        period = params["period"]
        if len(bars) < period + 1:
            return None

        # 计算RSI
        closes = [b.close for b in bars[-period-1:]]
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses) if np.mean(losses) > 0 else 0.0001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        ticker = bars[-1].ticker
        ts = bars[-1].timestamp

        if rsi < params["oversold"]:
            strength = (params["oversold"] - rsi) / params["oversold"]
            return Signal(
                timestamp=ts, ticker=ticker, action="buy",
                strength=strength,
                reason=f"RSI={rsi:.1f} 超卖 (<{params['oversold']})",
                algorithm=self.name,
            )

        if rsi > params["overbought"]:
            strength = (rsi - params["overbought"]) / (100 - params["overbought"])
            return Signal(
                timestamp=ts, ticker=ticker, action="sell",
                strength=strength,
                reason=f"RSI={rsi:.1f} 超买 (>{params['overbought']})",
                algorithm=self.name,
            )

        return None

    def get_warmup_days(self) -> int:
        return 30
