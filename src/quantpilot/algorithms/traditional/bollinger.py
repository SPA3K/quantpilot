"""布林带策略 — John Bollinger (1983)"""

import numpy as np
from quantpilot.algorithms import Algorithm, ParamDef
from quantpilot.core import Bar, Signal


class BollingerBands(Algorithm):
    """
    布林带: 价格触及下轨买入，触及上轨卖出。

    来源: John Bollinger, "Bollinger on Bollinger Bands" (2001)
    逻辑: 中轨=N日SMA，上轨=中轨+2σ，下轨=中轨-2σ。
          价格触及下轨=超卖（反弹概率大），触及上轨=超买（回调概率大）。
          带宽收窄后突破=趋势启动信号。
    """

    name = "布林带"
    description = "触及下轨买入，触及上轨卖出"
    category = "buy"
    params = [
        ParamDef("period", "int", 20, 10, 50, "均线周期"),
        ParamDef("num_std", "float", 2.0, 1.0, 3.0, "标准差倍数"),
    ]

    def compute(self, bars: list[Bar], params: dict) -> Signal | None:
        period = params["period"]
        num_std = params["num_std"]

        if len(bars) < period + 1:
            return None

        closes = [b.close for b in bars[-period:]]
        mid = np.mean(closes)
        std = np.std(closes, ddof=1)

        upper = mid + num_std * std
        lower = mid - num_std * std

        current = bars[-1].close
        ticker = bars[-1].ticker
        ts = bars[-1].timestamp

        # 触及下轨 → 买入
        if current <= lower:
            strength = min(1.0, (lower - current) / std if std > 0 else 0.5)
            return Signal(
                timestamp=ts, ticker=ticker, action="buy",
                strength=strength,
                reason=f"触及布林下轨 {current:.2f}<{lower:.2f}",
                algorithm=self.name,
            )

        # 触及上轨 → 卖出
        if current >= upper:
            strength = min(1.0, (current - upper) / std if std > 0 else 0.5)
            return Signal(
                timestamp=ts, ticker=ticker, action="sell",
                strength=strength,
                reason=f"触及布林上轨 {current:.2f}>{upper:.2f}",
                algorithm=self.name,
            )

        return None

    def get_warmup_days(self) -> int:
        return 30
