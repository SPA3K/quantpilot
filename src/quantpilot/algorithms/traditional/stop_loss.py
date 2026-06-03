"""止损策略"""

from quantpilot.algorithms import Algorithm, ParamDef
from quantpilot.core import Bar, Signal


class StopLoss(Algorithm):
    """
    止损: 亏损超过阈值时卖出。
    """

    name = "止损"
    description = "亏损超过阈值时卖出"
    category = "sell"
    params = [
        ParamDef("max_loss", "float", -0.05, -0.20, -0.02, "最大亏损比例"),
    ]

    def compute(self, bars: list[Bar], params: dict) -> Signal | None:
        return Signal(
            timestamp=bars[-1].timestamp,
            ticker=bars[-1].ticker,
            action="sell",
            strength=0.8,
            reason=f"止损{params['max_loss']:.0%}",
            algorithm=self.name,
        )

    def should_sell(self, current_price: float, avg_cost: float, params: dict) -> bool:
        """判断是否应该止损"""
        ret = (current_price - avg_cost) / avg_cost
        return ret <= params["max_loss"]

    def get_warmup_days(self) -> int:
        return 1
