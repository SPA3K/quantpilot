"""止盈策略"""

from quantpilot.algorithms import Algorithm, ParamDef
from quantpilot.core import Bar, Signal


class TakeProfit(Algorithm):
    """
    止盈: 收益达到目标时卖出。
    """

    name = "止盈"
    description = "收益达到目标时卖出"
    category = "sell"
    params = [
        ParamDef("target_return", "float", 0.10, 0.03, 0.50, "目标收益率"),
    ]

    def compute(self, bars: list[Bar], params: dict) -> Signal | None:
        # 止盈需要知道持仓成本，由引擎处理
        # 这里返回信号让引擎检查
        return Signal(
            timestamp=bars[-1].timestamp,
            ticker=bars[-1].ticker,
            action="sell",
            strength=0.5,
            reason=f"止盈{params['target_return']:.0%}",
            algorithm=self.name,
        )

    def should_sell(self, current_price: float, avg_cost: float, params: dict) -> bool:
        """判断是否应该止盈"""
        ret = (current_price - avg_cost) / avg_cost
        return ret >= params["target_return"]

    def get_warmup_days(self) -> int:
        return 1
