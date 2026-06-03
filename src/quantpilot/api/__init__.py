"""Strategy API — 用户接口"""

from typing import Union
from quantpilot.algorithms import Algorithm
from quantpilot.algorithms.traditional.take_profit import TakeProfit
from quantpilot.algorithms.traditional.stop_loss import StopLoss
from quantpilot.core import Bar, Signal


class Strategy:
    """
    策略定义。

    用法:
        strategy = Strategy(
            stocks=["宁德时代", "阳光电源"],
            buy=MACrossover(fast=5, slow=20),
            sell=[TakeProfit(target=0.10), StopLoss(max_loss=-0.05)],
            position=FixedPosition(size=50000),
        )
    """

    def __init__(
        self,
        stocks: list[str],
        buy: Union[Algorithm, list[Algorithm]],
        sell: Union[Algorithm, list[Algorithm]] = None,
        position: "PositionSizing" = None,
        logic: str = "buy_any",  # "buy_all" | "buy_any"
    ):
        self.stocks = stocks
        self.buy_algorithms = buy if isinstance(buy, list) else [buy]
        self.sell_algorithms = sell if isinstance(sell, list) else [sell] if sell else []
        self.position_sizing = position or FixedPosition(50000)
        self.logic = logic

    def check_buy(self, ticker: str, bars: list[Bar]) -> list[Signal]:
        """检查买入信号"""
        signals = []
        for algo in self.buy_algorithms:
            if isinstance(algo, (TakeProfit, StopLoss)):
                continue  # 跳过卖出类算法
            params = algo.get_default_params()
            signal = algo.compute(bars, params)
            if signal and signal.action == "buy":
                signals.append(signal)
        return signals

    def check_sell(self, ticker: str, bars: list[Bar]) -> list[Signal]:
        """检查卖出信号"""
        signals = []

        # 先检查止盈止损（需要知道持仓成本）
        for algo in self.sell_algorithms:
            params = algo.get_default_params()
            if isinstance(algo, TakeProfit) or isinstance(algo, StopLoss):
                # 这些需要引擎传入持仓信息，这里先返回信号
                signal = algo.compute(bars, params)
                if signal:
                    signals.append(signal)
            else:
                signal = algo.compute(bars, params)
                if signal and signal.action == "sell":
                    signals.append(signal)

        return signals

    def get_position_size(self, cash: float, bar: Bar, positions: dict) -> float:
        """计算仓位大小"""
        return self.position_sizing.compute(cash, bar, positions)

    def get_warmup_days(self) -> int:
        """预热期天数"""
        return max(
            algo.get_warmup_days()
            for algo in self.buy_algorithms + self.sell_algorithms
        )


class FixedPosition:
    """固定仓位"""

    def __init__(self, size: float):
        self.size = size

    def compute(self, cash: float, bar: Bar, positions: dict) -> float:
        return min(self.size, cash * 0.95)  # 留5%现金


class PercentPosition:
    """按资金比例"""

    def __init__(self, percent: float = 0.3):
        self.percent = percent

    def compute(self, cash: float, bar: Bar, positions: dict) -> float:
        return cash * self.percent
