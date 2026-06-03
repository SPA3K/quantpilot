"""海龟交易法 — Curtis Faith (2007)"""

import numpy as np
from quantpilot.algorithms import Algorithm, ParamDef
from quantpilot.core import Bar, Signal


class TurtleBreakout(Algorithm):
    """
    海龟交易法: 突破N日高点买入，跌破M日低点卖出。

    来源: Curtis Faith, "Way of the Turtle" (2007)
    逻辑: 价格突破近期高点=趋势启动，跌破近期低点=趋势结束。
          追涨杀跌，顺势而为。
    """

    name = "海龟交易法"
    description = "突破N日高点买入，跌破M日低点卖出"
    category = "buy"
    params = [
        ParamDef("entry_period", "int", 20, 10, 55, "入场突破周期"),
        ParamDef("exit_period", "int", 10, 5, 20, "出场跌破周期"),
    ]

    def compute(self, bars: list[Bar], params: dict) -> Signal | None:
        entry = params["entry_period"]
        exit_ = params["exit_period"]

        if len(bars) < max(entry, exit_) + 1:
            return None

        ticker = bars[-1].ticker
        ts = bars[-1].timestamp
        current_close = bars[-1].close

        # 入场：突破entry_period日最高价
        highs = [b.high for b in bars[-entry-1:-1]]
        entry_high = max(highs)

        if current_close > entry_high:
            strength = min(1.0, (current_close - entry_high) / entry_high * 10)
            return Signal(
                timestamp=ts, ticker=ticker, action="buy",
                strength=strength,
                reason=f"突破{entry}日高点 {entry_high:.2f}",
                algorithm=self.name,
            )

        # 出场：跌破exit_period日最低价
        lows = [b.low for b in bars[-exit_-1:-1]]
        exit_low = min(lows)

        if current_close < exit_low:
            strength = min(1.0, (exit_low - current_close) / current_close * 10)
            return Signal(
                timestamp=ts, ticker=ticker, action="sell",
                strength=strength,
                reason=f"跌破{exit_}日低点 {exit_low:.2f}",
                algorithm=self.name,
            )

        return None

    def get_warmup_days(self) -> int:
        return 60
