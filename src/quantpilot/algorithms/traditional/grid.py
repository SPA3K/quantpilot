"""网格交易策略"""

import numpy as np
from quantpilot.algorithms import Algorithm, ParamDef
from quantpilot.core import Bar, Signal


class GridTrading(Algorithm):
    """
    网格交易: 价格每跌N%买入一格，每涨N%卖出一格。

    来源: 经典网格交易法（无单一论文来源，广泛用于震荡市场）
    逻辑: 在价格区间内设置等距网格。
          跌破某格→买入（预期反弹）。
          涨破某格→卖出（预期回落）。
          适合横盘震荡市场，趋势市场容易踏空或套牢。
    """

    name = "网格交易"
    description = "跌N%买入一格，涨N%卖出一格"
    category = "buy"
    params = [
        ParamDef("grid_pct", "float", 0.03, 0.01, 0.10, "网格间距(百分比)"),
        ParamDef("lookback", "int", 60, 20, 120, "基准价格计算周期"),
        ParamDef("base_price_type", "int", 0, 0, 2, "基准价: 0=均价, 1=中位数, 2=当前价"),
    ]

    def compute(self, bars: list[Bar], params: dict) -> Signal | None:
        grid_pct = params["grid_pct"]
        lookback = params["lookback"]
        base_type = params["base_price_type"]

        if len(bars) < lookback:
            return None

        closes = [b.close for b in bars[-lookback:]]

        # 计算基准价格
        if base_type == 0:
            base = np.mean(closes)
        elif base_type == 1:
            base = np.median(closes)
        else:
            base = bars[-1].close

        current = bars[-1].close
        prev = bars[-2].close

        ticker = bars[-1].ticker
        ts = bars[-1].timestamp

        # 计算当前在第几格
        grid_level_now = (current - base) / (base * grid_pct)
        grid_level_prev = (prev - base) / (base * grid_pct)

        # 向下穿越网格线 → 买入
        if int(grid_level_prev) > int(grid_level_now):
            distance = abs(grid_level_now)
            strength = min(1.0, distance / 10)
            return Signal(
                timestamp=ts, ticker=ticker, action="buy",
                strength=max(0.1, strength),
                reason=f"网格买入 跌穿第{int(grid_level_now)}格 ({current:.2f})",
                algorithm=self.name,
            )

        # 向上穿越网格线 → 卖出
        if int(grid_level_prev) < int(grid_level_now):
            distance = abs(grid_level_now)
            strength = min(1.0, distance / 10)
            return Signal(
                timestamp=ts, ticker=ticker, action="sell",
                strength=max(0.1, strength),
                reason=f"网格卖出 涨穿第{int(grid_level_now)}格 ({current:.2f})",
                algorithm=self.name,
            )

        return None

    def get_warmup_days(self) -> int:
        return 60
