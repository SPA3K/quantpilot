"""OBV能量潮策略 — Joseph Granville (1963)"""

import numpy as np
from quantpilot.algorithms import Algorithm, ParamDef
from quantpilot.core import Bar, Signal


class OBVAlgorithm(Algorithm):
    """
    OBV能量潮: OBV趋势向上+价格突破买入，OBV背离卖出。

    来源: Joseph Granville, "A New Strategy of Daily Stock Market Timing" (1963)
    逻辑: OBV=累积成交量（涨日加量，跌日减量）。
          OBV创新高而价格未创新高=顶背离（卖出信号）。
          OBV趋势向上+价格突破=趋势确认。
    """

    name = "OBV能量潮"
    description = "OBV趋势确认买入，顶背离卖出"
    category = "buy"
    params = [
        ParamDef("obv_ma_period", "int", 20, 10, 50, "OBV均线周期"),
        ParamDef("divergence_days", "int", 20, 10, 40, "背离检测周期"),
    ]

    def compute(self, bars: list[Bar], params: dict) -> Signal | None:
        ma_period = params["obv_ma_period"]
        div_days = params["divergence_days"]

        if len(bars) < max(ma_period, div_days) + 2:
            return None

        # 计算OBV序列
        obv = [0.0]
        for i in range(1, len(bars)):
            if bars[i].close > bars[i - 1].close:
                obv.append(obv[-1] + bars[i].volume)
            elif bars[i].close < bars[i - 1].close:
                obv.append(obv[-1] - bars[i].volume)
            else:
                obv.append(obv[-1])

        # OBV均线
        obv_ma = np.mean(obv[-ma_period:])
        obv_now = obv[-1]

        ticker = bars[-1].ticker
        ts = bars[-1].timestamp

        # 顶背离检测：价格创新高但OBV未创新高
        recent_prices = [b.close for b in bars[-div_days:]]
        recent_obv = obv[-div_days:]

        price_at_start = bars[-div_days].close
        obv_at_start = obv[-div_days]

        # 价格在高位（接近N日高点）
        price_near_high = bars[-1].close >= max(recent_prices) * 0.98
        # OBV没有跟随
        obv_not_high = obv_now < max(recent_obv) * 0.95

        if price_near_high and obv_not_high:
            strength = min(1.0, (max(recent_obv) - obv_now) / abs(max(recent_obv)) if max(recent_obv) != 0 else 0.5)
            return Signal(
                timestamp=ts, ticker=ticker, action="sell",
                strength=max(0.2, strength),
                reason=f"OBV顶背离 价格高位但OBV下降",
                algorithm=self.name,
            )

        # OBV趋势向上 + 价格突破
        if obv_now > obv_ma and bars[-1].close > bars[-2].close:
            if bars[-1].close > bars[-3].close:  # 连涨
                strength = min(1.0, (obv_now - obv_ma) / abs(obv_ma) * 10) if obv_ma != 0 else 0.5
                return Signal(
                    timestamp=ts, ticker=ticker, action="buy",
                    strength=max(0.1, strength),
                    reason=f"OBV趋势向上+价格突破 OBV={obv_now:,.0f}",
                    algorithm=self.name,
                )

        return None

    def get_warmup_days(self) -> int:
        return 60
