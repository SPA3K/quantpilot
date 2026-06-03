"""KDJ策略 — George Lane (1950s)"""

import numpy as np
from quantpilot.algorithms import Algorithm, ParamDef
from quantpilot.core import Bar, Signal


class KDJAlgorithm(Algorithm):
    """
    KDJ: K上穿D买入，K下穿D卖出；J>100超买，J<0超卖。

    来源: George Lane, "Stochastics" (1950s)
    逻辑: RSV=当前价-N日最低/(N日最高-N日最低)。
          K=SMA(RSV,M1)，D=SMA(K,M2)，J=3K-2D。
          K上穿D=动量转多，J>100=极端超买。
    """

    name = "KDJ"
    description = "K上穿D买入，K下穿D卖出"
    category = "buy"
    params = [
        ParamDef("period", "int", 9, 5, 21, "RSV计算周期"),
        ParamDef("k_smooth", "int", 3, 2, 7, "K平滑周期"),
        ParamDef("d_smooth", "int", 3, 2, 7, "D平滑周期"),
    ]

    def _sma(self, values: list[float], period: int) -> list[float]:
        """SMA平滑（递推加权）"""
        if not values:
            return []
        result = [values[0]]
        for v in values[1:]:
            result.append((result[-1] * (period - 1) + v) / period)
        return result

    def compute(self, bars: list[Bar], params: dict) -> Signal | None:
        period = params["period"]

        if len(bars) < period + 2:
            return None

        # 计算RSV序列
        rsv_list = []
        for i in range(period, len(bars)):
            window = bars[i - period:i + 1]
            highest = max(b.high for b in window)
            lowest = min(b.low for b in window)
            if highest == lowest:
                rsv = 50.0
            else:
                rsv = (bars[i].close - lowest) / (highest - lowest) * 100
            rsv_list.append(rsv)

        if len(rsv_list) < 2:
            return None

        # 平滑
        k_line = self._sma(rsv_list, params["k_smooth"])
        d_line = self._sma(k_line, params["d_smooth"])

        if len(k_line) < 2 or len(d_line) < 2:
            return None

        k_now = k_line[-1]
        d_now = d_line[-1]
        k_prev = k_line[-2]
        d_prev = d_line[-2]
        j_now = 3 * k_now - 2 * d_now

        ticker = bars[-1].ticker
        ts = bars[-1].timestamp

        # K上穿D + J<50(不在超买区)
        if k_prev <= d_prev and k_now > d_now and j_now < 80:
            strength = min(1.0, (d_now - k_now + 20) / 40) if k_now > d_now else 0.3
            return Signal(
                timestamp=ts, ticker=ticker, action="buy",
                strength=max(0.1, strength),
                reason=f"KDJ金叉 K={k_now:.1f} D={d_now:.1f} J={j_now:.1f}",
                algorithm=self.name,
            )

        # K下穿D 或 J>100
        if (k_prev >= d_prev and k_now < d_now) or j_now > 100:
            strength = min(1.0, (j_now - 80) / 40) if j_now > 80 else 0.3
            return Signal(
                timestamp=ts, ticker=ticker, action="sell",
                strength=max(0.1, strength),
                reason=f"KDJ死叉 K={k_now:.1f} D={d_now:.1f} J={j_now:.1f}",
                algorithm=self.name,
            )

        return None

    def get_warmup_days(self) -> int:
        return 30
