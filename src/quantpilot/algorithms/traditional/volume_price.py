"""量价分析策略 — 成交量确认"""

import numpy as np
from quantpilot.algorithms import Algorithm, ParamDef
from quantpilot.core import Bar, Signal


class VolumePrice(Algorithm):
    """
    量价配合: 放量突破买入，缩量滞涨卖出。

    来源: Joseph Granville, "Granville's New Key to Stock Market Profits" (1960)
          + 量价关系经典理论
    逻辑: 价涨+量增=趋势确认（买入）。
          价平+量缩=动能衰竭（卖出预警）。
          量比=当日成交量/N日均量，>1.5=放量。
    """

    name = "量价配合"
    description = "放量上涨买入，缩量滞涨卖出"
    category = "buy"
    params = [
        ParamDef("volume_ratio", "float", 1.5, 1.1, 3.0, "放量阈值(量比)"),
        ParamDef("price_change", "float", 0.02, 0.005, 0.05, "最小涨幅"),
        ParamDef("lookback", "int", 20, 10, 60, "均量计算周期"),
    ]

    def compute(self, bars: list[Bar], params: dict) -> Signal | None:
        lookback = params["lookback"]
        vr = params["volume_ratio"]
        min_change = params["price_change"]

        if len(bars) < lookback + 1:
            return None

        current = bars[-1]
        prev = bars[-2]

        # 计算量比
        avg_vol = np.mean([b.volume for b in bars[-lookback - 1:-1]])
        if avg_vol <= 0:
            return None
        volume_ratio = current.volume / avg_vol

        # 计算涨跌幅
        price_change = (current.close - prev.close) / prev.close

        ticker = current.ticker
        ts = current.timestamp

        # 放量上涨
        if volume_ratio >= vr and price_change >= min_change:
            strength = min(1.0, volume_ratio / vr / 2 + price_change / min_change / 4)
            return Signal(
                timestamp=ts, ticker=ticker, action="buy",
                strength=strength,
                reason=f"放量上涨 量比={volume_ratio:.1f}x 涨幅={price_change:+.1%}",
                algorithm=self.name,
            )

        # 缩量滞涨（成交量<均量的50%，涨幅<0.5%）
        if volume_ratio < 0.5 and abs(price_change) < 0.005 and len(bars) > 5:
            # 检查前几天是否有过上涨
            recent_gains = sum(
                1 for b in bars[-5:]
                if bars[bars.index(b) - 1].close < b.close
            ) if bars.index(bars[-5]) > 0 else 0

            if recent_gains >= 2:  # 近期涨过
                strength = min(1.0, (0.5 - volume_ratio) * 2)
                return Signal(
                    timestamp=ts, ticker=ticker, action="sell",
                    strength=strength,
                    reason=f"缩量滞涨 量比={volume_ratio:.1f}x",
                    algorithm=self.name,
                )

        return None

    def get_warmup_days(self) -> int:
        return 30
