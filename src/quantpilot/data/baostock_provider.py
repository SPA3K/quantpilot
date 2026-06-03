"""Baostock data provider."""

import pandas as pd
import baostock as bs

from quantpilot.data import DataProvider


# 股票名称→代码映射（常用股票）
TICKER_MAP = {
    "宁德时代": "sz.300750",
    "阳光电源": "sz.300274",
    "隆基绿能": "sh.601012",
    "贵州茅台": "sh.600519",
    "中际旭创": "sz.300308",
    "立讯精密": "sz.002475",
    "同花顺": "sz.300033",
    "科大讯飞": "sz.002230",
    "比亚迪": "sz.002594",
    "招商银行": "sh.600036",
    "中国平安": "sh.601318",
    "五粮液": "sz.000858",
    "美的集团": "sz.000333",
    "格力电器": "sz.000651",
    "海康威视": "sz.002415",
    "恒瑞医药": "sh.600276",
    "迈瑞医疗": "sz.300760",
    "金山办公": "sh.688111",
    "新易盛": "sz.300502",
    "天孚通信": "sz.300394",
    "通威股份": "sh.600438",
    "天合光能": "sh.688599",
    "亿纬锂能": "sz.300014",
    "恩捷股份": "sz.002812",
    "紫金矿业": "sh.601899",
    "中国中免": "sh.601888",
}


class BaostockProvider(DataProvider):
    """Baostock数据提供者"""

    def __init__(self):
        self._logged_in = False

    def _login(self):
        if not self._logged_in:
            bs.login()
            self._logged_in = True

    def _logout(self):
        if self._logged_in:
            bs.logout()
            self._logged_in = False

    def resolve_ticker(self, name: str) -> str:
        """将股票名称解析为baostock代码"""
        if name in TICKER_MAP:
            return TICKER_MAP[name]
        # 尝试直接作为代码使用
        if name.startswith("sz.") or name.startswith("sh."):
            return name
        raise ValueError(f"Unknown ticker: {name}")

    def get_bars(self, ticker: str, start: str, end: str, freq: str = "daily") -> pd.DataFrame:
        """获取OHLCV数据"""
        code = self.resolve_ticker(ticker)
        self._login()

        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,volume,amount,turn",
            start_date=start,
            end_date=end,
            frequency="d" if freq == "daily" else "m",
            adjustflag="2",  # 前复权
        )

        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=rs.fields)
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume", "amount", "turn"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.rename(columns={"amount": "turnover", "turn": "turnover_rate"})
        df = df.dropna(subset=["close"])

        return df[["date", "open", "high", "low", "close", "volume", "turnover"]]
