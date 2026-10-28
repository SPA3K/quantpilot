"""Tencent data provider — supports HK stocks + A-shares via Tencent API.

No external dependencies beyond stdlib + pandas.
Works for any ticker: hk00700, sh600519, sz300750, usNVDA, etc.
"""

import urllib.request
import pandas as pd

from quantpilot.data import DataProvider


class TencentProvider(DataProvider):
    """Tencent行情数据提供者（港股+A股+美股）"""

    # 名称→腾讯代码映射（常用标的，可扩展）
    TICKER_MAP = {
        # 港股
        "腾讯控股": "hk00700", "阿里巴巴": "hk09988", "小米集团": "hk01810",
        "美团": "hk03690", "京东": "hk09618", "网易": "hk09999",
        "比亚迪股份": "hk01211", "中芯国际": "hk00981", "联想集团": "hk00992",
        # A股（腾讯格式）
        "贵州茅台": "sh600519", "宁德时代": "sz300750", "比亚迪": "sz002594",
        "招商银行": "sh600036", "中国平安": "sh601318", "五粮液": "sz000858",
        "美的集团": "sz000333", "海康威视": "sz002415", "隆基绿能": "sh601012",
        "中际旭创": "sz300308", "新易盛": "sz300502", "寒武纪": "sh688256",
        "东方财富": "sz300059", "紫金矿业": "sh601899",
        # 美股
        "英伟达": "usNVDA", "苹果": "usAAPL", "特斯拉": "usTSLA",
    }

    def resolve_ticker(self, name: str) -> str:
        """将股票名称解析为腾讯代码"""
        if name in self.TICKER_MAP:
            return self.TICKER_MAP[name]
        # 已是合法腾讯代码格式
        for prefix in ("hk", "sh", "sz", "us"):
            if name.startswith(prefix) and len(name) > 2:
                return name
        # 纯数字代码：自动判断市场
        if name.isdigit():
            if len(name) == 5:
                return f"hk{name}"
            if name.startswith(("6", "9")):
                return f"sh{name}"
            return f"sz{name}"
        raise ValueError(f"Unknown ticker: {name}")

    def get_quote(self, ticker: str) -> dict:
        """获取实时行情快照"""
        code = self.resolve_ticker(ticker)
        req = urllib.request.Request(
            f"https://qt.gtimg.cn/q={code}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        raw = urllib.request.urlopen(req, timeout=10).read().decode("gbk")
        p = raw.split("~")
        if len(p) < 40:
            raise ValueError(f"Invalid quote response for {code}")
        return {
            "name": p[1], "code": p[2],
            "price": float(p[3]), "prev_close": float(p[4]),
            "open": float(p[5]), "chg_pct": float(p[32]),
            "high": float(p[33]), "low": float(p[34]),
            "volume": float(p[6]) if p[6] else 0,
            "amount": float(p[37]) if p[37] else 0,
            "time": p[30],
        }

    def get_bars(self, ticker: str, start: str = "", end: str = "",
                 freq: str = "daily", count: int = 800) -> pd.DataFrame:
        """获取前复权日K线。

        Args:
            ticker: 股票代码或名称
            start/end: 预留接口（腾讯API按条数取，不按日期）
            freq: "daily"（目前仅支持日线）
            count: 取最近N根K线（默认800）

        Returns:
            DataFrame: [date, open, high, low, close, volume]
        """
        code = self.resolve_ticker(ticker)
        url = (
            f"https://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get"
            f"?param={code},day,,,{count},qfq"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        import json
        data = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
        node = data["data"][code]
        kl = node.get("qfqday") or node.get("day")
        if not kl:
            return pd.DataFrame()
        rows = [
            {
                "date": r[0],
                "open": float(r[1]),
                "high": float(r[3]),
                "low": float(r[4]),
                "close": float(r[2]),
                "volume": float(r[5]) if len(r) > 5 else 0,
            }
            for r in kl
        ]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df
