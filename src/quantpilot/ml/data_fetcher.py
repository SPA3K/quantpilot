"""
QuantPilot ML Data Fetcher
从baostock拉取全A股日线+基本面数据，存为parquet
支持增量更新
"""

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import baostock as bs

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "ml"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class MLDataFetcher:
    """全A股数据拉取 + 本地parquet缓存"""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._logged_in = False

    def _login(self):
        if not self._logged_in:
            lg = bs.login()
            if lg.error_code != "0":
                raise RuntimeError(f"baostock login failed: {lg.error_msg}")
            self._logged_in = True

    def _logout(self):
        if self._logged_in:
            bs.logout()
            self._logged_in = False

    # ── 股票列表 ──────────────────────────────────────────────

    def fetch_stock_list(self, date: str = None) -> pd.DataFrame:
        """获取指定日期的全A股列表，存parquet"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        cache_file = self.data_dir / "stock_list.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            if date in df["date"].values:
                logger.info(f"Stock list cache hit for {date}")
                return df[df["date"] == date]

        self._login()
        rs = bs.query_all_stock(day=date)
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            # Not a trading day - fall back to cached data
            if cache_file.exists():
                df = pd.read_parquet(cache_file)
                latest_date = df["date"].max()
                logger.warning(f"No stock data for {date} (not a trading day?), using cached {latest_date}")
                return df[df["date"] == latest_date]
            logger.warning(f"No stock data for {date}")
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=rs.fields)
        df["date"] = date

        # 追加到parquet
        if cache_file.exists():
            existing = pd.read_parquet(cache_file)
            df = pd.concat([existing, df]).drop_duplicates(
                subset=["code", "date"]
            ).reset_index(drop=True)

        df.to_parquet(cache_file, index=False)
        logger.info(f"Saved {len(df)} stocks for {date}")
        return df[df["date"] == date]

    # ── 日线行情 ──────────────────────────────────────────────

    def fetch_daily_bars(self, code: str, start: str, end: str) -> pd.DataFrame:
        """拉取单只股票日线OHLCV+估值"""
        self._login()
        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,volume,amount,turn,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="2",  # 前复权
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=rs.fields)
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume", "amount", "turn",
                     "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["isST"] = df["isST"].astype(int)
        df = df.rename(columns={"amount": "turnover", "turn": "turnover_rate"})
        df = df.dropna(subset=["close"])
        return df

    # ── 基本面数据 ──────────────────────────────────────────

    def fetch_fundamentals(self, code: str, years: list) -> pd.DataFrame:
        """拉取盈利+成长+杜邦数据"""
        self._login()
        all_rows = []

        for year in years:
            for quarter in [1, 2, 3, 4]:
                try:
                    # 盈利
                    rs = bs.query_profit_data(code=code, year=year, quarter=quarter)
                    while rs.error_code == "0" and rs.next():
                        row = dict(zip(rs.fields, rs.get_row_data()))
                        row["year"] = year
                        row["quarter"] = quarter
                        all_rows.append(row)
                except Exception:
                    pass

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        for col in df.columns:
            if col not in ["code", "pubDate", "statDate", "year", "quarter"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def fetch_growth(self, code: str, years: list) -> pd.DataFrame:
        """拉取成长性数据"""
        self._login()
        all_rows = []
        for year in years:
            for quarter in [1, 2, 3, 4]:
                try:
                    rs = bs.query_growth_data(code=code, year=year, quarter=quarter)
                    while rs.error_code == "0" and rs.next():
                        row = dict(zip(rs.fields, rs.get_row_data()))
                        row["year"] = year
                        row["quarter"] = quarter
                        all_rows.append(row)
                except Exception:
                    pass

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        for col in df.columns:
            if col not in ["code", "pubDate", "statDate", "year", "quarter"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def fetch_dupont(self, code: str, years: list) -> pd.DataFrame:
        """拉取杜邦分析数据"""
        self._login()
        all_rows = []
        for year in years:
            for quarter in [1, 2, 3, 4]:
                try:
                    rs = bs.query_dupont_data(code=code, year=year, quarter=quarter)
                    while rs.error_code == "0" and rs.next():
                        row = dict(zip(rs.fields, rs.get_row_data()))
                        row["year"] = year
                        row["quarter"] = quarter
                        all_rows.append(row)
                except Exception:
                    pass

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        for col in df.columns:
            if col not in ["code", "pubDate", "statDate", "year", "quarter",
                           "dupontROE", "dupontAssetSto498", "dupontDebtSto498",
                           "dupontProfitMargin", "dupontAssetTurnover"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # ── 行业分类 ──────────────────────────────────────────────

    def fetch_industry(self) -> pd.DataFrame:
        """获取行业分类"""
        cache_file = self.data_dir / "industry.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)

        self._login()
        rs = bs.query_stock_industry()
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=rs.fields)
        df.to_parquet(cache_file, index=False)
        logger.info(f"Saved industry data: {len(df)} stocks")
        return df

    # ── 指数成分股 ──────────────────────────────────────────

    def fetch_index_stocks(self, index: str = "hs300") -> pd.DataFrame:
        """获取指数成分股 (hs300/zz500)"""
        cache_file = self.data_dir / f"{index}_stocks.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)

        self._login()
        if index == "hs300":
            rs = bs.query_hs300_stocks()
        elif index == "zz500":
            rs = bs.query_zz500_stocks()
        else:
            raise ValueError(f"Unknown index: {index}")

        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=rs.fields)
        df.to_parquet(cache_file, index=False)
        return df

    # ── 批量拉取 ──────────────────────────────────────────────

    def fetch_all_daily(self, start: str = "2021-01-01", end: str = None,
                        stock_codes: list = None, resume: bool = True):
        """
        批量拉取全A股日线数据
        - stock_codes: 指定股票列表，None则拉全A股
        - resume: 断点续传，跳过已拉取的股票
        """
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        daily_dir = self.data_dir / "daily"
        daily_dir.mkdir(exist_ok=True)

        if stock_codes is None:
            stock_list = self.fetch_stock_list()
            # 只取A股主板（sh.6/sh.0/sz.0/sz.3）
            stock_codes = stock_list[
                stock_list["code"].str.match(r"^(sh\.6|sz\.0|sz\.3)")
            ]["code"].tolist()

        logger.info(f"Fetching daily bars for {len(stock_codes)} stocks, {start} to {end}")

        failed = []
        for i, code in enumerate(stock_codes):
            cache_file = daily_dir / f"{code.replace('.', '_')}.parquet"

            if resume and cache_file.exists():
                existing = pd.read_parquet(cache_file)
                if len(existing) > 0 and existing["date"].max() >= pd.Timestamp(end) - timedelta(days=5):
                    continue
                # 增量更新
                last_date = existing["date"].max().strftime("%Y-%m-%d")
                new_start = (pd.Timestamp(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                if new_start >= end:
                    continue
                new_data = self.fetch_daily_bars(code, new_start, end)
                if len(new_data) > 0:
                    df = pd.concat([existing, new_data]).drop_duplicates(subset=["date"]).sort_values("date")
                    df.to_parquet(cache_file, index=False)
            else:
                df = self.fetch_daily_bars(code, start, end)
                if len(df) > 0:
                    df.to_parquet(cache_file, index=False)
                else:
                    failed.append(code)

            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i+1}/{len(stock_codes)}")
                time.sleep(1)  # 避免频率限制

        logger.info(f"Done. Failed: {len(failed)}")
        if failed:
            pd.Series(failed).to_csv(self.data_dir / "failed_daily.csv", index=False)
        return failed

    def fetch_all_fundamentals(self, start_year: int = 2021, stock_codes: list = None,
                                resume: bool = True):
        """批量拉取基本面数据"""
        fund_dir = self.data_dir / "fundamentals"
        fund_dir.mkdir(exist_ok=True)

        if stock_codes is None:
            stock_list = self.fetch_stock_list()
            stock_codes = stock_list[
                stock_list["code"].str.match(r"^(sh\.6|sz\.0|sz\.3)")
            ]["code"].tolist()

        years = list(range(start_year, datetime.now().year + 1))
        logger.info(f"Fetching fundamentals for {len(stock_codes)} stocks, years {years}")

        failed = []
        for i, code in enumerate(stock_codes):
            cache_file = fund_dir / f"{code.replace('.', '_')}.parquet"
            if resume and cache_file.exists():
                continue

            try:
                profit = self.fetch_fundamentals(code, years)
                growth = self.fetch_growth(code, years)
                dupont = self.fetch_dupont(code, years)

                # 合并
                dfs = [df for df in [profit, growth, dupont] if len(df) > 0]
                if dfs:
                    merged = dfs[0]
                    for df in dfs[1:]:
                        on_cols = [c for c in ["code", "year", "quarter"] if c in df.columns and c in merged.columns]
                        merged = merged.merge(df, on=on_cols, how="outer", suffixes=("", "_dup"))
                    merged.to_parquet(cache_file, index=False)
            except Exception as e:
                failed.append((code, str(e)))

            if (i + 1) % 100 == 0:
                logger.info(f"Fundamentals progress: {i+1}/{len(stock_codes)}")
                time.sleep(1)

        logger.info(f"Fundamentals done. Failed: {len(failed)}")
        return failed

    # ── 加载已有数据 ──────────────────────────────────────────

    def load_all_daily(self) -> pd.DataFrame:
        """加载所有已缓存的日线数据为一个大DataFrame"""
        daily_dir = self.data_dir / "daily"
        if not daily_dir.exists():
            raise FileNotFoundError(f"No daily data found at {daily_dir}")

        files = list(daily_dir.glob("*.parquet"))
        logger.info(f"Loading {len(files)} daily files...")
        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            df["code"] = f.stem.replace("_", ".")
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.sort_values(["code", "date"]).reset_index(drop=True)
        logger.info(f"Loaded {len(combined)} rows, {combined['code'].nunique()} stocks")
        return combined

    def load_all_fundamentals(self) -> pd.DataFrame:
        """加载所有基本面数据"""
        fund_dir = self.data_dir / "fundamentals"
        if not fund_dir.exists():
            raise FileNotFoundError(f"No fundamentals data at {fund_dir}")

        files = list(fund_dir.glob("*.parquet"))
        logger.info(f"Loading {len(files)} fundamental files...")
        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            if "code" not in df.columns:
                df["code"] = f.stem.replace("_", ".")
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)


# ── CLI 入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description="QuantPilot ML Data Fetcher")
    parser.add_argument("--action", choices=["daily", "fundamentals", "industry", "all"],
                        default="all", help="What to fetch")
    parser.add_argument("--start", default="2021-01-01", help="Start date for daily data")
    parser.add_argument("--end", default=None, help="End date (default: today)")
    parser.add_argument("--stocks", type=int, default=None, help="Limit to N stocks (for testing)")
    parser.add_argument("--no-resume", action="store_true", help="Force re-download")

    args = parser.parse_args()
    fetcher = MLDataFetcher()

    try:
        if args.action in ("daily", "all"):
            codes = None
            if args.stocks:
                sl = fetcher.fetch_stock_list()
                codes = sl[sl["code"].str.match(r"^(sh\.6|sz\.0|sz\.3)")]["code"].tolist()[:args.stocks]
            fetcher.fetch_all_daily(start=args.start, end=args.end,
                                     stock_codes=codes, resume=not args.no_resume)

        if args.action in ("fundamentals", "all"):
            codes = None
            if args.stocks:
                sl = fetcher.fetch_stock_list()
                codes = sl[sl["code"].str.match(r"^(sh\.6|sz\.0|sz\.3)")]["code"].tolist()[:args.stocks]
            fetcher.fetch_all_fundamentals(stock_codes=codes, resume=not args.no_resume)

        if args.action in ("industry", "all"):
            fetcher.fetch_industry()
            fetcher.fetch_index_stocks("hs300")
            fetcher.fetch_index_stocks("zz500")
    finally:
        fetcher._logout()
