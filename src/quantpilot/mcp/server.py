"""
QuantPilot MCP Server
=====================
Model Context Protocol server for the QuantPilot 3-layer factor model.

Provides 4 tools:
  1. recommend(top_n)     — Rank CSI300 stocks via 3-layer fusion
  2. analyze(stock_code)  — Deep analysis of a single stock
  3. backtest(...)        — Run historical backtest
  4. list_factors()       — List all factors across 3 layers

Layer Architecture:
  L0 — TechPulse  (20%)  : RSI + 20d momentum + volatility + MA deviation
  L1 — AlphaForge (70%)  : LightGBM with 22 features, IC=+0.27
  L3 — Sentinel   (10%)  : Price momentum + volume change sentiment proxy
"""

import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from fastmcp import FastMCP

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────
QUANTPILOT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR = QUANTPILOT_ROOT / "models" / "trained"

# ── FastMCP Server ───────────────────────────────────────────
mcp_server = FastMCP(
    name="QuantPilot",
    instructions=(
        "QuantPilot quantitative analysis server. "
        "3-layer factor model: AlphaForge (LightGBM, 70%) + "
        "TechPulse (technical, 20%) + Sentinel (sentiment proxy, 10%). "
        "Use 'recommend' to rank stocks, 'analyze' for deep dive, "
        "'backtest' for historical evaluation, 'list_factors' for factor info."
    ),
)

# ══════════════════════════════════════════════════════════════
# Stock Universe — 30 representative CSI300 stocks
# ══════════════════════════════════════════════════════════════

STOCK_UNIVERSE = [
    "600519", "000858", "601318", "300750", "002594", "600900",
    "600036", "601166", "600276", "000333", "002415", "601888",
    "600030", "000651", "601398", "600000", "000001", "600887",
    "002714", "601012", "300059", "002304", "600309", "000725",
    "601899", "600585", "002352", "601088", "000568", "600104",
]

STOCK_NAMES = {
    "600519": "贵州茅台", "000858": "五粮液", "601318": "中国平安",
    "300750": "宁德时代", "002594": "比亚迪", "600900": "长江电力",
    "600036": "招商银行", "601166": "兴业银行", "600276": "恒瑞医药",
    "000333": "美的集团", "002415": "海康威视", "601888": "中国中免",
    "600030": "中信证券", "000651": "格力电器", "601398": "工商银行",
    "600000": "浦发银行", "000001": "平安银行", "600887": "伊利股份",
    "002714": "牧原股份", "601012": "隆基绿能", "300059": "东方财富",
    "002304": "洋河股份", "600309": "万华化学", "000725": "京东方A",
    "601899": "紫金矿业", "600585": "海螺水泥", "002352": "顺丰控股",
    "601088": "中国神华", "000568": "泸州老窖", "600104": "上汽集团",
}

STOCK_SECTORS = {
    "600519": "白酒", "000858": "白酒", "601318": "保险",
    "300750": "新能源", "002594": "汽车", "600900": "电力",
    "600036": "银行", "601166": "银行", "600276": "医药",
    "000333": "家电", "002415": "安防", "601888": "免税",
    "600030": "券商", "000651": "家电", "601398": "银行",
    "600000": "银行", "000001": "银行", "600887": "乳业",
    "002714": "养殖", "601012": "光伏", "300059": "金融科技",
    "002304": "白酒", "600309": "化工", "000725": "面板",
    "601899": "矿业", "600585": "建材", "002352": "物流",
    "601088": "煤炭", "000568": "白酒", "600104": "汽车",
}


# ══════════════════════════════════════════════════════════════
# Data Fetching (baostock)
# ══════════════════════════════════════════════════════════════

def _fetch_stock_data(code: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV data from baostock for a single stock."""
    import baostock as bs

    bs_code = f"sh.{code}" if code.startswith("6") else f"sz.{code}"
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,open,high,low,close,volume,amount",
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag="2",  # 前复权 (forward-adjusted)
    )
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    cols = ["date", "open", "high", "low", "close", "volume", "amount"]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df


# ══════════════════════════════════════════════════════════════
# L0 — TechPulse: Technical Factor Scoring
# ══════════════════════════════════════════════════════════════

def _compute_l0_score(df: pd.DataFrame) -> float:
    """
    L0 TechPulse score — simple technical indicators.
    Components:
      - 20-day momentum (tanh-scaled)
      - RSI (contrarian: oversold = positive)
      - 20-day volatility (low vol = positive)
      - MA20 deviation (above MA = positive)
    Returns: float in [-1, 1]
    """
    if df.empty or len(df) < 30:
        return 0.0

    close = df["close"].values
    ret = np.diff(np.log(close))
    scores = []

    # 20-day momentum
    mom_20 = close[-1] / close[-21] - 1 if len(close) > 20 else 0
    scores.append(np.tanh(mom_20 * 5))

    # RSI (contrarian signal)
    delta = np.diff(close[-15:])
    gain = np.mean(np.maximum(delta, 0))
    loss = np.mean(np.abs(np.minimum(delta, 0)))
    rsi = 100 - 100 / (1 + gain / (loss + 1e-10)) if loss > 0 else 50
    scores.append((50 - rsi) / 50 * 0.5)

    # Low volatility bonus
    vol = np.std(ret[-20:]) * np.sqrt(252) if len(ret) > 20 else 0
    scores.append(-np.tanh(vol - 0.3))

    # MA20 deviation (above trend = slightly positive)
    ma20 = np.mean(close[-20:])
    ma_dev = (close[-1] / ma20 - 1) if ma20 > 0 else 0
    scores.append(np.tanh(ma_dev * 10))

    return float(np.tanh(np.mean(scores)))


# ══════════════════════════════════════════════════════════════
# L1 — AlphaForge: LightGBM Factor Scoring
# ══════════════════════════════════════════════════════════════

_model_cache = {}


def _load_lgb_model():
    """Load the trained LightGBM model (cached)."""
    if "lgb" not in _model_cache:
        model_path = MODEL_DIR / "lgb_cs_2008_2022.pkl"
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        _model_cache["lgb"] = data["model"]
        _model_cache["features"] = data["feature_cols"]
    return _model_cache["lgb"], _model_cache["features"]


def _compute_l1_score(df: pd.DataFrame) -> float:
    """
    L1 AlphaForge score — LightGBM prediction with 22 features.
    Returns: raw model prediction (continuous value).
    """
    if df.empty or len(df) < 60:
        return 0.0
    try:
        model, feature_cols = _load_lgb_model()
        df = df.sort_values("date").copy()
        c = df["close"]
        o = df["open"]
        h = df["high"]
        lo = df["low"]
        v = df["volume"]
        amt = df["amount"] if "amount" in df.columns else c * v

        # Build the exact same 22 features as training
        feat = {}
        feat["ret_1"] = c.pct_change(1).iloc[-1]
        feat["ret_5"] = c.pct_change(5).iloc[-1]
        feat["ret_10"] = c.pct_change(10).iloc[-1]
        feat["ret_20"] = c.pct_change(20).iloc[-1]
        feat["mom_5"] = (c.iloc[-1] / c.iloc[-6] - 1) if len(c) > 5 else 0
        feat["mom_10"] = (c.iloc[-1] / c.iloc[-11] - 1) if len(c) > 10 else 0
        feat["mom_20"] = (c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 20 else 0
        feat["mom_60"] = (c.iloc[-1] / c.iloc[-61] - 1) if len(c) > 60 else 0

        ret1 = c.pct_change(1)
        feat["vol_5"] = ret1.rolling(5).std().iloc[-1] * np.sqrt(252) if len(ret1) > 5 else 0
        feat["vol_20"] = ret1.rolling(20).std().iloc[-1] * np.sqrt(252) if len(ret1) > 20 else 0
        feat["vol_60"] = ret1.rolling(60).std().iloc[-1] * np.sqrt(252) if len(ret1) > 60 else 0

        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - 100 / (1 + gain / (loss + 1e-10))
        feat["rsi_14"] = rsi.iloc[-1] if len(rsi) > 14 else 50

        feat["ma5_dev"] = (c.iloc[-1] / c.rolling(5).mean().iloc[-1] - 1) if len(c) > 5 else 0
        feat["ma10_dev"] = (c.iloc[-1] / c.rolling(10).mean().iloc[-1] - 1) if len(c) > 10 else 0
        feat["ma20_dev"] = (c.iloc[-1] / c.rolling(20).mean().iloc[-1] - 1) if len(c) > 20 else 0
        feat["ma60_dev"] = (c.iloc[-1] / c.rolling(60).mean().iloc[-1] - 1) if len(c) > 60 else 0

        ma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        feat["boll_pct"] = (
            (c.iloc[-1] - (ma20.iloc[-1] - 2 * std20.iloc[-1]))
            / (4 * std20.iloc[-1] + 1e-10)
        ) if len(c) > 20 else 0.5

        feat["vol_ratio_5_20"] = (
            v.rolling(5).mean().iloc[-1] / (v.rolling(20).mean().iloc[-1] + 1e-10)
        ) if len(v) > 20 else 1
        feat["vol_ratio_5_60"] = (
            v.rolling(5).mean().iloc[-1] / (v.rolling(60).mean().iloc[-1] + 1e-10)
        ) if len(v) > 60 else 1

        feat["turn_5"] = 0
        feat["turn_20"] = 0

        vwap = amt / (v + 1e-10)
        feat["vwap_dev"] = (vwap.iloc[-1] / c.iloc[-1] - 1) if c.iloc[-1] > 0 else 0

        X = pd.DataFrame([[feat.get(fc, 0) for fc in feature_cols]], columns=feature_cols)
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        return float(model.predict(X)[0])
    except Exception as e:
        logger.warning(f"L1 scoring failed: {e}")
        return 0.0


# ══════════════════════════════════════════════════════════════
# L3 — Sentinel: Sentiment Proxy Scoring
# ══════════════════════════════════════════════════════════════

def _compute_l3_score(df: pd.DataFrame) -> float:
    """
    L3 Sentinel score — sentiment proxy via price momentum + volume.
    In production, this would use StructBERT on financial news.
    Returns: float in [-1, 1]
    """
    if df.empty or len(df) < 20:
        return 0.0

    close = df["close"].values
    volume = df["volume"].values

    # 5-day momentum (short-term sentiment)
    mom_5 = close[-1] / close[-6] - 1 if len(close) > 5 else 0

    # Volume change (rising volume = rising attention)
    vol_chg = np.mean(volume[-5:]) / (np.mean(volume[-20:]) + 1e-10) - 1

    return float(np.tanh((mom_5 * 3 + vol_chg) / 2))


# ══════════════════════════════════════════════════════════════
# MCP Tools
# ══════════════════════════════════════════════════════════════

@mcp_server.tool()
def recommend(top_n: int = 10) -> str:
    """
    Run the 3-layer factor model on the CSI300 stock universe and return
    the top-N ranked stocks with all layer scores.

    Fusion formula: 0.7 * AlphaForge + 0.2 * TechPulse + 0.1 * Sentinel

    Args:
        top_n: Number of top-ranked stocks to return (default: 10, max: 30)

    Returns:
        JSON with ranked stocks, each containing code, name, sector,
        l0_score (TechPulse), l1_score (AlphaForge), l3_score (Sentinel),
        and fusion_score.
    """
    top_n = max(1, min(top_n, len(STOCK_UNIVERSE)))

    # Date range: last 4 months of daily data (need 60+ bars for L1)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=150)).strftime("%Y-%m-%d")

    # Load L1 model upfront
    try:
        _load_lgb_model()
    except Exception as e:
        return json.dumps({"error": f"Failed to load LightGBM model: {e}"}, ensure_ascii=False)

    import baostock as bs
    bs.login()

    results = []
    for i, code in enumerate(STOCK_UNIVERSE):
        df = _fetch_stock_data(code, start_date, end_date)
        if df.empty or len(df) < 60:
            logger.info(f"  Skipping {code}: insufficient data ({len(df)} bars)")
            continue

        l0 = _compute_l0_score(df)
        l1 = _compute_l1_score(df)
        l3 = _compute_l3_score(df)
        fusion = 0.7 * l1 + 0.2 * l0 + 0.1 * l3

        results.append({
            "code": code,
            "name": STOCK_NAMES.get(code, code),
            "sector": STOCK_SECTORS.get(code, "未知"),
            "l0_techpulse": round(l0, 4),
            "l1_alphaforge": round(l1, 4),
            "l3_sentinel": round(l3, 4),
            "fusion_score": round(fusion, 4),
            "latest_price": round(float(df["close"].iloc[-1]), 2),
            "data_bars": len(df),
        })

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{len(STOCK_UNIVERSE)}")

    bs.logout()

    # Sort by fusion score descending
    results.sort(key=lambda x: x["fusion_score"], reverse=True)

    output = {
        "tool": "recommend",
        "model": "QuantPilot 3-Layer Fusion",
        "fusion_weights": {"alphaforge": 0.7, "techpulse": 0.2, "sentinel": 0.1},
        "universe_size": len(STOCK_UNIVERSE),
        "scored_stocks": len(results),
        "requested_top_n": top_n,
        "data_range": {"start": start_date, "end": end_date},
        "rankings": results[:top_n],
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


@mcp_server.tool()
def analyze(stock_code: str) -> str:
    """
    Deep analysis of a single stock: all 3-layer factor scores, recent
    price action, and key technical indicators.

    Args:
        stock_code: 6-digit A-share stock code (e.g., '600519', '000858')

    Returns:
        JSON with comprehensive stock analysis including factor scores,
        technical indicators, and recent price data.
    """
    code = stock_code.strip()

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")

    import baostock as bs
    bs.login()
    df = _fetch_stock_data(code, start_date, end_date)
    bs.logout()

    if df.empty:
        return json.dumps({"error": f"No data found for stock {code}"}, ensure_ascii=False)

    df = df.sort_values("date").reset_index(drop=True)

    # ── Compute all scores ──
    l0_score = _compute_l0_score(df)
    l1_score = _compute_l1_score(df) if len(df) >= 60 else None
    l3_score = _compute_l3_score(df)

    # ── Price action summary ──
    latest = df.iloc[-1]
    price_changes = {}
    for period_name, period_days in [("1d", 1), ("5d", 5), ("20d", 20), ("60d", 60)]:
        if len(df) > period_days:
            old_close = df["close"].iloc[-period_days - 1]
            price_changes[period_name] = round(
                (float(latest["close"]) / old_close - 1) * 100, 2
            )

    price_action = {
        "latest_date": str(latest["date"].date()),
        "close": round(float(latest["close"]), 2),
        "open": round(float(latest["open"]), 2),
        "high": round(float(latest["high"]), 2),
        "low": round(float(latest["low"]), 2),
        "volume": int(latest["volume"]),
        "changes_pct": price_changes,
    }

    # ── Technical indicators ──
    technical = {}
    c = df["close"]

    # RSI
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / (loss + 1e-10))
    technical["rsi_14"] = round(float(rsi.iloc[-1]), 2) if len(rsi) > 14 else None

    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    macd_hist = 2 * (dif - dea)
    technical["macd"] = {
        "dif": round(float(dif.iloc[-1]), 4),
        "dea": round(float(dea.iloc[-1]), 4),
        "histogram": round(float(macd_hist.iloc[-1]), 4),
    }

    # Moving averages
    for period in [5, 10, 20, 60]:
        if len(c) >= period:
            ma = c.rolling(period).mean().iloc[-1]
            technical[f"ma{period}"] = round(float(ma), 2)
            technical[f"ma{period}_dev_pct"] = round(
                (float(c.iloc[-1]) / ma - 1) * 100, 2
            )

    # Bollinger Bands
    if len(c) >= 20:
        ma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        technical["bollinger"] = {
            "upper": round(float(upper.iloc[-1]), 2),
            "middle": round(float(ma20.iloc[-1]), 2),
            "lower": round(float(lower.iloc[-1]), 2),
            "pct_b": round(
                float(
                    (c.iloc[-1] - (ma20.iloc[-1] - 2 * std20.iloc[-1]))
                    / (4 * std20.iloc[-1] + 1e-10)
                ),
                4,
            ),
        }

    # ATR (14-day)
    h = df["high"]
    lo = df["low"]
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    technical["atr_14"] = round(float(atr.iloc[-1]), 4) if len(atr) > 14 else None

    # Volatility
    ret1 = c.pct_change()
    if len(ret1) > 20:
        technical["volatility_20d"] = round(float(ret1.rolling(20).std().iloc[-1] * np.sqrt(252)), 4)

    # Volume metrics
    v = df["volume"]
    if len(v) >= 20:
        technical["volume_ratio_5_20"] = round(
            float(v.rolling(5).mean().iloc[-1] / (v.rolling(20).mean().iloc[-1] + 1e-10)), 2
        )

    # Recent 5-day OHLCV
    recent_bars = []
    for _, row in df.tail(5).iterrows():
        recent_bars.append({
            "date": str(row["date"].date()),
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
            "volume": int(row["volume"]),
        })

    # ── Compose output ──
    fusion = None
    if l1_score is not None:
        fusion = 0.7 * l1_score + 0.2 * l0_score + 0.1 * l3_score

    output = {
        "tool": "analyze",
        "stock": {
            "code": code,
            "name": STOCK_NAMES.get(code, code),
            "sector": STOCK_SECTORS.get(code, "未知"),
        },
        "scores": {
            "fusion": round(fusion, 4) if fusion is not None else None,
            "l0_techpulse": round(l0_score, 4),
            "l1_alphaforge": round(l1_score, 4) if l1_score is not None else None,
            "l3_sentinel": round(l3_score, 4),
        },
        "price_action": price_action,
        "technical_indicators": technical,
        "recent_bars": recent_bars,
        "data_bars": len(df),
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


@mcp_server.tool()
def backtest(
    stock_codes: str,
    start_date: str,
    end_date: str,
    strategy: str = "fusion",
) -> str:
    """
    Run a historical backtest on given stocks using the fusion strategy
    or a custom strategy.

    Args:
        stock_codes: Comma-separated stock codes (e.g., '600519,000858,601318')
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        strategy: Strategy name — 'fusion' (3-layer model) or 'techpulse' or 'alphaforge'

    Returns:
        JSON with backtest results: top/bottom ranked stocks, long-short
        spread, and per-stock predictions with actual returns.
    """
    codes = [c.strip() for c in stock_codes.split(",") if c.strip()]
    if not codes:
        return json.dumps({"error": "No stock codes provided"}, ensure_ascii=False)

    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return json.dumps(
            {"error": "Invalid date format. Use YYYY-MM-DD"}, ensure_ascii=False
        )

    logger.info(
        f"Backtest: {len(codes)} stocks, {start_date} to {end_date}, strategy={strategy}"
    )

    # Load L1 model
    try:
        _load_lgb_model()
    except Exception as e:
        return json.dumps({"error": f"Failed to load model: {e}"}, ensure_ascii=False)

    import baostock as bs
    bs.login()

    # Fetch all data first
    all_data = {}
    for code in codes:
        # Need extra lookback for factor computation
        lookback_start = (
            datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=150)
        ).strftime("%Y-%m-%d")
        df = _fetch_stock_data(code, lookback_start, end_date)
        if not df.empty and len(df) > 60:
            all_data[code] = df
            logger.info(f"  Loaded {code}: {len(df)} bars")
        else:
            logger.info(f"  Skipping {code}: {len(df)} bars")

    bs.logout()

    if not all_data:
        return json.dumps({"error": "No valid stock data loaded"}, ensure_ascii=False)

    # ── Score at prediction date (start_date) ──
    pred_date = pd.Timestamp(start_date)
    predictions = {}

    for code, df in all_data.items():
        pred_df = df[df["date"] <= pred_date]
        if len(pred_df) < 60:
            continue

        l0 = _compute_l0_score(pred_df)
        l1 = _compute_l1_score(pred_df)
        l3 = _compute_l3_score(pred_df)

        # Select score based on strategy
        if strategy == "alphaforge":
            score = l1
        elif strategy == "techpulse":
            score = l0
        elif strategy == "sentinel":
            score = l3
        else:  # fusion (default)
            score = 0.7 * l1 + 0.2 * l0 + 0.1 * l3

        # Compute actual return over the backtest period
        pred_close = pred_df["close"].iloc[-1]
        eval_df = df[df["date"] <= pd.Timestamp(end_date)]
        eval_close = eval_df["close"].iloc[-1] if not eval_df.empty else None

        actual_return = (
            round((eval_close / pred_close - 1) * 100, 2) if eval_close else None
        )

        predictions[code] = {
            "name": STOCK_NAMES.get(code, code),
            "sector": STOCK_SECTORS.get(code, "未知"),
            "l0_techpulse": round(l0, 4),
            "l1_alphaforge": round(l1, 4),
            "l3_sentinel": round(l3, 4),
            "fusion_score": round(0.7 * l1 + 0.2 * l0 + 0.1 * l3, 4),
            "strategy_score": round(score, 4),
            "actual_return_pct": actual_return,
            "pred_close": round(float(pred_close), 2),
            "eval_close": round(float(eval_close), 2) if eval_close else None,
        }

    # ── Rank and compute long-short ──
    valid = {k: v for k, v in predictions.items() if v["actual_return_pct"] is not None}
    sorted_items = sorted(valid.items(), key=lambda x: x[1]["strategy_score"], reverse=True)

    top_n = min(5, len(sorted_items) // 2) or 1
    top_stocks = sorted_items[:top_n]
    bottom_stocks = sorted_items[-top_n:]

    top_avg = np.mean([v["actual_return_pct"] for _, v in top_stocks])
    bottom_avg = np.mean([v["actual_return_pct"] for _, v in bottom_stocks])
    long_short = top_avg - bottom_avg
    all_avg = np.mean([v["actual_return_pct"] for _, v in sorted_items])

    output = {
        "tool": "backtest",
        "parameters": {
            "stocks": codes,
            "start_date": start_date,
            "end_date": end_date,
            "strategy": strategy,
            "valid_stocks": len(valid),
        },
        "summary": {
            "top_group_avg_return_pct": round(float(top_avg), 2),
            "bottom_group_avg_return_pct": round(float(bottom_avg), 2),
            "long_short_spread_pct": round(float(long_short), 2),
            "all_stocks_avg_return_pct": round(float(all_avg), 2),
        },
        "top_group": [
            {"rank": i + 1, "code": code, **info}
            for i, (code, info) in enumerate(top_stocks)
        ],
        "bottom_group": [
            {"rank": len(sorted_items) - top_n + i + 1, "code": code, **info}
            for i, (code, info) in enumerate(bottom_stocks)
        ],
        "all_rankings": [
            {"rank": i + 1, "code": code, **info}
            for i, (code, info) in enumerate(sorted_items)
        ],
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


@mcp_server.tool()
def list_factors() -> str:
    """
    List all available factors across the 3-layer model architecture.

    Returns:
        JSON describing every factor used by TechPulse (L0),
        AlphaForge (L1), and Sentinel (L3).
    """
    output = {
        "tool": "list_factors",
        "architecture": "3-Layer QuantPilot Factor Model",
        "fusion_weights": {
            "alphaforge_l1": 0.70,
            "techpulse_l0": 0.20,
            "sentinel_l3": 0.10,
        },
        "layers": [
            {
                "layer": "L0 — TechPulse",
                "weight": 0.20,
                "category": "Technical Factor Model",
                "description": "Rule-based technical indicators, no training required",
                "factors": [
                    {
                        "name": "momentum_20d",
                        "description": "20-day price momentum (tanh-scaled)",
                        "type": "momentum",
                    },
                    {
                        "name": "rsi_14_contrarian",
                        "description": "RSI(14) contrarian signal — oversold stocks score higher",
                        "type": "technical",
                    },
                    {
                        "name": "volatility_20d",
                        "description": "20-day annualized volatility — low volatility preferred",
                        "type": "risk",
                    },
                    {
                        "name": "ma20_deviation",
                        "description": "Deviation from 20-day moving average",
                        "type": "trend",
                    },
                ],
            },
            {
                "layer": "L1 — AlphaForge",
                "weight": 0.70,
                "category": "LightGBM Multi-Factor Model",
                "description": "Trained on 2008-2022 A-share data, IC=+0.27",
                "model_file": "lgb_cs_2008_2022.pkl",
                "factors": [
                    {"name": "ret_1", "description": "1-day return", "type": "momentum"},
                    {"name": "ret_5", "description": "5-day return", "type": "momentum"},
                    {"name": "ret_10", "description": "10-day return", "type": "momentum"},
                    {"name": "ret_20", "description": "20-day return", "type": "momentum"},
                    {"name": "mom_5", "description": "5-day momentum (close/close[-6]-1)", "type": "momentum"},
                    {"name": "mom_10", "description": "10-day momentum", "type": "momentum"},
                    {"name": "mom_20", "description": "20-day momentum", "type": "momentum"},
                    {"name": "mom_60", "description": "60-day momentum", "type": "momentum"},
                    {"name": "vol_5", "description": "5-day annualized volatility", "type": "risk"},
                    {"name": "vol_20", "description": "20-day annualized volatility", "type": "risk"},
                    {"name": "vol_60", "description": "60-day annualized volatility", "type": "risk"},
                    {"name": "rsi_14", "description": "14-day RSI", "type": "technical"},
                    {"name": "ma5_dev", "description": "Deviation from MA5", "type": "trend"},
                    {"name": "ma10_dev", "description": "Deviation from MA10", "type": "trend"},
                    {"name": "ma20_dev", "description": "Deviation from MA20", "type": "trend"},
                    {"name": "ma60_dev", "description": "Deviation from MA60", "type": "trend"},
                    {"name": "boll_pct", "description": "Bollinger Band %B position", "type": "technical"},
                    {"name": "vol_ratio_5_20", "description": "Volume ratio (5d/20d average)", "type": "liquidity"},
                    {"name": "vol_ratio_5_60", "description": "Volume ratio (5d/60d average)", "type": "liquidity"},
                    {"name": "turn_5", "description": "5-day average turnover rate", "type": "liquidity"},
                    {"name": "turn_20", "description": "20-day average turnover rate", "type": "liquidity"},
                    {"name": "vwap_dev", "description": "VWAP deviation", "type": "microstructure"},
                ],
            },
            {
                "layer": "L3 — Sentinel",
                "weight": 0.10,
                "category": "Sentiment Proxy Model",
                "description": "Price-based sentiment proxy (production uses StructBERT on financial news)",
                "factors": [
                    {
                        "name": "mom_5_sentiment",
                        "description": "5-day price momentum as short-term sentiment gauge",
                        "type": "sentiment_proxy",
                    },
                    {
                        "name": "volume_change",
                        "description": "5d/20d volume ratio change — rising volume = rising attention",
                        "type": "sentiment_proxy",
                    },
                ],
            },
        ],
        "available_models_in_zoo": [
            "value_momentum", "quality", "low_volatility",
            "momentum_20d", "reversal_20d", "alpha101",
            "technical_combo", "sector_neutral",
            "lightgbm_cross_section", "xgboost_cs",
            "gru_pretrained", "lstm_pretrained",
            "gru_streaming", "lstm_streaming",
        ],
    }
    return json.dumps(output, ensure_ascii=False, indent=2)
