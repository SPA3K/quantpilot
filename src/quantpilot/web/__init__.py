"""QuantPilot Web Server — FastAPI + 策略搭建器."""

import json
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional

from quantpilot.algorithms.traditional import ALL_TRADITIONAL, BUY_ALGORITHMS, SELL_ALGORITHMS
from quantpilot.core.backtest import BacktestEngine
from quantpilot.api import Strategy, FixedPosition, PercentPosition
from quantpilot.data.baostock_provider import BaostockProvider, TICKER_MAP

app = FastAPI(title="QuantPilot", version="0.3.0")

# ── Cached demo data (offline mode) ──
_DEMO_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "demo" / "stocks.json"
_demo_cache = None

def _load_demo_cache():
    global _demo_cache
    if _demo_cache is None and _DEMO_DATA_PATH.exists():
        import pandas as pd
        with open(_DEMO_DATA_PATH) as f:
            raw = json.load(f)
        _demo_cache = {}
        for name, rows in raw.items():
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["date","open","high","low","close","volume","amount"])
            for col in ["open","high","low","close","volume","amount"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["close"]).reset_index(drop=True)
            _demo_cache[name] = df
    return _demo_cache or {}

# ── API Routes ──

@app.get("/api/algorithms")
def list_algorithms():
    """列出所有可用算法组件"""
    result = []
    for algo in ALL_TRADITIONAL:
        result.append({
            "name": algo.name,
            "description": algo.description,
            "category": algo.category,
            "params": [
                {
                    "name": p.name,
                    "type": p.type,
                    "default": p.default,
                    "min": p.min_val,
                    "max": p.max_val,
                    "description": p.description,
                }
                for p in algo.params
            ],
        })
    return result


@app.get("/api/stocks")
def list_stocks():
    """列出可用股票（优先返回缓存的demo数据）"""
    demo = _load_demo_cache()
    if demo:
        return [{"name": name, "code": name, "cached": True} for name in demo.keys()]
    return [{"name": name, "code": code, "cached": False} for name, code in TICKER_MAP.items()]


class BacktestRequest(BaseModel):
    stocks: list[str]
    start: str  # YYYY-MM-DD
    end: str
    initial_capital: float = 100000
    buy_algorithms: list[dict]  # [{name: str, params: {key: value}}]
    sell_algorithms: list[dict] = []
    position_size: float = 50000  # 固定仓位金额
    position_mode: str = "fixed"  # "fixed" | "percent"
    position_percent: float = 0.3


@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    """运行回测"""
    try:
        # 1. 构建买入算法
        algo_map = {a.name: a for a in ALL_TRADITIONAL}
        buy_algos = []
        for spec in req.buy_algorithms:
            algo = algo_map.get(spec["name"])
            if not algo:
                raise HTTPException(400, f"Unknown algorithm: {spec['name']}")
            # 创建实例并设置参数
            import copy
            algo_instance = copy.deepcopy(algo)
            if spec.get("params"):
                algo_instance._forced_params = spec["params"]
            buy_algos.append(algo_instance)

        sell_algos = []
        for spec in req.sell_algorithms:
            algo = algo_map.get(spec["name"])
            if not algo:
                raise HTTPException(400, f"Unknown algorithm: {spec['name']}")
            import copy
            algo_instance = copy.deepcopy(algo)
            if spec.get("params"):
                algo_instance._forced_params = spec["params"]
            sell_algos.append(algo_instance)

        # 2. 仓位管理
        if req.position_mode == "percent":
            position = PercentPosition(req.position_percent)
        else:
            position = FixedPosition(req.position_size)

        # 3. 构建Strategy
        strategy = Strategy(
            stocks=req.stocks,
            buy=buy_algos if len(buy_algos) > 1 else buy_algos[0] if buy_algos else buy_algos,
            sell=sell_algos if sell_algos else None,
            position=position,
        )

        # 3. 拉取数据（优先用缓存）
        demo = _load_demo_cache()
        data = {}
        for stock in req.stocks:
            if stock in demo:
                # 用缓存的demo数据，按日期范围筛选
                df = demo[stock].copy()
                df = df[(df["date"] >= req.start) & (df["date"] <= req.end)]
                if not df.empty:
                    data[stock] = df
                    continue
            try:
                provider = BaostockProvider()
                df = provider.get_bars(stock, req.start, req.end)
                if not df.empty:
                    data[stock] = df
            except Exception as e:
                print(f"Warning: failed to fetch {stock}: {e}")

        if not data:
            raise HTTPException(400, "No data fetched for any stock")

        # 5. 运行回测
        engine = BacktestEngine(initial_capital=req.initial_capital)
        result = engine.run(strategy, data, req.start, req.end)

        # 6. 格式化输出
        equity_data = [
            {"date": e[0].strftime("%Y-%m-%d"), "value": round(e[1], 2)}
            for e in result.equity_curve
        ]

        trades_data = [
            {
                "date": t.timestamp.strftime("%Y-%m-%d"),
                "ticker": t.ticker,
                "action": t.action,
                "shares": t.shares,
                "price": round(t.price, 2),
                "commission": round(t.commission, 2),
                "pnl": round(t.pnl, 2),
                "reason": t.reason,
            }
            for t in result.trades
        ]

        return {
            "metrics": {
                "total_return": round(result.total_return * 100, 2),
                "annual_return": round(result.annual_return * 100, 2),
                "max_drawdown": round(result.max_drawdown * 100, 2),
                "sharpe_ratio": round(result.sharpe_ratio, 2),
                "win_rate": round(result.win_rate * 100, 1),
                "profit_loss_ratio": round(result.profit_loss_ratio, 1),
                "total_trades": result.total_trades,
                "avg_holding_days": round(result.avg_holding_days, 1),
                "initial_capital": req.initial_capital,
                "final_capital": round(equity_data[-1]["value"] if equity_data else req.initial_capital, 2),
            },
            "equity_curve": equity_data,
            "trades": trades_data,
            "attribution": result.attribution,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Backtest failed: {str(e)}")


# ── Demo Presets (pre-computed offline results) ──
_DEMO_STRATEGIES_PATH = Path(__file__).parent.parent.parent.parent / "data" / "demo" / "strategies.json"
_demo_strategies_cache = None

def _load_demo_strategies():
    global _demo_strategies_cache
    if _demo_strategies_cache is None and _DEMO_STRATEGIES_PATH.exists():
        with open(_DEMO_STRATEGIES_PATH) as f:
            _demo_strategies_cache = json.load(f)
    return _demo_strategies_cache or {}


@app.get("/api/demo/presets")
def list_demo_presets():
    """列出所有预生成的demo策略（仅返回摘要，不含曲线/交易明细）"""
    strategies = _load_demo_strategies()
    presets = []
    for key, s in strategies.items():
        presets.append({
            "id": key,
            "name": s["name"],
            "desc": s["desc"],
            "stocks": s["stocks"],
            "metrics": s["metrics"],
        })
    return presets


@app.get("/api/demo/backtest/{strategy_id}")
def get_demo_backtest(strategy_id: str):
    """返回某个预生成策略的完整回测结果（含净值曲线+交易记录）"""
    strategies = _load_demo_strategies()
    if strategy_id not in strategies:
        raise HTTPException(404, f"Strategy '{strategy_id}' not found in demo presets")
    return strategies[strategy_id]


# ── Serve Frontend ──

FRONTEND_DIR = Path(__file__).parent / "frontend"

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the main HTML page"""
    html_path = FRONTEND_DIR / "index.html"
    return FileResponse(str(html_path))


# Mount static files if directory exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def start_server(host="0.0.0.0", port=8080):
    """启动服务器"""
    import uvicorn
    print(f"🚀 QuantPilot Web Server starting on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
