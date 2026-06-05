"""预生成demo回测数据 — 离线展示用."""
import json, sys, os, copy
sys.path.insert(0, os.path.expanduser('~/workspace/quantpilot/src'))

import pandas as pd
from quantpilot.core.backtest import BacktestEngine
from quantpilot.api import Strategy, FixedPosition
from quantpilot.algorithms.traditional import (
    MACrossover, RSIAlgorithm, StopLoss, TakeProfit,
    MACDAlgorithm, TurtleBreakout, BollingerBands
)

with open(os.path.expanduser('~/workspace/quantpilot/data/demo/stocks.json')) as f:
    raw = json.load(f)

demo_stocks = {}
for name in ["宁德时代", "贵州茅台", "比亚迪", "中国平安", "招商银行"]:
    if name not in raw:
        continue
    rows = [r for r in raw[name] if r[0] >= "2023-01-01"]
    df = pd.DataFrame(rows, columns=["date","open","high","low","close","volume","amount"])
    for col in ["open","high","low","close","volume","amount"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df["date"] = pd.to_datetime(df["date"])
    demo_stocks[name] = df

print(f"加载了 {len(demo_stocks)} 只股票")

strategy_presets = [
    {"name": "双均线+止损", "desc": "经典趋势跟踪：MA5上穿MA20买入，跌破止损",
     "buy": MACrossover, "buy_p": {"fast_period": 5, "slow_period": 20},
     "sell": StopLoss, "sell_p": {"max_loss": -0.08},
     "stocks": ["宁德时代", "贵州茅台"]},
    {"name": "RSI超卖反弹", "desc": "RSI<30超卖买入，>70超买卖出",
     "buy": RSIAlgorithm, "buy_p": {"period": 14, "oversold": 30, "overbought": 70},
     "sell": TakeProfit, "sell_p": {"target_return": 0.15},
     "stocks": ["比亚迪", "宁德时代"]},
    {"name": "海龟突破", "desc": "突破20日高点买入，跌破10日低点卖出",
     "buy": TurtleBreakout, "buy_p": {"entry_period": 20, "exit_period": 10},
     "sell": StopLoss, "sell_p": {"max_loss": -0.1},
     "stocks": ["贵州茅台", "招商银行"]},
    {"name": "MACD金叉", "desc": "DIF上穿DEA买入，下穿卖出",
     "buy": MACDAlgorithm, "buy_p": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
     "sell": StopLoss, "sell_p": {"max_loss": -0.07},
     "stocks": ["比亚迪", "中国平安"]},
    {"name": "布林带策略", "desc": "触及下轨买入，触及上轨卖出",
     "buy": BollingerBands, "buy_p": {"period": 20, "num_std": 2.0},
     "sell": StopLoss, "sell_p": {"max_loss": -0.06},
     "stocks": ["宁德时代", "招商银行"]},
]

results = {}
engine = BacktestEngine(initial_capital=100000)

for p in strategy_presets:
    name = p["name"]
    algo_buy = copy.deepcopy(p["buy"]())
    algo_buy._forced_params = p["buy_p"]
    algo_sell = copy.deepcopy(p["sell"]())
    algo_sell._forced_params = p["sell_p"]

    strategy = Strategy(
        stocks=p["stocks"],
        buy=algo_buy,
        sell=[algo_sell],
        position=FixedPosition(50000),
    )
    data = {s: demo_stocks[s] for s in p["stocks"] if s in demo_stocks}
    if not data:
        print(f"❌ {name}: no data"); continue

    try:
        result = engine.run(strategy, data, "2023-01-01", "2025-05-30")
        equity = [{"date": e[0].strftime("%Y-%m-%d"), "value": round(e[1], 2)} for e in result.equity_curve]
        trades = [{"date": t.timestamp.strftime("%Y-%m-%d"), "ticker": t.ticker,
                    "action": t.action, "shares": t.shares, "price": round(t.price, 2),
                    "commission": round(t.commission, 2), "pnl": round(t.pnl, 2),
                    "reason": t.reason} for t in result.trades]
        m = {
            "total_return": round(result.total_return * 100, 2),
            "annual_return": round(result.annual_return * 100, 2),
            "max_drawdown": round(result.max_drawdown * 100, 2),
            "sharpe_ratio": round(result.sharpe_ratio, 2),
            "win_rate": round(result.win_rate * 100, 1),
            "profit_loss_ratio": round(result.profit_loss_ratio, 1),
            "total_trades": result.total_trades,
            "avg_holding_days": round(result.avg_holding_days, 1),
            "final_capital": round(equity[-1]["value"] if equity else 100000, 2),
            "initial_capital": 100000,
        }
        results[name] = {"name": name, "desc": p["desc"], "stocks": p["stocks"],
                         "metrics": m, "equity_curve": equity, "trades": trades}
        print(f"✅ {name}: 收益{m['total_return']}% 回撤{m['max_drawdown']}% 夏普{m['sharpe_ratio']} {m['total_trades']}笔")
    except Exception as e:
        print(f"❌ {name}: {e}")

with open(os.path.expanduser('~/workspace/quantpilot/data/demo/strategies.json'), 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n保存 {len(results)} 组策略到 data/demo/strategies.json")
