"""
QuantPilot 示例: 双均线交叉策略回测

运行: cd ~/workspace/quantpilot && .venv/bin/python examples/simple_ma.py
"""

from quantpilot.core.backtest import BacktestEngine
from quantpilot.api import Strategy, FixedPosition
from quantpilot.algorithms.traditional import MACrossover, TakeProfit, StopLoss
from quantpilot.data.baostock_provider import BaostockProvider

# 1. 准备数据
print("📊 加载数据...")
provider = BaostockProvider()
stocks = ["宁德时代", "阳光电源", "隆基绿能"]
data = {}
for stock in stocks:
    df = provider.get_bars(stock, "2024-01-01", "2026-06-01")
    if not df.empty:
        data[stock] = df
        print(f"  {stock}: {len(df)} 天")
    else:
        print(f"  {stock}: 无数据")

provider._logout()

if not data:
    print("❌ 无可用数据")
    exit(1)

# 2. 定义策略
print("\n🧱 构建策略...")
strategy = Strategy(
    stocks=stocks,
    buy=MACrossover(),
    sell=[TakeProfit(), StopLoss()],
    position=FixedPosition(size=50000),
)
print(f"  买入: {strategy.buy_algorithms[0].name}")
print(f"  卖出: {[a.name for a in strategy.sell_algorithms]}")

# 3. 运行回测
print("\n📈 运行回测...")
engine = BacktestEngine(
    initial_capital=100000,
    commission_rate=0.0003,
    slippage=0.001,
    stamp_tax=0.001,
)
result = engine.run(strategy, data, "2024-01-01", "2026-06-01")

# 4. 输出结果
print(result.summary())

# 5. 交易明细
print("\n📋 最近10笔交易:")
for trade in result.trades[-10:]:
    emoji = "🟢" if trade.action == "buy" else "🔴"
    print(f"  {emoji} {trade.timestamp.strftime('%Y-%m-%d')} {trade.ticker} "
          f"{trade.action} {trade.shares}股 @{trade.price:.2f} "
          f"({trade.reason})")
