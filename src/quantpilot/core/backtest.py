"""Event-driven backtest engine."""

import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from quantpilot.core import (
    Bar, Signal, Position, Trade, BacktestResult
)


class BacktestEngine:
    """
    事件驱动回测引擎。

    设计原则:
    1. 逐bar模拟真实交易流程
    2. 零未来信息泄露
    3. 真实成本（手续费+滑点+印花税）
    4. 每笔交易可审计
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_rate: float = 0.0003,   # 万三手续费
        min_commission: float = 5.0,        # 最低5元
        slippage: float = 0.001,           # 0.1%滑点
        stamp_tax: float = 0.001,          # 千一印花税（卖出）
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage = slippage
        self.stamp_tax = stamp_tax

    def run(self, strategy, data: dict, start: str, end: str) -> BacktestResult:
        """
        运行回测。

        Args:
            strategy: Strategy对象
            data: {ticker: DataFrame} 日线数据
            start: 回测起始日期
            end: 回测结束日期

        Returns:
            BacktestResult
        """
        # 初始化状态
        cash = self.initial_capital
        positions: dict[str, Position] = {}
        trades: list[Trade] = []
        equity_curve = []
        attribution = defaultdict(float)

        # 构建时间线
        all_dates = set()
        ticker_bars = {}
        for ticker, df in data.items():
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            df = df.sort_values("date")
            ticker_bars[ticker] = df
            all_dates.update(df["date"].tolist())

        timeline = sorted(all_dates)

        # 预热期（让算法有足够的历史数据）
        warmup = max(
            strategy.get_warmup_days(),
            60  # 最少60天预热
        )

        # 逐bar执行
        for i, date in enumerate(timeline):
            if i < warmup:
                # 预热期：只记录净值
                equity = cash + self._position_value(positions, ticker_bars, date)
                equity_curve.append((date, equity))
                continue

            # 1. 获取当前所有股票的bars
            current_bars = {}
            for ticker in strategy.stocks:
                if ticker not in ticker_bars:
                    continue
                df = ticker_bars[ticker]
                hist = df[df["date"] <= date]
                if len(hist) < 1:
                    continue
                row = hist.iloc[-1]
                current_bars[ticker] = Bar(
                    timestamp=date, ticker=ticker,
                    open=row["open"], high=row["high"],
                    low=row["low"], close=row["close"],
                    volume=row["volume"], turnover=row.get("turnover", 0),
                )

            # 2. 构建历史bars列表（每个ticker）
            history = {}
            for ticker in strategy.stocks:
                if ticker not in ticker_bars:
                    continue
                df = ticker_bars[ticker]
                hist = df[df["date"] <= date]
                bars = []
                for _, row in hist.iterrows():
                    bars.append(Bar(
                        timestamp=row["date"], ticker=ticker,
                        open=row["open"], high=row["high"],
                        low=row["low"], close=row["close"],
                        volume=row["volume"], turnover=row.get("turnover", 0),
                    ))
                history[ticker] = bars

            # 3. 先处理卖出信号（优先级高于买入）
            for ticker in strategy.stocks:
                if ticker not in positions or ticker not in current_bars:
                    continue

                sell_signals = strategy.check_sell(ticker, history.get(ticker, []))
                if sell_signals:
                    # 取最强的卖出信号
                    signal = max(sell_signals, key=lambda s: s.strength)
                    pos = positions[ticker]
                    price = current_bars[ticker].close * (1 - self.slippage)

                    # 计算费用
                    amount = pos.shares * price
                    commission = max(amount * self.commission_rate, self.min_commission)
                    tax = amount * self.stamp_tax
                    pnl = (price - pos.avg_cost) * pos.shares - commission - tax

                    # 执行卖出
                    cash += amount - commission - tax
                    trades.append(Trade(
                        timestamp=date, ticker=ticker, action="sell",
                        shares=pos.shares, price=price,
                        commission=commission + tax, pnl=pnl,
                        reason=signal.reason,
                    ))
                    attribution[signal.algorithm] += pnl
                    del positions[ticker]

            # 4. 处理买入信号
            for ticker in strategy.stocks:
                if ticker in positions or ticker not in current_bars:
                    continue

                buy_signals = strategy.check_buy(ticker, history.get(ticker, []))
                if buy_signals:
                    signal = max(buy_signals, key=lambda s: s.strength)

                    # 计算仓位
                    position_size = strategy.get_position_size(
                        cash, current_bars[ticker], positions
                    )
                    if position_size <= 0:
                        continue

                    price = current_bars[ticker].close * (1 + self.slippage)
                    shares = int(position_size / price / 100) * 100  # A股100股整数倍
                    if shares <= 0:
                        continue

                    amount = shares * price
                    commission = max(amount * self.commission_rate, self.min_commission)

                    if cash < amount + commission:
                        continue

                    # 执行买入
                    cash -= amount + commission
                    positions[ticker] = Position(
                        ticker=ticker, shares=shares,
                        avg_cost=price, entry_date=date,
                    )
                    trades.append(Trade(
                        timestamp=date, ticker=ticker, action="buy",
                        shares=shares, price=price,
                        commission=commission, reason=signal.reason,
                    ))

            # 5. 记录净值
            equity = cash + self._position_value(positions, ticker_bars, date)
            equity_curve.append((date, equity))

        # 6. 强制平仓（回测结束）
        for ticker, pos in list(positions.items()):
            if ticker in ticker_bars and len(ticker_bars[ticker]) > 0:
                last_row = ticker_bars[ticker].iloc[-1]
                price = last_row["close"]
                amount = pos.shares * price
                commission = max(amount * self.commission_rate, self.min_commission)
                tax = amount * self.stamp_tax
                pnl = (price - pos.avg_cost) * pos.shares - commission - tax

                cash += amount - commission - tax
                trades.append(Trade(
                    timestamp=timeline[-1], ticker=ticker, action="sell",
                    shares=pos.shares, price=price,
                    commission=commission + tax, pnl=pnl,
                    reason="回测结束强制平仓",
                ))

        # 7. 计算指标
        return self._compute_metrics(
            equity_curve, trades, attribution, timeline
        )

    def _position_value(self, positions, ticker_bars, date) -> float:
        """计算当前持仓市值"""
        total = 0.0
        for ticker, pos in positions.items():
            if ticker in ticker_bars:
                df = ticker_bars[ticker]
                hist = df[df["date"] <= date]
                if len(hist) > 0:
                    total += pos.shares * hist.iloc[-1]["close"]
        return total

    def _compute_metrics(self, equity_curve, trades, attribution, timeline):
        """计算回测指标"""
        if not equity_curve:
            return BacktestResult(
                total_return=0, annual_return=0, max_drawdown=0,
                sharpe_ratio=0, win_rate=0, profit_loss_ratio=0,
                total_trades=0, avg_holding_days=0,
            )

        equities = [e[1] for e in equity_curve]
        final_equity = equities[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # 年化收益
        days = (timeline[-1] - timeline[0]).days
        if days > 0:
            annual_return = (1 + total_return) ** (365 / days) - 1
        else:
            annual_return = 0

        # 最大回撤
        peak = equities[0]
        max_dd = 0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak
            if dd < max_dd:
                max_dd = dd

        # 夏普比
        daily_returns = [equities[i] / equities[i-1] - 1
                        for i in range(1, len(equities))]
        if daily_returns:
            avg_ret = np.mean(daily_returns)
            std_ret = np.std(daily_returns)
            sharpe = (avg_ret * 252) / (std_ret * np.sqrt(252)) if std_ret > 0 else 0
        else:
            sharpe = 0

        # 胜率和盈亏比
        sell_trades = [t for t in trades if t.action == "sell" and t.pnl != 0]
        if sell_trades:
            wins = [t for t in sell_trades if t.pnl > 0]
            losses = [t for t in sell_trades if t.pnl <= 0]
            win_rate = len(wins) / len(sell_trades)
            avg_win = np.mean([t.pnl for t in wins]) if wins else 0
            avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 1
            pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            win_rate = 0
            pl_ratio = 0

        # 平均持仓天数
        holding_days = []
        for i, t in enumerate(trades):
            if t.action == "sell":
                # 找对应的买入
                for j in range(i-1, -1, -1):
                    if trades[j].action == "buy" and trades[j].ticker == t.ticker:
                        delta = (t.timestamp - trades[j].timestamp).days
                        holding_days.append(delta)
                        break
        avg_holding = np.mean(holding_days) if holding_days else 0

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            profit_loss_ratio=pl_ratio,
            total_trades=len(sell_trades),
            avg_holding_days=avg_holding,
            equity_curve=equity_curve,
            trades=trades,
            attribution=dict(attribution),
        )
