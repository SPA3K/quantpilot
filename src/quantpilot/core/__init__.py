"""Core data classes for QuantPilot."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Bar:
    """单根K线数据"""
    timestamp: datetime
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float = 0.0


@dataclass
class Signal:
    """交易信号"""
    timestamp: datetime
    ticker: str
    action: str          # "buy" | "sell"
    strength: float      # 0-1, 信号强度
    reason: str          # 人类可读的原因
    algorithm: str = ""  # 产生信号的算法名称


@dataclass
class Position:
    """持仓"""
    ticker: str
    shares: int
    avg_cost: float
    entry_date: datetime


@dataclass
class Trade:
    """交易记录"""
    timestamp: datetime
    ticker: str
    action: str          # "buy" | "sell"
    shares: int
    price: float
    commission: float
    pnl: float = 0.0     # 平仓盈亏（卖出时）
    reason: str = ""


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_loss_ratio: float
    total_trades: int
    avg_holding_days: float
    equity_curve: list = field(default_factory=list)  # [(datetime, float)]
    trades: list = field(default_factory=list)         # [Trade]
    attribution: dict = field(default_factory=dict)    # {algorithm: pnl}

    def summary(self) -> str:
        """生成文本摘要"""
        lines = [
            "=" * 50,
            "  回测结果",
            "=" * 50,
            f"  总收益:    {self.total_return:+.2%}",
            f"  年化收益:  {self.annual_return:+.2%}",
            f"  最大回撤:  {self.max_drawdown:.2%}",
            f"  夏普比:    {self.sharpe_ratio:.2f}",
            f"  胜率:      {self.win_rate:.1%}",
            f"  盈亏比:    {self.profit_loss_ratio:.1f}:1",
            f"  交易次数:  {self.total_trades}",
            f"  平均持仓:  {self.avg_holding_days:.1f}天",
        ]
        if self.attribution:
            lines.append("")
            lines.append("  收益归因:")
            for algo, pnl in sorted(self.attribution.items(), key=lambda x: -x[1]):
                lines.append(f"    {algo}: {pnl:+,.0f}")
        lines.append("=" * 50)
        return "\n".join(lines)
