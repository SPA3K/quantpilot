"""Traditional algorithm components — 12个免费策略积木."""

from quantpilot.algorithms.traditional.ma_crossover import MACrossover
from quantpilot.algorithms.traditional.rsi import RSIAlgorithm
from quantpilot.algorithms.traditional.turtle import TurtleBreakout
from quantpilot.algorithms.traditional.take_profit import TakeProfit
from quantpilot.algorithms.traditional.stop_loss import StopLoss
from quantpilot.algorithms.traditional.macd import MACDAlgorithm
from quantpilot.algorithms.traditional.bollinger import BollingerBands
from quantpilot.algorithms.traditional.kdj import KDJAlgorithm
from quantpilot.algorithms.traditional.volume_price import VolumePrice
from quantpilot.algorithms.traditional.obv import OBVAlgorithm
from quantpilot.algorithms.traditional.atr_trailing import ATRTrailingStop
from quantpilot.algorithms.traditional.grid import GridTrading

ALL_TRADITIONAL = [
    MACrossover(),
    RSIAlgorithm(),
    MACDAlgorithm(),
    BollingerBands(),
    KDJAlgorithm(),
    TurtleBreakout(),
    VolumePrice(),
    OBVAlgorithm(),
    ATRTrailingStop(),
    GridTrading(),
    TakeProfit(),
    StopLoss(),
]

# 按类别分组
BUY_ALGORITHMS = [a for a in ALL_TRADITIONAL if a.category == "buy"]
SELL_ALGORITHMS = [a for a in ALL_TRADITIONAL if a.category == "sell"]

__all__ = [
    "MACrossover", "RSIAlgorithm", "MACDAlgorithm",
    "BollingerBands", "KDJAlgorithm", "TurtleBreakout",
    "VolumePrice", "OBVAlgorithm", "ATRTrailingStop",
    "GridTrading", "TakeProfit", "StopLoss",
    "ALL_TRADITIONAL", "BUY_ALGORITHMS", "SELL_ALGORITHMS",
]
