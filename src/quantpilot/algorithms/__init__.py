"""Algorithm base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from quantpilot.core import Bar, Signal


@dataclass
class ParamDef:
    """参数定义"""
    name: str
    type: str        # "int" | "float"
    default: Any
    min_val: Any = None
    max_val: Any = None
    description: str = ""


class Algorithm(ABC):
    """所有算法组件的基类"""

    name: str = ""
    description: str = ""
    category: str = ""  # "buy" | "sell" | "position"
    params: list[ParamDef] = []

    @abstractmethod
    def compute(self, bars: list[Bar], params: dict) -> Signal | None:
        """
        计算信号。
        输入: 历史bars（到当前bar为止，按时间升序）
        输出: Signal 或 None（无信号）
        """
        ...

    def get_default_params(self) -> dict:
        """返回默认参数"""
        return {p.name: p.default for p in self.params}

    def get_warmup_days(self) -> int:
        """返回预热期天数（需要多少历史数据）"""
        return 60
