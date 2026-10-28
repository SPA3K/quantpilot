"""事件日历 — 财报/发布会/关键节点管理。

任何公司都可以配置自己的事件日历（财报日、产品发布、除权、监管节点等）。

事件权重:
  3 = 年报/季报/旗舰产品发布（高影响）
  2 = 中型活动/行业会议（中影响）
  1 = 常规公告/低权产品发布（低影响）
"""

import datetime
import json
from pathlib import Path

DEFAULT_DIR = Path.home() / ".hermes" / "scripts"


class EventCalendar:
    """事件日历管理"""

    def __init__(self, stock_code: str, base_dir: Path = None):
        self.base_dir = base_dir or DEFAULT_DIR
        self.path = self.base_dir / f"{stock_code}_events.json"

    def load(self) -> dict:
        if not self.path.exists():
            return {"past": [], "upcoming": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, data: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add_event(self, date_str: str, title: str,
                  event_type: str = "other", weight: int = 2, feeling: str = ""):
        """添加一个事件。自动归类到 past 或 upcoming。"""
        data = self.load()
        d = datetime.date.fromisoformat(date_str)
        today = datetime.date.today()
        entry = {"date": date_str, "type": event_type,
                 "title": title, "weight": weight}
        if feeling:
            entry["feeling"] = feeling
        if d < today:
            data.setdefault("past", []).append(entry)
        else:
            data.setdefault("upcoming", []).append(entry)
        self.save(data)

    def upcoming(self, within_days: int = 90) -> list[dict]:
        """未来 N 天内的事件"""
        data = self.load()
        today = datetime.date.today()
        result = []
        for e in data.get("upcoming", []):
            d = datetime.date.fromisoformat(e["date"])
            days = (d - today).days
            if 0 <= days <= within_days:
                result.append({**e, "days_from_today": days})
        return sorted(result, key=lambda x: x["days_from_today"])

    def recent_past(self, within_days: int = 60) -> list[dict]:
        """过去 N 天内的事件"""
        data = self.load()
        today = datetime.date.today()
        result = []
        for e in data.get("past", []):
            d = datetime.date.fromisoformat(e["date"])
            days_ago = (today - d).days
            if 0 <= days_ago <= within_days:
                result.append({**e, "days_ago": days_ago})
        return sorted(result, key=lambda x: -x["days_ago"])

    def nearest(self) -> tuple[int, int] | None:
        """返回 (距今天数, 权重) 中最近的事件"""
        data = self.load()
        today = datetime.date.today()
        all_events = data.get("past", []) + data.get("upcoming", [])
        if not all_events:
            return None
        best = min(
            all_events,
            key=lambda e: abs(
                (datetime.date.fromisoformat(e["date"]) - today).days
            ),
        )
        dist = abs(
            (datetime.date.fromisoformat(best["date"]) - today).days
        )
        return dist, best.get("weight", 2)
