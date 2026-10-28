"""内幕消息渠道 — 人机协作接口。

用户通过对话向 agent 同步非公开信息，agent 写入本地文件。
监控脚本每次运行时读取，将内幕纳入综合判断。

设计原则：
- 本地文件存储，不上传任何服务器
- 按时间倒序排列，最新消息在最上
- 每条消息标注可信度（高/中/低）
- 内幕信息仅用于内部参考，不得在公开场合引用
"""

import datetime
from pathlib import Path

DEFAULT_DIR = Path.home() / ".hermes" / "scripts"


class InsiderChannel:
    """内幕消息读写接口"""

    def __init__(self, stock_code: str, base_dir: Path = None):
        self.base_dir = base_dir or DEFAULT_DIR
        self.path = self.base_dir / f"insider_{stock_code}.md"

    def add(self, content: str, confidence: str = "中") -> str:
        """追加一条内幕消息。返回写入的行。"""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        line = f"[{now}] 可信度{confidence} | {content}"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            text = self.path.read_text(encoding="utf-8")
            # 在标记行后插入
            marker = "## 消息记录"
            if marker in text:
                idx = text.index(marker) + len(marker)
                text = text[:idx] + f"\n\n{line}" + text[idx:]
            else:
                text += f"\n{line}"
            self.path.write_text(text, encoding="utf-8")
        else:
            header = (
                f"# 内幕消息渠道 ({self.path.stem})\n\n"
                f"> ⚠️ 仅供内部参考，不得在公开场合引用。\n\n"
                f"## 消息记录\n\n{line}\n"
            )
            self.path.write_text(header, encoding="utf-8")
        return line

    def recent(self, n: int = 5) -> list[str]:
        """读取最近 n 条内幕消息"""
        if not self.path.exists():
            return []
        records = [
            l.strip()
            for l in self.path.read_text(encoding="utf-8").splitlines()
            if l.strip().startswith("[")
        ]
        return records[-n:]
