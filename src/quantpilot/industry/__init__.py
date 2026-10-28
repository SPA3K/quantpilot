"""QuantPilot 行业监控模式 — 板块轮动扫描与信号追踪。

三个组件：
- market_monitor:  盘中行情采集 + 技术指标 + 交易日历（结构化输出）
- sector_signal:   五板块（半导体/CPO/具身/科技宽基/银行）龙头股 ML 信号聚合
- signal_tracker:  逐日记录预测 vs 实际，统计方向准确率与累计收益

数据目录：$QUANTPILOT_DATA_DIR（默认 ~/.quantpilot/industry/），
状态文件与日线缓存均落在该目录，不污染仓库。
"""

__version__ = "0.1.0"
