# Autoresearch — 单股模型自动优化框架

> AI agent 自动迭代实验：改代码 → 训练 → 评估 → 保留或回滚。
> 每晚自动运行，白天留给行情任务。

## 核心理念

受 [karpathy/autoresearch](https://github.com/karpathy/autoresearch) 启发：
固定评估协议 + 单一优化指标 + agent 自主决定改什么。

## 文件结构

```
autoresearch/
├── program.md      # Agent 指令（人读，也给 agent 读）
├── evaluate.py     # 固定评估协议（不可修改）
├── train.py        # 可自由修改的训练脚本（agent 每轮改这个）
├── experiments.log # 实验记录（自动追加）
└── data_cache.pkl  # 数据缓存（首次运行生成）
```

## 运行方式

```bash
# 手动跑一轮
cd autoresearch && python train.py

# 自动化运行（cron 示例：每晚21:00-08:00每小时一轮）
# 0 0-8,21-23 * * * cd /path/to/quantpilot/autoresearch && python train.py
```

## 评估指标

| 指标 | 说明 |
|------|------|
| `acc` | 整体准确率 |
| `conf_n` | 高信心样本数（score>0.55 或 <0.45） |
| `conf_acc` | 高信心子集准确率（**主要优化目标**） |
| `conf_avg_fwd_ret` | 高信心样本平均前瞻收益（实战价值） |

## Agent 每轮做什么

1. 读 `experiments.log` 看历史结果
2. 决定本轮改什么（超参数/特征/标签/模型类型）
3. 改 `train.py`
4. 运行，看指标
5. 提升→保留；下降→回滚
6. 追加记录到 `experiments.log`
