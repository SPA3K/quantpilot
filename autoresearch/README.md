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

## 运行结果 — 首期战役存档（2026-10-28 → 11-06）

首个完整跑通的战役：**小米集团-W（01810.HK）单股模型**，工作区 `xiaomi_autoresearch/`（独立评估协议，不入库）。
2026-11-06 用户决定收官，cron「小米模型夜间优化」与「autoresearch周报」已暂停（`hermes cron resume` 可恢复）。

| 项 | 结果 |
|---|---|
| 轮次 | round3 → round57（日志 51 轮；round7~10 无日志） |
| 最终基线 | **val=80.00%(15) / test_conf=54.17%(24) / fwd=-0.785% / dir_ret=+0.096%**（数据末根 2026-11-05，train/val/test=517/111/111） |
| KEEP | **15 条** = 新阻尼规则 6 条（强看涨 p≥0.6、量价背离、rsi14 中性区 Q2、10 日超卖、温和放量 Q50-75、20 日横盘 Q25-50）+ 参数 4 条（max_bin 255→63、R3 带宽 25→30 与 50→40、R1 k0.5→0.3）+ 数据滚动刷新 5 次（round6/20/32/44/47） |
| ROLLBACK | 6 条（evw3 规则、时间衰减权重、边际看跌阻尼、colsample、round24 与 round30 的 R2-k） |
| 净提升停滞 | 自 round33（第 7 条阻尼规则落地）后 **24 轮无模型净提升**，其间仅 2 次数据刷新 KEEP |
| 健康 | `pytest_round57.out`（11-06 08:07）**7 passed** |

**沉淀下来的方法学（可复用）**

1. **选择性校准有效，全局概率变换无效** —— 只在"历史上不可靠的 regime"上收缩置信度（shrink toward 0.5）；全局锐化/装袋（subsample、soft label、全局 sharpening）全部失败。
2. **阻尼必须双指标判** —— conf_acc 升但方向收益崩（如 -0.05%→-0.44%）= 否决；acc 微升且方向收益也升才 keep（`conf_avg_fwd_ret` 是方向盲的，看 direction-aware return）。
3. **落地标准 = val 入围 + 独立 30 种子双门预注册** —— 5 种子门只用于入围，test 用与生成种子不相交的 30 种子；round48 起 Gate B 要求 conf 升且效应量 ≥+0.5pp 或胜率 ≥15/30，且 dir 不降。
4. **数据滚动刷新日 keep** —— 800 根窗口纯滑动（n 恒 739），盘前/盘中不拉半日 bar；刷新带来的 ±7pp 属窗口噪声带，不回滚陈旧缓存。
5. **小样本陷阱** —— val 仅 111 样本（conf n 11~15），3 个样本 ≈ 4pp 摆动；单次 val 赢不作数，val/test 错误结构倒挂是常态。

> 完整逐轮证据链见 `xiaomi_autoresearch/experiments.log`（append-only）与 `program.md` 关闭清单。

## Agent 每轮做什么

1. 读 `experiments.log` 看历史结果
2. 决定本轮改什么（超参数/特征/标签/模型类型）
3. 改 `train.py`
4. 运行，看指标
5. 提升→保留；下降→回滚
6. 追加记录到 `experiments.log`
