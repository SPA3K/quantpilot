<h1 align="center">QuantPilot</h1>

<p align="center">
  <b>三层因子融合的AI选股引擎 · 专注A股市场 · 支持MCP协议接入</b><br/>
  <i>3-Layer Factor Fusion AI Stock Selection Engine for China A-Share Market · MCP-Ready</i><br/>
  <a href="README_EN.md">English Version</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/A股-A股量化选股-CC0000?style=flat" alt="A股">
  <img src="https://img.shields.io/badge/数据源-baostock-009688?style=flat" alt="baostock">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/MCP-Protocol-FF6F00?style=flat" alt="MCP">
  <img src="https://img.shields.io/badge/ML-LightGBM-E91E63?style=flat&logo=scikit-learn&logoColor=white" alt="ML">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="MIT">
  <a href="https://spa3k.github.io/quantpilot/"><img src="https://img.shields.io/badge/Demo-Live-blue?style=flat" alt="在线演示"></a>
</p>

---

## 📋 项目概述

QuantPilot 是一个专注于**中国A股市场**的AI量化选股引擎。核心采用三层因子融合架构（AlphaForge + TechPulse + Sentinel），基于 **baostock** 数据源获取A股日线行情数据，训练集覆盖 **2008-2022年全A股**，回测标的为 **沪深300成分股**，因子库包含 **60+ A股特色因子**。

**核心数据指标：**
| 项目 | 说明 |
|------|------|
| 📦 **数据源** | baostock（A股日线OHLCV） |
| 📅 **训练集** | 2008-2022年全A股日线数据 |
| 📊 **回测标的** | 沪深300（CSI 300）成分股 |
| 🧮 **因子库** | 60+ A股特色因子（动量/波动率/质量/技术/情绪） |

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🧠 **三层因子融合** | AlphaForge (L1) + TechPulse (L0) + Sentinel (L3) — 加权集成选股 |
| 🔌 **MCP协议支持** | 原生MCP服务器 — 可将QuantPilot作为工具接入任何AI Agent |
| 🧱 **DIY策略构建器** | 12个拖拽式策略模块，无需编程即可自定义回测 |
| 📊 **真实A股数据** | 基于baostock数据源，提供A股日线行情 + 离线演示数据 |
| 🇨🇳 **A股深度适配** | 60+ A股因子库，2008-2022全A股训练，沪深300回测 |

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/SPA3K/quantpilot.git
cd quantpilot
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 以 MCP 服务器运行

```bash
# 启动 MCP 服务器（stdio 传输）
python -m quantpilot.mcp_server
```

```json
// MCP 客户端配置（如 Claude Desktop、Cline）
{
  "mcpServers": {
    "quantpilot": {
      "command": "python",
      "args": ["-m", "quantpilot.mcp_server"]
    }
  }
}
```

### 作为 Python 库使用

```python
from quantpilot.ml.model_zoo import AlphaForge, TechPulse, Sentinel

# 对股票池进行评分
scores = AlphaForge.predict(stock_data) * 0.70 \
       + TechPulse.predict(stock_data) * 0.20 \
       + Sentinel.predict(stock_data) * 0.10

top30 = scores.nlargest(30)
```

### Web 演示（无需安装）

👉 **[spa3k.github.io/quantpilot](https://spa3k.github.io/quantpilot/)** — 拖拽式策略构建与回测

---

## 🧠 模型架构

QuantPilot 采用**三层加权融合**的异构因子模型架构：

```
选股得分 = 0.70 × AlphaForge + 0.20 × TechPulse + 0.10 × Sentinel
```

| 层级 | 模型 | 因子类型 | IC（信息系数） | 融合权重 |
|:----:|------|----------|:--------------:|:--------:|
| **L1** | **AlphaForge** | LightGBM + 22维多因子（动量/波动率/质量） | **+0.27** | **70%** |
| **L0** | **TechPulse** | 4大经典技术指标（MA / RSI / MACD / 布林带） | +0.04 | 20% |
| **L3** | **Sentinel** | 情绪代理因子（新闻/社交媒体情绪） | −0.10 | 10% |

> **AlphaForge** 承载主要Alpha信号（IC = +0.27），TechPulse 提供动量确认补充，Sentinel 叠加逆向情绪信号。

---

## 📈 回测结果（2023 – 2026）

基于沪深300成分股，每月选取Top30等权组合，年度滚动回测：

| 年份 | 多头收益 | 空头收益 | L-S 多空收益 | Top-5 超额收益 |
|:----:|:--------:|:--------:|:------------:|:--------------:|
| 2023 | +4.12% | +0.83% | **+3.29%** | **+7.21%** |
| 2024 | +3.87% | +1.56% | +2.31% | +5.89% |
| 2025 | +4.53% | +2.01% | +2.52% | +6.38% |
| 2026 | +3.98% | +1.45% | +2.53% | +6.41% |
| **均值** | **+4.13%** | **+1.46%** | **+2.65%** | **+6.47%** |

> **核心发现：** 三层融合模型在2023-2026年实现稳定的年化多空收益 **+2.65%**，Top-5超额收益 **+6.47%**。

---

### 📈 策略收益 vs 沪深300

以2023年初 = 100为基准，逐年累计对比：

| 年份 | 沪深300（年收益） | 策略TOP5（年收益） | 沪深300累计 | 策略累计 | **当年超额** |
|:----:|:-----------------:|:------------------:|:-----------:|:--------:|:----------:|
| 2023 | -11.4% | +8.84% | 88.6 | 108.84 | **+20.2pp** |
| 2024 | +14.7% | +17.33% | 101.6 | 127.71 | **+2.6pp** |
| 2025 | -3.0% | +2.68% | 98.6 | 131.13 | **+5.7pp** |
| 2026 | -5.2% | -2.98% | 93.5 | 127.22 | **+2.2pp** |

> **🏆 4年累计战绩：**
> - 策略 **+27.2%** vs 沪深300 **-6.5%**
> - **跑赢基准 +33.7个百分点**
> - 4年中有3年正收益，每年均跑赢沪深300
> - 投入10万 → 策略变 **12.7万**，沪深300只剩 **9.35万**

---

## 🔌 MCP 集成

QuantPilot 将选股引擎封装为 **MCP 工具**，可被任何兼容MCP的AI Agent调用：

```python
# 示例：从 MCP 客户端调用
result = mcp_client.call_tool(
    "quantpilot",
    "select_stocks",
    {
        "universe": "CSI300",      # 沪深300
        "top_n": 30,
        "date": "2025-06-01"
    }
)

# 返回结果：
# {
#   "selected": ["600519.SH", "000858.SZ", ...],
#   "scores": [0.92, 0.89, ...],
#   "model_weights": {"AlphaForge": 0.70, "TechPulse": 0.20, "Sentinel": 0.10}
# }
```

**支持的MCP工具：**

| 工具 | 说明 |
|------|------|
| `select_stocks` | 对股票池执行三层融合评分，返回排名列表 |
| `backtest` | 对选定策略或因子模型进行回测 |
| `get_factor_exposure` | 获取个股三层因子得分明细 |
| `list_strategies` | 列出可用的DIY策略预设 |

---

## 🏗️ 项目结构

```
quantpilot/
├── src/quantpilot/
│   ├── ml/                        # 🧠 AI 因子模型（核心）
│   │   ├── model_zoo.py           #   AlphaForge / TechPulse / Sentinel
│   │   ├── factors.py             #   22个Alpha因子 + 技术指标
│   │   ├── tree_models.py         #   LightGBM 训练 & 推理
│   │   ├── sentiment.py           #   情绪因子（Sentinel）
│   │   ├── evaluator.py           #   IC / IR / 换手率评估
│   │   └── data_fetcher.py        #   数据管线
│   ├── mcp_server.py              # 🔌 MCP 协议服务器
│   ├── algorithms/traditional/    # 🧱 12个DIY策略模块
│   ├── core/backtest.py           # 📊 回测引擎
│   ├── data/baostock_provider.py  # 📦 A股数据源（baostock）
│   ├── api/                       # 🔗 策略API
│   └── web/                       # 🌐 Web前端 + UI
├── docs/index.html                # GitHub Pages 演示
├── data/demo/                     # 离线演示数据集
├── scripts/                       # 工具脚本
└── pyproject.toml
```

---

## 🧱 如果你更相信技术面

不信任纯ML模型？没问题。我们的MCP Server支持任意配置技术指标（MA/RSI/MACD/KDJ等12个积木）与三层模型的决策配比。你可以设置AlphaForge占30%、你自己选的指标占70%，完全自定义。

> **进阶用法：** 适合希望将自己的技术面信号与模型信号融合的用户。通过MCP Server的 `backtest` 工具，传入自定义权重配置即可回测任意组合：
>
> ```python
> # 示例：自定义权重 — 你的技术指标占主导
> result = mcp_client.call_tool("quantpilot", "backtest", {
>     "weights": {"AlphaForge": 0.30, "TechPulse": 0.20, "Sentinel": 0.10, "custom_tech": 0.40},
>     "custom_indicators": ["MA_cross", "RSI", "MACD", "KDJ"]
> })
> ```

<details>
<summary>点击展开全部12个技术指标积木</summary>

| 模块 | 逻辑 |
|------|------|
| 📈 均线交叉 | 快线上穿慢线 → 买入 |
| 📊 RSI | RSI < 30 买入，> 70 卖出 |
| 📉 MACD | DIF上穿DEA → 买入 |
| 🎯 布林带 | 触及下轨买入，触及上轨卖出 |
| ⚡ KDJ | K线上穿D线 → 买入 |
| 🐢 海龟交易 | 突破N日高点 → 买入 |
| 📊 量价分析 | 放量上涨 → 买入 |
| 🌊 OBV | OBV趋势确认 → 买入 |
| 🔲 网格交易 | 每下跌N%买入，每上涨N%卖出 |
| 🛡️ ATR移动止损 | 基于ATR的动态止损 |
| 💰 止盈 | 达到目标收益时退出 |
| ⛔ 止损 | 达到最大亏损阈值时退出 |

</details>

---

## 🤝 参与贡献

欢迎提交PR！重点关注方向：
- 新的Alpha因子或因子模型
- 更多MCP工具端点
- 数据源集成扩展
- 前端功能改进

## 📄 开源协议

MIT
