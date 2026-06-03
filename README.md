# QuantPilot 🎯

> AI-Powered Investment Research Workbench for A-Share Market

用 LightGBM + SHAP 做 A 股板块因子选股。每个推荐都有因子解释，每行代码你都能看。

## Pre-trained Models

📦 **HuggingFace:** [SPA3K/quantpilot-models](https://huggingface.co/SPA3K/quantpilot-models)

| 板块 | 股票数 | IC ↑ | IR ↑ | vs基线 | Top因子 | 状态 |
|------|--------|------|------|--------|---------|------|
| AI应用 | 18 | **0.076** | 0.27 | +76x | ma_dist_60d, turnover_mean_5d, ret_60d | ✅ |
| 新能源 | 18 | **0.043** | 0.15 | +43x | ma_dist_10d, ma_dist_60d, rsi_14 | ✅ |
| CPO光模块 | 18 | -0.001 | -0.00 | -1x | bb_position, macd_signal, macd | ❌ |
| PCB电路板 | 17 | -0.050 | -0.15 | -50x | macd_signal, ma_dist_20d, turnover_mean_5d | ❌ |
| 具身智能 | 18 | -0.002 | -0.01 | -2x | ma_dist_10d, std_20d, macd_signal | ❌ |
| 消费白马 | 18 | -0.028 | -0.09 | -28x | ma_dist_60d, turnover_mean_20d | ❌ |

### 指标说明

| 指标 | 含义 | 方向 | 合格线 | 说明 |
|------|------|------|--------|------|
| **IC** | 截面预测能力 | ↑越大越好 | >0.03 | 模型预测排名与实际收益的相关性，基线≈0 |
| **IR** | 信息比率 | ↑越大越好 | >0.5 | IC的均值/标准差，衡量预测稳定性 |
| **vs基线** | 相对提升 | ↑越大越好 | >0 | 模型IC / 随机IC(≈0.001)的倍数 |
| **Sharpe** | 风险调整收益 | ↑越大越好 | >1.0 | 年化收益/年化波动，基线≈0 |
| **MaxDD** | 最大回撤 | ↓越小越好 | >-20% | 最大亏损幅度 |

*AI应用和新能源板块模型表现正向，可作为选股参考。其他板块IC为负，说明因子方向可能需要调整或数据量不足。*

## Quick Start

### 下载预训练模型

```bash
pip install huggingface-hub
huggingface-cli download SPA3K/quantpilot-models --local-dir ~/.quantpilot/models/prebuilt
```

### As MCP Tool (Claude Desktop)

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "quantpilot": {
      "command": "uv",
      "args": ["--directory", "/path/to/quantpilot", "run", "quantpilot-mcp"]
    }
  }
}
```

### Train Your Own

```bash
git clone https://github.com/SPA3K/quantpilot.git
cd quantpilot
uv sync
uv run quantpilot-train --sector cpo
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `sector_ranking` | 板块内股票 ML 因子排名 |
| `train_sector_model` | 训练板块专属模型 |
| `stock_analysis` | 单股深度分析 |
| `compare_sectors` | 多板块因子权重对比 |
| `list_sectors` | 查看可用板块 |
| `get_model_info` | 模型元数据 |
| `backtest_model` | 历史回测 |

## Tech Stack

- **ML**: LightGBM (LGBMRanker) + SHAP
- **Factors**: Alpha158 (简化版 45 维)
- **Data**: baostock (免费 A 股数据)
- **Protocol**: MCP (Model Context Protocol)
- **Validation**: Purged Walk-Forward (防泄漏)

## License

MIT
