# QuantPilot 🎯

> AI-Powered Investment Research Workbench for A-Share Market

用 LightGBM + SHAP 做 A 股板块因子选股。每个推荐都有因子解释，每行代码你都能看。

## Pre-trained Models

📦 **HuggingFace:** [SPA3K/quantpilot-cpo](https://huggingface.co/SPA3K/quantpilot-cpo)

| Sector | Stocks | IC | Sharpe | Download |
|--------|--------|-----|--------|----------|
| CPO光模块 | 18 | 0.029 | 1.18 | [model.lgb](https://huggingface.co/SPA3K/quantpilot-cpo/resolve/main/model.lgb) |

*更多板块训练中...*

## Quick Start

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

```bash
# Install
git clone https://github.com/SPA3K/quantpilot.git
cd quantpilot
uv sync

# 下载预训练模型
huggingface-cli download SPA3K/quantpilot-cpo --local-dir ~/.quantpilot/models/prebuilt/cpo

# 或手动下载
curl -L https://huggingface.co/SPA3K/quantpilot-cpo/resolve/main/model.lgb -o ~/.quantpilot/models/prebuilt/cpo/model.lgb
curl -L https://huggingface.co/SPA3K/quantpilot-cpo/resolve/main/metadata.json -o ~/.quantpilot/models/prebuilt/cpo/metadata.json

# 启动 MCP server
uv run quantpilot-mcp
```

### Train Your Own

```bash
# 训练指定板块
uv run quantpilot-train --sector cpo

# 训练全部预定义板块
uv run quantpilot-train --all
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `sector_ranking` | 板块内股票 ML 因子排名 |
| `train_sector_model` | 训练板块专属模型 |
| `stock_analysis` | 单股深度分析（调用板块模型） |
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

## Model Performance

```
CPO光模块板块 (18只股票, 725天数据)
├── IC:     0.029  (截面预测能力)
├── IR:     0.71   (信息比率)
├── Sharpe: 1.18   (风险调整收益)
└── Top因子: bb_position, macd_signal, macd, ma_dist_60d
```

## License

MIT
