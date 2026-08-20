"""
QuantPilot EVAL Engine
统一评测：对任意模型跑回测，输出标准化评测报告
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    模型评测引擎
    输入: 多个模型 + 因子面板
    输出: 每个模型的回测指标 + 模型间对比
    """

    def __init__(self, factor_panel: pd.DataFrame, top_n: int = 20,
                 forward_days: int = 20):
        self.panel = factor_panel.copy()
        self.panel["date"] = pd.to_datetime(self.panel["date"])
        self.top_n = top_n
        self.forward_days = forward_days

        # 计算未来收益标签
        self.panel["fwd_ret"] = self.panel.groupby("code")["close"].transform(
            lambda x: x.shift(-forward_days) / x - 1
        )

        # 获取月末日期
        self.panel["ym"] = self.panel["date"].dt.to_period("M")
        self.rebal_dates = (
            self.panel.groupby("ym")["date"]
            .max()
            .reset_index()
        )

    def evaluate_model(self, model, min_stocks: int = 100) -> dict:
        """
        对单个模型跑全历史回测
        """
        from quantpilot.ml.model_zoo import BaseModel

        monthly_results = []

        for _, row in self.rebal_dates.iterrows():
            ym = row["ym"]
            date = row["date"]

            # 取该截面数据
            date_data = self.panel[self.panel["date"] == date]
            if len(date_data) < min_stocks:
                continue

            # 模型预测
            try:
                preds = model.predict(self.panel, date)
            except Exception as e:
                logger.warning(f"{model.name} failed on {date}: {e}")
                continue

            if preds.empty:
                continue

            # 合并预测和实际收益
            merged = preds.merge(
                date_data[["code", "fwd_ret"]].dropna(),
                on="code", how="inner"
            )

            if len(merged) < self.top_n:
                continue

            # 选Top N
            top = merged.nlargest(self.top_n, "score")
            bottom = merged.nsmallest(self.top_n, "score")

            monthly_results.append({
                "month": str(ym),
                "date": str(date.date()),
                "top_ret": top["fwd_ret"].mean(),
                "bottom_ret": bottom["fwd_ret"].mean(),
                "long_short": top["fwd_ret"].mean() - bottom["fwd_ret"].mean(),
                "market_ret": merged["fwd_ret"].mean(),
                "excess_ret": top["fwd_ret"].mean() - merged["fwd_ret"].mean(),
                "n_stocks": len(merged),
            })

        if not monthly_results:
            return {"model": model.name, "error": "No valid predictions"}

        df = pd.DataFrame(monthly_results)

        # 计算汇总指标
        total_ls = (1 + df["long_short"]).cumprod().iloc[-1] - 1
        total_excess = (1 + df["excess_ret"]).cumprod().iloc[-1] - 1
        n_months = len(df)

        sharpe_ls = df["long_short"].mean() / (df["long_short"].std() + 1e-10) * np.sqrt(12)
        sharpe_excess = df["excess_ret"].mean() / (df["excess_ret"].std() + 1e-10) * np.sqrt(12)

        # 最大回撤
        cumret = (1 + df["excess_ret"]).cumprod()
        max_dd = ((cumret / cumret.cummax()) - 1).min()

        # 胜率
        win_rate = (df["excess_ret"] > 0).mean()
        ls_win_rate = (df["long_short"] > 0).mean()

        # 信息比率
        ir = df["excess_ret"].mean() / (df["excess_ret"].std() + 1e-10) * np.sqrt(12)

        result = {
            "model": model.name,
            "category": model.category,
            "description": model.description,
            "n_months": n_months,
            # 多空策略
            "long_short_return": float(total_ls),
            "long_short_sharpe": float(sharpe_ls),
            "long_short_win_rate": float(ls_win_rate),
            # 超额收益
            "excess_return": float(total_excess),
            "excess_sharpe": float(sharpe_excess),
            "excess_win_rate": float(win_rate),
            "information_ratio": float(ir),
            # 风险
            "max_drawdown": float(max_dd),
            "monthly_volatility": float(df["excess_ret"].std() * np.sqrt(12)),
            # 月度明细
            "monthly_details": df.to_dict(orient="records"),
        }

        return result

    def evaluate_all(self, models: list, output_dir: str = None) -> pd.DataFrame:
        """
        评测多个模型，输出对比表
        """
        results = []
        for model in models:
            logger.info(f"Evaluating {model.name}...")
            result = self.evaluate_model(model)
            results.append(result)
            logger.info(f"  Excess return: {result.get('excess_return', 'N/A'):.2%}, "
                       f"Sharpe: {result.get('excess_sharpe', 'N/A'):.2f}")

        # 汇总表
        summary = pd.DataFrame([{
            "Model": r["model"],
            "Category": r["category"],
            "Excess Return": f"{r.get('excess_return', 0)*100:.1f}%",
            "Excess Sharpe": f"{r.get('excess_sharpe', 0):.2f}",
            "Long-Short Return": f"{r.get('long_short_return', 0)*100:.1f}%",
            "LS Sharpe": f"{r.get('long_short_sharpe', 0):.2f}",
            "Max Drawdown": f"{r.get('max_drawdown', 0)*100:.1f}%",
            "Win Rate": f"{r.get('excess_win_rate', 0)*100:.0f}%",
            "IR": f"{r.get('information_ratio', 0):.2f}",
            "Months": r.get("n_months", 0),
        } for r in results])

        # 保存
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            summary.to_csv(out / "eval_summary.csv", index=False)
            with open(out / "eval_details.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Saved to {out}")

        return summary


def format_eval_report(summary: pd.DataFrame) -> str:
    """格式化评测报告"""
    lines = []
    lines.append("=" * 70)
    lines.append("QuantPilot Model Zoo — EVAL Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append(summary.to_string(index=False))
    lines.append("")
    lines.append("-" * 70)

    # 找最佳模型
    if not summary.empty:
        best_excess = summary.loc[summary["Excess Sharpe"].str.rstrip('%').astype(float).idxmax()]
        best_ls = summary.loc[summary["LS Sharpe"].str.rstrip('%').astype(float).idxmax()]
        lines.append(f"Best Excess Sharpe: {best_excess['Model']} ({best_excess['Excess Sharpe']})")
        lines.append(f"Best Long-Short:    {best_ls['Model']} ({best_ls['LS Sharpe']})")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from src.quantpilot.ml.data_fetcher import MLDataFetcher
    from quantpilot.ml.model_zoo import MODEL_REGISTRY

    fetcher = MLDataFetcher()

    # 加载因子面板
    factor_file = fetcher.data_dir / "factors_panel.parquet"
    if not factor_file.exists():
        print("Run factors.py first!")
        sys.exit(1)

    print("Loading factor panel...")
    panel = pd.read_parquet(factor_file)
    print(f"Loaded: {len(panel)} rows, {panel['code'].nunique()} stocks")

    # 评测所有已注册模型
    evaluator = ModelEvaluator(panel, top_n=20, forward_days=20)
    models = list(MODEL_REGISTRY.values())

    print(f"\nEvaluating {len(models)} models...")
    summary = evaluator.evaluate_all(models, output_dir=str(fetcher.data_dir / "eval"))

    print(f"\n{format_eval_report(summary)}")
