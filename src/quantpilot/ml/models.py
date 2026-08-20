"""
QuantPilot ML Model
- Purged K-Fold CV (AFML Ch7)
- LightGBM 分类模型（预测未来N日收益排名Top20%）
- 滚动训练 + 月度调仓回测
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ── Purged K-Fold CV (from AFML) ──────────────────────────────

class PurgedKFoldCV:
    """
    金融时间序列的交叉验证
    - Purge: 训练集和测试集之间留gap，避免look-ahead bias
    - Embargo: 测试集之后也留gap，避免标签泄露
    """

    def __init__(self, n_splits: int = 5, pct_embargo: float = 0.01):
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo

    def split(self, X: pd.DataFrame, y: pd.Series = None,
              groups: pd.Series = None, timestamps: pd.Series = None):
        """
        生成Purged K-Fold索引
        X: 特征矩阵 (index=时间顺序)
        timestamps: 时间戳列 (如果有)
        """
        if timestamps is not None:
            times = pd.Series(range(len(X)), index=timestamps.sort_values())
        else:
            times = pd.Series(range(len(X)))

        n = len(X)
        fold_size = n // self.n_splits
        embargo_size = int(n * self.pct_embargo)

        indices = np.arange(n)

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)

            # Purge: 训练集排除测试集附近的样本
            purge_start = max(0, test_start - embargo_size)
            purge_end = min(n, test_end + embargo_size)

            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([indices[:purge_start], indices[purge_end:]])

            yield train_idx, test_idx


# ── 标签构建 ──────────────────────────────────────────────

def build_labels(df: pd.DataFrame, forward_days: int = 20,
                 quantile: float = 0.2) -> pd.DataFrame:
    """
    构建分类标签：未来N日收益率排名前20% → 1, 否则 → 0
    按日期截面分组排名
    """
    df = df.copy()

    # 未来N日收益率（按股票分组）
    df["fwd_ret"] = df.groupby("code")["close"].transform(
        lambda x: x.shift(-forward_days) / x - 1
    )

    # 按日期截面排名
    df["label"] = df.groupby("date")["fwd_ret"].transform(
        lambda x: (x >= x.quantile(1 - quantile)).astype(int)
    )

    return df


# ── LightGBM 训练 ──────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """获取因子列（排除非特征列）"""
    exclude = {"date", "code", "open", "high", "low", "close", "volume",
               "turnover", "turnover_rate", "peTTM", "pbMRQ", "psTTM",
               "pcfNcfTTM", "isST", "fwd_ret", "label", "obv",
               "pubDate", "statDate", "year", "quarter"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64]]


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   params: dict = None):
    """训练LightGBM分类模型"""
    import lightgbm as lgb

    if params is None:
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [lgb.log_evaluation(period=100), lgb.early_stopping(50)]

    model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    return model


# ── 滚动训练框架 ──────────────────────────────────────────

class RollingTrainer:
    """
    滚动窗口训练：
    - 每月重新训练模型
    - 训练窗口: 过去2年数据
    - 预测窗口: 下个月
    - 样本权重: 时间衰减
    """

    def __init__(self, train_window_days: int = 504,  # ~2年
                 forward_days: int = 20,  # 预测20日收益
                 top_quantile: float = 0.2,  # Top 20%
                 rebalance_freq: str = "M"):  # 月度调仓
        self.train_window = train_window_days
        self.forward_days = forward_days
        self.top_quantile = top_quantile
        self.rebalance_freq = rebalance_freq

    def run(self, factor_panel: pd.DataFrame,
            feature_cols: List[str] = None) -> pd.DataFrame:
        """
        滚动训练 + 预测
        返回: 每月每只股票的预测概率
        """
        df = factor_panel.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if feature_cols is None:
            feature_cols = get_feature_columns(df)

        # 构建标签
        df = build_labels(df, self.forward_days, self.top_quantile)
        df = df.dropna(subset=["label", "fwd_ret"])

        # 获取月末日期（调仓日）
        df["ym"] = df["date"].dt.to_period("M")
        rebal_dates = df.groupby("ym")["date"].max().values

        all_predictions = []

        for i, rebal_date in enumerate(rebal_dates):
            # 训练窗口
            train_end = pd.Timestamp(rebal_date) - pd.Timedelta(days=self.forward_days)
            train_start = train_end - pd.Timedelta(days=self.train_window)

            train_mask = (df["date"] >= train_start) & (df["date"] <= train_end)
            pred_mask = df["ym"] == df[df["date"] == rebal_date]["ym"].iloc[0]

            if train_mask.sum() < 1000:
                logger.warning(f"Skipping {rebal_date}: insufficient training data ({train_mask.sum()})")
                continue

            X_train = df.loc[train_mask, feature_cols]
            y_train = df.loc[train_mask, "label"]

            X_pred = df.loc[pred_mask, feature_cols]
            y_pred = df.loc[pred_mask, "label"]

            # 处理NaN
            X_train = X_train.fillna(0)
            X_pred = X_pred.fillna(0)

            # 训练
            try:
                model = train_lightgbm(X_train, y_train, X_pred, y_pred)
                pred_proba = model.predict(X_pred)

                pred_df = df.loc[pred_mask, ["date", "code", "close", "fwd_ret", "label"]].copy()
                pred_df["pred_proba"] = pred_proba
                pred_df["rebal_date"] = rebal_date
                all_predictions.append(pred_df)

                # 打印模型性能
                from sklearn.metrics import roc_auc_score, accuracy_score
                auc = roc_auc_score(y_pred, pred_proba)
                acc = accuracy_score(y_pred, (pred_proba > 0.5).astype(int))
                logger.info(f"Rebal {rebal_date}: AUC={auc:.4f}, Acc={acc:.4f}, "
                           f"Train={len(X_train)}, Pred={len(X_pred)}")

            except Exception as e:
                logger.error(f"Training failed for {rebal_date}: {e}")
                continue

        if not all_predictions:
            return pd.DataFrame()

        return pd.concat(all_predictions, ignore_index=True)


# ── 回测引擎 ──────────────────────────────────────────────

class SimpleBacktester:
    """
    简单月度调仓回测
    - 每月初根据模型预测选Top N只股票
    - 等权重持仓
    - 计算收益/夏普/最大回撤
    """

    def __init__(self, top_n: int = 20, initial_capital: float = 1000000):
        self.top_n = top_n
        self.initial_capital = initial_capital

    def run(self, predictions: pd.DataFrame) -> dict:
        """
        运行回测
        predictions: RollingTrainer的输出 (含date, code, pred_proba, fwd_ret)
        """
        if predictions.empty:
            return {"error": "No predictions"}

        # 按月选股
        predictions["ym"] = pd.to_datetime(predictions["rebal_date"]).dt.to_period("M")
        monthly_returns = []

        for ym, group in predictions.groupby("ym"):
            # 选预测概率最高的Top N
            top_stocks = group.nlargest(self.top_n, "pred_proba")
            # 等权重平均收益
            avg_ret = top_stocks["fwd_ret"].mean()
            # 基准：等权全市场
            bench_ret = group["fwd_ret"].mean()
            monthly_returns.append({
                "month": str(ym),
                "strategy_return": avg_ret,
                "benchmark_return": bench_ret,
                "excess_return": avg_ret - bench_ret,
                "top_n": len(top_stocks),
            })

        if not monthly_returns:
            return {"error": "No monthly returns"}

        ret_df = pd.DataFrame(monthly_returns)

        # 计算累计收益
        ret_df["strategy_cumret"] = (1 + ret_df["strategy_return"]).cumprod()
        ret_df["benchmark_cumret"] = (1 + ret_df["benchmark_return"]).cumprod()

        # 计算指标
        total_ret = ret_df["strategy_cumret"].iloc[-1] - 1
        bench_total = ret_df["benchmark_cumret"].iloc[-1] - 1
        excess_total = total_ret - bench_total

        # 年化收益
        n_months = len(ret_df)
        ann_ret = (1 + total_ret) ** (12 / n_months) - 1 if n_months > 0 else 0

        # 夏普比
        monthly_excess = ret_df["strategy_return"] - ret_df["benchmark_return"]
        sharpe = monthly_excess.mean() / (monthly_excess.std() + 1e-10) * np.sqrt(12)

        # 最大回撤
        cummax = ret_df["strategy_cumret"].cummax()
        drawdown = (ret_df["strategy_cumret"] - cummax) / cummax
        max_drawdown = drawdown.min()

        # 胜率
        win_rate = (monthly_excess > 0).mean()

        return {
            "total_return": float(total_ret),
            "benchmark_return": float(bench_total),
            "excess_return": float(excess_total),
            "annualized_return": float(ann_ret),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "n_months": n_months,
            "monthly_details": ret_df.to_dict(orient="records"),
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from src.quantpilot.ml.data_fetcher import MLDataFetcher

    fetcher = MLDataFetcher()

    # 加载因子面板
    factor_file = fetcher.data_dir / "factors_panel.parquet"
    if not factor_file.exists():
        print("Run factors.py first to compute factors!")
        sys.exit(1)

    print("Loading factor panel...")
    factors = pd.read_parquet(factor_file)
    print(f"Loaded {len(factors)} rows, {factors['code'].nunique()} stocks")

    # 滚动训练
    print("\nRolling training...")
    trainer = RollingTrainer(train_window_days=504, forward_days=20, top_quantile=0.2)
    predictions = trainer.run(factors)
    print(f"Predictions: {len(predictions)} rows")

    if not predictions.empty:
        # 回测
        print("\nBacktesting...")
        backtester = SimpleBacktester(top_n=20)
        results = backtester.run(predictions)

        print(f"\n{'='*50}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*50}")
        print(f"Total Return:     {results['total_return']*100:.2f}%")
        print(f"Benchmark Return: {results['benchmark_return']*100:.2f}%")
        print(f"Excess Return:    {results['excess_return']*100:.2f}%")
        print(f"Annualized:       {results['annualized_return']*100:.2f}%")
        print(f"Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:     {results['max_drawdown']*100:.2f}%")
        print(f"Win Rate:         {results['win_rate']*100:.1f}%")
        print(f"Months:           {results['n_months']}")

        # 保存结果
        import json
        output = fetcher.data_dir / "backtest_results.json"
        with open(output, "w") as f:
            json.dump({k: v for k, v in results.items() if k != "monthly_details"}, f, indent=2)
        print(f"\nResults saved to {output}")
