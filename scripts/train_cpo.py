"""CPO板块训练脚本 — 使用baostock真实A股数据"""
import json, os, sys, warnings, time
import numpy as np, pandas as pd
import baostock as bs
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quantpilot.engine.factors import extract_factors_batch
from quantpilot.engine.trainer import purged_walk_forward, save_model
from quantpilot.engine.explainer import explain_global
from quantpilot.engine.registry import register_model
from quantpilot.config import ensure_dirs

ensure_dirs()

CPO = [
    ("sz.002475","立讯精密"), ("sz.300308","中际旭创"), ("sz.300502","新易盛"),
    ("sz.300394","天孚通信"), ("sz.002281","光迅科技"), ("sz.300570","太辰光"),
    ("sh.603083","剑桥科技"), ("sz.002396","星网锐捷"), ("sz.300548","博创科技"),
    ("sz.300620","光库科技"), ("sz.002952","鼎通科技"), ("sz.300913","兆龙互连"),
    ("sz.301205","联特科技"), ("sh.688600","天岳先进"), ("sz.002902","铭普光磁"),
    ("sz.301486","致尚科技"), ("sz.002897","意华股份"), ("sh.600498","烽火通信"),
]

print("=" * 55)
print("  QuantPilot — CPO光模块 训练")
print("=" * 55)

# 1. 拉数据
t0 = time.time()
lg = bs.login()
stock_data = {}
name_map = {}
for code, name in CPO:
    rs = bs.query_history_k_data_plus(
        code, "date,open,high,low,close,volume,amount,turn",
        start_date="2023-06-01", end_date="2026-06-01",
        frequency="d", adjustflag="2")
    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())
    if len(rows) < 200:
        continue
    df = pd.DataFrame(rows, columns=rs.fields)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open","high","low","close","volume","amount","turn"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={"amount":"turnover","turn":"turnover_rate"}).dropna(subset=["close"])
    stock_data[code] = df[["date","open","high","low","close","volume","turnover","turnover_rate"]]
    name_map[code] = name
bs.logout()
print(f"  数据: {len(stock_data)} 只股票 ({time.time()-t0:.0f}s)")

# 2. 提取因子 (close 现在会自动保留)
t1 = time.time()
factors_df = extract_factors_batch(stock_data)
factor_cols = [c for c in factors_df.columns if c not in ["date","ticker","close"]]
factor_cols = [c for c in factor_cols if factors_df[c].notna().sum() > len(factors_df)*0.5]
print(f"  因子: {len(factors_df)} 行 × {len(factor_cols)} 维 ({time.time()-t1:.0f}s)")

# 3. 前瞻收益 (向量化 — 用 close 列)
t2 = time.time()
factors_df = factors_df.sort_values(["ticker","date"]).reset_index(drop=True)
factors_df["fwd_ret_20d"] = factors_df.groupby("ticker")["close"].transform(
    lambda x: x.pct_change(20).shift(-20)
)
factors_df = factors_df.dropna(subset=["fwd_ret_20d"])
print(f"  前瞻收益: {len(factors_df)} 样本 ({time.time()-t2:.0f}s)")

# 4. 标签
factors_df["label"] = factors_df.groupby("date", group_keys=False)["fwd_ret_20d"].transform(
    lambda x: pd.qcut(x, q=5, labels=False, duplicates="drop") if len(x.dropna()) >= 5 else 0
)

# 5. 训练
t3 = time.time()
print("  训练 LightGBMRanker...")
model, metrics = purged_walk_forward(factors_df, factor_cols, n_splits=5)
print(f"  完成 ({time.time()-t3:.0f}s)")

# 6. SHAP
X_all = factors_df[factor_cols].fillna(0).values
global_shap = explain_global(model, X_all, factor_cols)

# 7. 输出
print(f"\n{'='*55}")
print(f"  📊 CPO光模块 模型指标")
print(f"{'='*55}")
print(f"  IC:    {metrics['ic_mean']:.4f}")
print(f"  IR:    {metrics['ir']:.2f}")
print(f"  Folds: {metrics['n_folds']}")
print()
print("  Top 10 因子:")
for f in global_shap["top_factors"][:10]:
    print(f"  {f['rank']:2d}. {f['name']:25s} {f['importance']:.4f}")
print("=" * 55)

# 8. 保存
save_model(model, "cpo", metrics, factor_cols)
register_model("cpo", metrics, metadata={
    "n_stocks": len(stock_data),
    "n_factors": len(factor_cols),
    "stock_names": name_map,
})
base = os.path.expanduser("~/.quantpilot/models/prebuilt/cpo")
with open(os.path.join(base, "stocks.json"), "w") as f:
    json.dump([{"code":c,"name":name_map.get(c,c)} for c in stock_data], f, ensure_ascii=False, indent=2)
print(f"\n✅ 保存到 {base}/")
print(f"   model.lgb | metadata.json | stocks.json")
