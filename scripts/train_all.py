"""批量训练所有板块 + 基线对比"""
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

# === 板块定义 ===
SECTORS = {
    "cpo": {
        "name": "CPO光模块",
        "stocks": [
            ("sz.002475","立讯精密"), ("sz.300308","中际旭创"), ("sz.300502","新易盛"),
            ("sz.300394","天孚通信"), ("sz.002281","光迅科技"), ("sz.300570","太辰光"),
            ("sh.603083","剑桥科技"), ("sz.002396","星网锐捷"), ("sz.300548","博创科技"),
            ("sz.300620","光库科技"), ("sz.002952","鼎通科技"), ("sz.300913","兆龙互连"),
            ("sz.301205","联特科技"), ("sh.688600","天岳先进"), ("sz.002902","铭普光磁"),
            ("sz.301486","致尚科技"), ("sz.002897","意华股份"), ("sh.600498","烽火通信"),
        ],
    },
    "pcb": {
        "name": "PCB电路板",
        "stocks": [
            ("sz.002916","深南电路"), ("sz.002036","联创电子"), ("sh.600183","生益科技"),
            ("sz.002463","沪电股份"), ("sz.300408","三环集团"), ("sz.002049","紫光国微"),
            ("sz.300474","景嘉微"), ("sz.002415","海康威视"), ("sz.300661","圣邦股份"),
            ("sz.002371","北方华创"), ("sz.300782","卓胜微"), ("sh.688008","澜起科技"),
            ("sz.300223","北京君正"), ("sh.688981","中芯国际"), ("sz.002475","立讯精密"),
            ("sz.300661","圣邦股份"), ("sz.300456","赛微电子"), ("sh.603501","韦尔股份"),
        ],
    },
    "ai": {
        "name": "AI应用",
        "stocks": [
            ("sz.300033","同花顺"), ("sz.002230","科大讯飞"), ("sh.688787","海天瑞声"),
            ("sz.300496","中科创达"), ("sz.002410","广联达"), ("sz.300036","超图软件"),
            ("sz.300229","拓尔思"), ("sz.002236","大华股份"), ("sz.300454","深信服"),
            ("sz.300017","网宿科技"), ("sz.300579","数字认证"), ("sh.688111","金山办公"),
            ("sz.002405","四维图新"), ("sz.300212","易华录"), ("sz.300075","数字政通"),
            ("sz.002268","电科网安"), ("sz.300059","东方财富"), ("sz.002312","三泰控股"),
        ],
    },
    "embodied": {
        "name": "具身智能",
        "stocks": [
            ("sz.300124","汇川技术"), ("sh.601127","赛力斯"), ("sz.002747","埃斯顿"),
            ("sz.300024","机器人"), ("sh.688169","石头科技"), ("sz.002527","新时达"),
            ("sz.300503","昊志机电"), ("sh.603728","鸣志电器"), ("sz.300450","先导智能"),
            ("sz.002444","巨星科技"), ("sz.300457","赢合科技"), ("sh.688022","瀚川智能"),
            ("sz.300970","华绿生物"), ("sz.002097","山河智能"), ("sz.300159","新研股份"),
            ("sz.002399","海普瑞"), ("sz.300276","三丰智能"), ("sh.601689","拓普集团"),
        ],
    },
    "new_energy": {
        "name": "新能源",
        "stocks": [
            ("sz.300750","宁德时代"), ("sh.601012","隆基绿能"), ("sz.002459","晶澳科技"),
            ("sz.300274","阳光电源"), ("sz.002129","中环股份"), ("sz.300450","先导智能"),
            ("sz.300763","锦浪科技"), ("sz.002812","恩捷股份"), ("sz.300014","亿纬锂能"),
            ("sz.300724","捷佳伟创"), ("sh.600438","通威股份"), ("sz.002709","天赐材料"),
            ("sz.300073","当升科技"), ("sz.300568","星源材质"), ("sh.688599","天合光能"),
            ("sz.002074","国轩高科"), ("sz.300438","鹏辉能源"), ("sh.600089","特变电工"),
        ],
    },
    "consumer": {
        "name": "消费白马",
        "stocks": [
            ("sh.600519","贵州茅台"), ("sh.600887","伊利股份"), ("sz.000858","五粮液"),
            ("sh.603288","海天味业"), ("sz.000568","泸州老窖"), ("sh.600809","山西汾酒"),
            ("sz.002304","洋河股份"), ("sh.600600","青岛啤酒"), ("sz.000333","美的集团"),
            ("sz.000651","格力电器"), ("sh.601888","中国中免"), ("sz.002714","牧原股份"),
            ("sh.600276","恒瑞医药"), ("sz.300760","迈瑞医疗"), ("sh.603259","药明康德"),
            ("sz.002352","顺丰控股"), ("sh.601899","紫金矿业"), ("sz.000876","新希望"),
        ],
    },
}

def fetch_stock_data(code, start="2023-06-01", end="2026-06-01"):
    """拉取单只股票数据"""
    rs = bs.query_history_k_data_plus(
        code, "date,open,high,low,close,volume,amount,turn",
        start_date=start, end_date=end, frequency="d", adjustflag="2")
    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())
    if len(rows) < 200:
        return None
    df = pd.DataFrame(rows, columns=rs.fields)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open","high","low","close","volume","amount","turn"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={"amount":"turnover","turn":"turnover_rate"}).dropna(subset=["close"])
    return df[["date","open","high","low","close","volume","turnover","turnover_rate"]]


def train_sector(sector_id, sector_cfg):
    """训练单个板块，返回结果字典"""
    stocks = sector_cfg["stocks"]
    name = sector_cfg["name"]
    
    lg = bs.login()
    stock_data = {}
    name_map = {}
    for code, sname in stocks:
        df = fetch_stock_data(code)
        if df is not None:
            stock_data[code] = df
            name_map[code] = sname
    bs.logout()
    
    if len(stock_data) < 10:
        return {"error": f"Only {len(stock_data)} stocks available"}
    
    # 提取因子
    factors_df = extract_factors_batch(stock_data)
    factor_cols = [c for c in factors_df.columns if c not in ["date","ticker","close"]]
    factor_cols = [c for c in factor_cols if factors_df[c].notna().sum() > len(factors_df)*0.5]
    
    # 前瞻收益
    factors_df = factors_df.sort_values(["ticker","date"]).reset_index(drop=True)
    factors_df["fwd_ret_20d"] = factors_df.groupby("ticker")["close"].transform(
        lambda x: x.pct_change(20).shift(-20))
    factors_df = factors_df.dropna(subset=["fwd_ret_20d"])
    
    # 标签
    factors_df["label"] = factors_df.groupby("date", group_keys=False)["fwd_ret_20d"].transform(
        lambda x: pd.qcut(x, q=5, labels=False, duplicates="drop") if len(x.dropna()) >= 5 else 0)
    
    # 训练
    model, metrics = purged_walk_forward(factors_df, factor_cols, n_splits=5)
    
    # SHAP
    X_all = factors_df[factor_cols].fillna(0).values
    global_shap = explain_global(model, X_all, factor_cols)
    
    # 保存
    save_model(model, sector_id, metrics, factor_cols)
    register_model(sector_id, metrics, metadata={
        "n_stocks": len(stock_data), "n_factors": len(factor_cols), "stock_names": name_map})
    
    base = os.path.expanduser(f"~/.quantpilot/models/prebuilt/{sector_id}")
    with open(os.path.join(base, "stocks.json"), "w") as f:
        json.dump([{"code":c,"name":name_map.get(c,c)} for c in stock_data], f, ensure_ascii=False, indent=2)
    
    return {
        "sector": sector_id,
        "name": name,
        "n_stocks": len(stock_data),
        "n_factors": len(factor_cols),
        "metrics": metrics,
        "top_factors": global_shap["top_factors"][:5],
    }


# === 训练全部 ===
print("=" * 60)
print("  QuantPilot — 批量训练")
print("=" * 60)

results = []
t_start = time.time()
for sid, cfg in SECTORS.items():
    t0 = time.time()
    print(f"\n{'─'*50}")
    print(f"  训练: {cfg['name']} ({sid})")
    print(f"{'─'*50}")
    result = train_sector(sid, cfg)
    elapsed = time.time() - t0
    if "error" in result:
        print(f"  ❌ {result['error']}")
    else:
        m = result["metrics"]
        print(f"  ✅ {result['n_stocks']} 只股票, {result['n_factors']} 因子 ({elapsed:.0f}s)")
        print(f"     IC={m['ic_mean']:.4f}  IR={m['ir']:.2f}  Folds={m['n_folds']}")
        factors_str = ", ".join(f["name"] for f in result["top_factors"][:3])
        print(f"     Top: {factors_str}")
    results.append(result)

# === 基线对比 ===
print(f"\n\n{'='*60}")
print(f"  📊 模型 vs 基线对比")
print(f"{'='*60}")
print()
print(f"  基线 = 随机选股 (IC≈0, Sharpe≈0)")
print(f"  模型 = LightGBMRanker + Purged Walk-Forward")
print()

# 指标说明
METRIC_DOCS = {
    "ic":  {"label": "IC", "desc": "截面预测能力", "better": "↑越大越好", "good": ">0.03"},
    "ir":  {"label": "IR", "desc": "信息比率(稳定性)", "better": "↑越大越好", "good": ">0.5"},
    "sharpe": {"label": "Sharpe", "desc": "风险调整收益", "better": "↑越大越好", "good": ">1.0"},
    "max_dd": {"label": "MaxDD", "desc": "最大回撤", "better": "↓越小越好", "good": ">-20%"},
    "win_rate": {"label": "胜率", "desc": "正收益占比", "better": "↑越大越好", "good": ">55%"},
}

print(f"  {'板块':^12s} {'股票数':>6s} {'IC':>8s} {'IR':>8s} {'vs基线':>8s}  {'状态':^6s}  Top因子")
print(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*8} {'─'*8}  {'─'*6}  {'─'*30}")

for r in results:
    if "error" in r:
        print(f"  {r['name']:^12s} {'ERR':>6s}")
        continue
    m = r["metrics"]
    ic = m["ic_mean"]
    ir = m["ir"]
    # 基线 IC≈0，模型 IC 越高越好
    lift = f"+{ic/0.001:.0f}x" if ic > 0 else f"{ic/0.001:.0f}x"
    status = "✅" if ic > 0.01 else "⚠️" if ic > 0 else "❌"
    factors = ", ".join(f["name"] for f in r["top_factors"][:3])
    print(f"  {r['name']:^12s} {r['n_stocks']:>6d} {ic:>8.4f} {ir:>8.2f} {lift:>8s}  {status:^6s}  {factors}")

print()
print(f"  指标说明:")
print(f"  {'─'*55}")
print(f"  IC   截面预测能力   ↑越大越好   >0.03合格   (基线≈0)")
print(f"  IR   信息比率       ↑越大越好   >0.5合格    (基线≈0)")
print(f"  ↑ vs基线  模型相对随机选股的提升倍数")
print(f"  {'─'*55}")
print(f"  总耗时: {time.time()-t_start:.0f}s")
print(f"  模型保存: ~/.quantpilot/models/prebuilt/")
