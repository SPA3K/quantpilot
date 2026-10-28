#!/usr/bin/env python3
"""行业监控模式 showcase — 标准指标重算（用价格序列，非累计收益序列）。"""
import json, urllib.request, math, sys

ETF_REP = {
    "银行ETF":    "sh512800",
    "半导体":     "sh512480",
    "CPO/光通信": "sh515880",
    "具身/机器人": "sh562500",
    "科技宽基":   "sh515030",
}

def get_kline(code, days=130):
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/kline/kline"
           f"?param={code},day,,,{days},")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
    n = data["data"][code]
    kl = n.get("day") or n.get("qfqday")
    return {r[0]: float(r[2]) for r in kl}

def max_drawdown(prices):
    """标准最大回撤：基于价格序列。"""
    peak, mdd = prices[0], 0.0
    for p in prices:
        peak = max(peak, p)
        mdd = min(mdd, (p - peak) / peak)
    return mdd * 100

def annual_vol(prices):
    """年化波动率：日收益率标准差 × sqrt(244)。"""
    rets = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
    mu = sum(rets)/len(rets)
    var = sum((r-mu)**2 for r in rets)/len(rets)
    return math.sqrt(var) * math.sqrt(244) * 100

def main():
    data = {n: get_kline(c) for n, c in ETF_REP.items()}
    common = set.intersection(*[set(v.keys()) for v in data.values()])
    dates = sorted(common)
    print(f"共同区间: {dates[0]} → {dates[-1]}  共 {len(dates)} 交易日\n")

    print("=== 期末累计收益 / 最大回撤 / 年化波动率 / 盈亏比 ===")
    print(f"{'板块':<14}{'累计':>9}{'最大回撤':>11}{'年化波动':>10}")
    for n in ETF_REP:
        px = [data[n][d] for d in dates]
        ret = (px[-1]/px[0] - 1)*100
        mdd = max_drawdown(px)
        vol = annual_vol(px)
        print(f"{n:<14}{ret:>8.2f}%{mdd:>10.2f}%{vol:>9.2f}%")

    # 关键：防御价值 = 相对科技宽基的超额
    print("\n=== 相对科技宽基的超额收益（防御价值）===")
    tech = [data["科技宽基"][d] for d in dates]
    tech_ret = tech[-1]/tech[0] - 1
    for n in ETF_REP:
        if n == "科技宽基": continue
        px = [data[n][d] for d in dates]
        r = px[-1]/px[0] - 1
        print(f"{n:<14} {(r-tech_ret)*100:>+8.2f}pp")

    # 分化起点：银行累计收益首次转正 vs 科技见顶回落
    print("\n=== 关键时点追踪 ===")
    bk = [data["银行ETF"][d] for d in dates]
    # 科技宽基峰值日
    peak_i = tech.index(max(tech))
    print(f"科技宽基峰值日: {dates[peak_i]}  (价格 {max(tech):.3f})")
    print(f"  该日银行累计 {(bk[peak_i]/bk[0]-1)*100:+.2f}%")
    # 峰值日 → 期末，各板块在这段暴跌中的表现
    print(f"\n从科技峰值日({dates[peak_i]})到期末的区间表现:")
    for n in ETF_REP:
        px = [data[n][d] for d in dates[peak_i:]]
        r = (px[-1]/px[0] - 1)*100
        print(f"  {n:<14} {r:>+8.2f}%")

    # 银行首次跑赢科技宽基（累计口径）
    bk_cum = [bk[i]/bk[0]-1 for i in range(len(dates))]
    tech_cum = [tech[i]/tech[0]-1 for i in range(len(dates))]
    cross_i = next((i for i in range(len(dates)) if bk_cum[i] > tech_cum[i]), None)
    if cross_i is not None:
        print(f"\n银行累计首次跑赢科技宽基: {dates[cross_i]}")
        print(f"  银行 {bk_cum[cross_i]*100:+.2f}% vs 科技宽基 {tech_cum[cross_i]*100:+.2f}%")
        lead = sum(1 for i in range(len(dates)) if bk_cum[i] > tech_cum[i])
        print(f"  此后跑赢天数占比: {lead}/{len(dates)} = {lead/len(dates)*100:.0f}%")

    json.dump({n: {d: data[n][d] for d in dates} for n in ETF_REP},
              open(r"C:\Users\shiel\workspace\quantpilot\industry_showcase.json", "w"),
              ensure_ascii=False, indent=1)
    print("\n[saved industry_showcase.json]")

if __name__ == "__main__":
    main()
