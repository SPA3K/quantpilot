#!/usr/bin/env python3
"""
单股监控脚本模板 — 通用版。

用法:
    python monitor.py <code>            # 监控任意标的（港股/ A股 / 美股）
    python monitor.py sh600519          # A股示例
    python monitor.py hk00700           # 港股示例

自动集成: 行情 + 技术面 + 事件日历 + 新闻舆情 + 内幕渠道 + ML信号
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import urllib.request, json, datetime, re, sys
from pathlib import Path

# ─── 配置 ───
MODEL_DIR = Path.home() / "workspace" / "quantpilot" / "models"
SINGLE_STOCK_DIR = Path.home() / "workspace" / "quantpilot" / "src" / "quantpilot" / "single_stock"
sys.path.insert(0, str(SINGLE_STOCK_DIR.parent.parent))

# ─── 行情 ───
def get_quote(code):
    req = urllib.request.Request(f"https://qt.gtimg.cn/q={code}", headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=10).read().decode("gbk")
    p = raw.split("~")
    return {"name": p[1], "price": float(p[3]), "prev_close": float(p[4]),
            "open": float(p[5]), "chg_pct": float(p[32]),
            "high": float(p[33]), "low": float(p[34]),
            "amount": float(p[37]) if p[37] else 0}

# ─── K线 + 技术指标 ───
def get_kline(code, days=320):
    if code.startswith("hk"):
        url = f"https://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get?param={code},day,,,{days},qfq"
    else:
        url = f"https://web.ifzq.gtimg.cn/appstock/app/kline/kline?param={code},day,,,{days},"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())
    node = data["data"][code]
    kl = node.get("qfqday") or node.get("day")
    return [{"date": r[0], "open": float(r[1]), "close": float(r[2]),
             "high": float(r[3]), "low": float(r[4]),
             "volume": float(r[5]) if len(r) > 5 else 0} for r in kl]

def sma(c, n): return sum(c[-n:]) / n if len(c) >= n else None

def rsi(c, n=14):
    if len(c) < n + 1: return None
    g = [max(c[i]-c[i-1], 0) for i in range(-n, 0)]
    l = [max(c[i-1]-c[i], 0) for i in range(-n, 0)]
    ag, al = sum(g)/n, sum(l)/n
    return round(100 - 100/(1+ag/al), 1) if al else 100

def compute_tech(kline):
    c = [k["close"] for k in kline]
    if len(c) < 30: return {}
    ma5, ma10, ma20, ma60 = sma(c,5), sma(c,10), sma(c,20), sma(c,60)
    if ma5 and ma10 and ma20:
        arrange = "多头排列" if ma5 > ma10 > ma20 else ("空头排列" if ma5 < ma10 < ma20 else "纠缠")
    else: arrange = "数据不足"
    yd = kline[-250:] if len(kline) >= 250 else kline
    return {"ma5": round(ma5,2) if ma5 else None, "ma10": round(ma10,2) if ma10 else None,
            "ma20": round(ma20,2) if ma20 else None, "ma60": round(ma60,2) if ma60 else None,
            "rsi14": rsi(c), "arrange": arrange,
            "chg_1d": round((c[-1]-c[-2])/c[-2]*100, 2),
            "chg_5d": round((c[-1]-c[-6])/c[-6]*100, 2) if len(c)>=6 else None,
            "chg_20d": round((c[-1]-c[-21])/c[-21]*100, 2) if len(c)>=21 else None,
            "hi52": max(k["high"] for k in yd), "lo52": min(k["low"] for k in yd)}

# ─── 舆情 ───
POS_WORDS = ["大涨","暴涨","超预期","利好","突破","新高","上调","增持","回购","扩产",
             "获准","签约","中标","合作","发布","升级","创新","增长","盈利","扭亏",
             "热销","火爆","订单","领先","获批","专利","最佳","第一","龙头","买入"]
NEG_WORDS = ["大跌","暴跌","利空","下滑","亏损","下调","减持","违规","处罚","诉讼",
             "召回","事故","裁员","关停","延期","延迟","缩水","退市","做空","爆雷",
             "低于","不及","走弱","破位","死叉","竞争加剧","份额下降","净利下降","空头"]

def text_sentiment(text):
    p = sum(1 for w in POS_WORDS if w in text)
    n = sum(1 for w in NEG_WORDS if w in text)
    return round((p-n)/(p+n), 3) if (p+n) else 0.0

def get_news(keyword):
    try:
        param = json.dumps({"uid":"","keyword":keyword,"type":["cmsArticleWebOld"],
            "client":"web","clientType":"web","clientVersion":"curr",
            "param":{"cmsArticleWebOld":{"searchScope":"default","sort":"default","pageIndex":1,"pageSize":10}}})
        url = "https://search-api-web.eastmoney.com/search/jsonp?cb=cb&param=" + urllib.parse.quote(param)
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0","Referer":"https://so.eastmoney.com/"})
        raw = urllib.request.urlopen(req, timeout=8).read().decode()
        j = json.loads(raw[raw.index("(")+1:raw.rindex(")")])
        arts = j.get("result",{}).get("cmsArticleWebOld",[])
        clean = lambda s: re.sub(r"</?em>","",s)
        return [{"title": clean(a.get("title","")), "date": a.get("date",""),
                 "content": clean(a.get("content",""))[:100]} for a in arts[:8]]
    except: return []

def get_weibo(keywords):
    try:
        req = urllib.request.Request("https://weibo.com/ajax/side/hotSearch",
            headers={"User-Agent":"Mozilla/5.0","Referer":"https://weibo.com"})
        data = json.loads(urllib.request.urlopen(req, timeout=8).read().decode())
        items = data.get("data",{}).get("realtime",[])
        hits = [{"word": i.get("word",""), "heat": i.get("num",0)}
                for i in items if any(k in i.get("word","") for k in keywords)]
        return hits
    except: return []

# ─── 事件 + 内幕 ───
def load_events(code):
    p = Path.home() / ".hermes" / "scripts" / f"{code}_events.json"
    if not p.exists(): return {"upcoming": [], "past_60d": []}
    cal = json.loads(p.read_text(encoding="utf-8"))
    today = datetime.date.today()
    up = []
    for e in cal.get("upcoming", []):
        d = (datetime.date.fromisoformat(e["date"]) - today).days
        up.append({**e, "days": d})
    past = []
    for e in cal.get("past", []):
        d = (today - datetime.date.fromisoformat(e["date"])).days
        if d <= 60: past.append({**e, "days_ago": d})
    return {"upcoming": up, "past_60d": past}

def load_insider(code):
    p = Path.home() / ".hermes" / "scripts" / f"insider_{code}.md"
    if not p.exists(): return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip().startswith("[")]

# ─── ML 信号 ───
def get_ml_signal(code, kline):
    pkl = MODEL_DIR / f"autoresearch_{code}.pkl"
    if not pkl.exists():
        return {"available": False, "note": "模型未训练"}
    try:
        import pickle, numpy as np
        sys.path.insert(0, str(Path(__file__).parent))
        from evaluate import build_rows, load_data
        model = pickle.loads(pkl.read_bytes())
        # 用与训练一致的特征计算
        rows = build_rows(kline, horizon=1)
        if not rows: return {"available": False, "note": "数据不足"}
        last = rows[-1]
        feat_names = [k for k in last if not k.startswith("_") and k != "label"]
        x = np.array([[last[f] for f in feat_names]])
        prob = model.predict_proba(x)[0]
        score = round(float(prob[1]), 3)
        direction = "看涨" if score > 0.55 else ("看跌" if score < 0.45 else "观望")
        return {"available": True, "score": score, "direction": direction}
    except Exception as e:
        return {"available": False, "note": f"推理失败: {str(e)[:80]}"}

# ─── 主流程 ───
def main():
    if len(sys.argv) < 2:
        print("用法: python monitor.py <code>（如 sh600519 / hk00700）")
        sys.exit(1)
    code = sys.argv[1]
    quote = get_quote(code)
    kline = get_kline(code)
    tech = compute_tech(kline)
    news = get_news(quote["name"])
    weibo_kw = [quote["name"].replace("-W",""), quote["name"][:2]]
    weibo = get_weibo(weibo_kw)
    events = load_events(code)
    insider = load_insider(code)
    ml = get_ml_signal(code, kline)

    out = [f"# {quote['name']} ({code}) 盘中监控"]
    out.append(f"\n## 实时行情")
    out.append(f"- 现价: {quote['price']} ({quote['chg_pct']:+.2f}%)")
    out.append(f"- 今开: {quote['open']} | 高: {quote['high']} | 低: {quote['low']}")
    out.append(f"- 成交额: {quote['amount']/1e8:.2f}亿")
    out.append(f"\n## 技术面")
    out.append(f"- MA5={tech['ma5']} MA10={tech['ma10']} MA20={tech['ma20']} MA60={tech['ma60']} → {tech['arrange']}")
    out.append(f"- RSI={tech['rsi14']} | 1日{tech['chg_1d']}% 5日{tech['chg_5d']}% 20日{tech['chg_20d']}%")
    out.append(f"- 52周: {tech['lo52']}~{tech['hi52']}")
    out.append(f"\n## 🤖 ML信号")
    if ml["available"]: out.append(f"- {ml['score']} → **{ml['direction']}**")
    else: out.append(f"- {ml['note']}")
    out.append(f"\n## 📅 事件")
    for e in events["upcoming"][:3]: out.append(f"- [{e['days']}天后] {e['date']} {e['title']}")
    for e in events["past_60d"][:3]: out.append(f"- [{e['days_ago']}天前] {e['title']} → {e.get('feeling','')}")
    out.append(f"\n## 📰 新闻")
    scores = []
    for n in news[:5]:
        s = text_sentiment(n["title"] + n.get("content",""))
        scores.append(s)
        emoji = "+" if s > 0.1 else ("-" if s < -0.1 else "=")
        out.append(f"- [{n['date'][:10]}] {n['title']} {emoji}")
    if scores:
        avg = sum(scores)/len(scores)
        out.append(f"- 舆情: {avg:+.2f}")
    out.append(f"\n## 🔥 热搜")
    if weibo:
        for h in weibo[:3]: out.append(f"- {h['word']} ({h['heat']:,})")
    else: out.append("- 无相关热搜")
    out.append(f"\n## 🔒 内幕")
    for rec in insider[-3:]: out.append(f"- {rec}")
    if not insider: out.append("- 暂无")

    print("\n".join(out))

if __name__ == "__main__":
    import urllib.parse
    main()
