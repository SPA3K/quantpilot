#!/usr/bin/env python3
"""
固定评估协议 — 不可修改。
提供: load_data() + evaluate() 供 train.py 调用。
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import urllib.request, json, pickle
from pathlib import Path
import numpy as np

DATA_CACHE = Path(__file__).parent / "data_cache.pkl"

def _get_kline(code, days=800):
    url = f"https://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get?param={code},day,,,{days},qfq"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
    node = data["data"][code]
    kl = node.get("qfqday") or node.get("day")
    return [{"date": r[0], "open": float(r[1]), "close": float(r[2]),
             "high": float(r[3]), "low": float(r[4]),
             "volume": float(r[5]) if len(r) > 5 else 0} for r in kl]

def _rsi(closes, n=14):
    if len(closes) < n + 1: return 50.0
    gains = [max(closes[i]-closes[i-1], 0) for i in range(-n, 0)]
    losses = [max(closes[i-1]-closes[i], 0) for i in range(-n, 0)]
    ag, al = sum(gains)/n, sum(losses)/n
    if al == 0: return 100.0
    return 100 - 100/(1+ag/al)

def _macd_hist(closes):
    if len(closes) < 35: return 0
    kf, ks, kd = 2/13, 2/27, 2/10
    ef, es, ed = closes[0], closes[0], 0; difs = []
    for c in closes:
        ef = c*kf+ef*(1-kf); es = c*ks+es*(1-ks); d = ef-es; difs.append(d)
        ed = d*kd+ed*(1-kd)
    return (difs[-1]-ed)*2

def build_rows(kline, horizon=1):
    closes = [k["close"] for k in kline]
    vols = [k["volume"] for k in kline]
    rows = []
    for i in range(60, len(closes) - horizon):
        c = closes[:i+1]; v = vols[:i+1]
        ma5 = sum(c[-5:])/5; ma10 = sum(c[-10:])/10
        ma20 = sum(c[-20:])/20; ma60 = sum(c[-60:])/60
        r14 = _rsi(c, 14)
        vma20 = sum(v[-20:])/20 if len(v)>=20 and sum(v[-20:]) else 1
        vr = v[-1]/vma20 if vma20 else 1
        r1 = (c[-1]-c[-2])/c[-2] if c[-2] else 0
        r5 = (c[-1]-c[-6])/c[-6] if len(c)>=6 else 0
        r10 = (c[-1]-c[-11])/c[-11] if len(c)>=11 else 0
        r20 = (c[-1]-c[-21])/c[-21] if len(c)>=21 else 0
        w = c[-21:]; rs = [(w[j]-w[j-1])/w[j-1] for j in range(1,len(w))]
        mr = sum(rs)/len(rs) if rs else 0
        v20 = (sum((r-mr)**2 for r in rs)/len(rs))**0.5 if rs else 0
        mh = _macd_hist(c)
        puvd = 1 if (r5>0 and vr<0.8) else 0
        pdvu = 1 if (r5<0 and vr>1.2) else 0
        fwd = (closes[i+horizon]-closes[i])/closes[i]
        rows.append({
            "ma5_ma20_ratio": ma5/ma20 if ma20 else 1,
            "ma10_ma60_ratio": ma10/ma60 if ma60 else 1,
            "price_ma20_bias": (c[-1]-ma20)/ma20 if ma20 else 0,
            "price_ma60_bias": (c[-1]-ma60)/ma60 if ma60 else 0,
            "rsi14": r14, "macd_hist": mh, "vol_ratio": vr,
            "ret_1d": r1, "ret_5d": r5, "ret_10d": r10, "ret_20d": r20,
            "volatility_20d": v20,
            "price_up_vol_down": puvd, "price_down_vol_up": pdvu,
            "label": 1 if fwd>0 else 0,
            "_fwd_ret": fwd,
        })
    return rows

def load_data(code=None, horizon=1):
    """加载数据并切分。code 为腾讯格式（港股 hk###### / 沪市 sh###### / 深市 sz######）。"""
    if code is None:
        code = os.environ.get("AUTORESEARCH_CODE")
        if not code:
            raise ValueError(
                "请指定标的：设置环境变量 AUTORESEARCH_CODE（如 sh600519），"
                "或调用 load_data(code=...)。"
            )
    cache_key = f"{code}_{horizon}"
    if DATA_CACHE.exists():
        meta = pickle.loads(DATA_CACHE.read_bytes())
        if meta.get("cache_key") == cache_key:
            return (meta["X_train"], meta["y_train"], meta["X_val"], meta["y_val"],
                    meta["X_test"], meta["y_test"], meta["features"], meta)
    print(f"拉取 {code} 数据(首次运行)...")
    kline = _get_kline(code, 800)
    rows = build_rows(kline, horizon)
    features = [k for k in rows[0] if not k.startswith("_") and k != "label"]
    X = np.array([[r[f] for f in features] for r in rows])
    y = np.array([r["label"] for r in rows])
    fwd = np.array([r["_fwd_ret"] for r in rows])
    n = len(X); t1, t2 = int(n*0.70), int(n*0.85)
    meta = {
        "X_train": X[:t1], "y_train": y[:t1],
        "X_val": X[t1:t2], "y_val": y[t1:t2],
        "X_test": X[t2:], "y_test": y[t2:], "fwd_test": fwd[t2:],
        "features": features, "horizon": horizon, "cache_key": cache_key,
    }
    DATA_CACHE.write_bytes(pickle.dumps(meta))
    print(f"数据: train={t1} val={t2-t1} test={n-t2}")
    return (meta["X_train"], meta["y_train"], meta["X_val"], meta["y_val"],
            meta["X_test"], meta["y_test"], features, meta)

def evaluate(model, meta):
    """返回指标 dict。model 需有 predict_proba 方法。"""
    X_test, y_test, fwd = meta["X_test"], meta["y_test"], meta["fwd_test"]
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob > 0.5).astype(int)
    acc = (pred == y_test).mean()
    conf = (prob > 0.55) | (prob < 0.45)
    n_conf = int(conf.sum())
    conf_acc = (pred[conf] == y_test[conf]).mean() if n_conf > 0 else 0
    conf_fwd = fwd[conf] if n_conf > 0 else np.array([0])
    avg_fwd = conf_fwd.mean() if len(conf_fwd) else 0
    return {
        "acc": round(float(acc), 4),
        "conf_n": n_conf,
        "conf_acc": round(float(conf_acc), 4),
        "conf_avg_fwd_ret": round(float(avg_fwd), 6),
    }
