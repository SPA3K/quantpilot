#!/usr/bin/env python3
"""
QuantPilot Signal Generator v2 — L1 ML + 舆情原文
对每个ETF板块的龙头股跑L1 LightGBM预测 + 抓取东方财富最新新闻标题。

输出格式供Agent链路消费：
  L1分数 + 原始新闻文本 → Agent做情绪分析+辩论+决策

用法:
  python3 ml_etf_signal_v2.py           # 输出JSON到stdout
  python3 ml_etf_signal_v2.py --save    # 同时保存到cache
"""

import sys, os, json, pickle, datetime, argparse, re, time
import urllib.request, urllib.parse
import os
from pathlib import Path
STATE_DIR = Path(os.environ.get("QUANTPILOT_DATA_DIR")
             or (Path.home() / ".quantpilot" / "industry"))

import numpy as np
import pandas as pd

# ═══════ 配置 ═══════

MODEL_PATH = Path.home() / "workspace" / "quantpilot" / "models" / "trained" / "lgb_cs_2008_2022.pkl"
DATA_DIR = Path.home() / "workspace" / "quantpilot" / "data" / "ml" / "daily"
CACHE_PATH = STATE_DIR / "ml_signal_cache.json"

ETF_STOCKS = {
    "芯片/半导体": {
        "etfs": ["512480", "512760", "159995", "588200"],
        "stocks": [
            ("688256", "寒武纪", 0.25),
            ("688981", "中芯国际", 0.20),
            ("002371", "北方华创", 0.20),
            ("603501", "韦尔股份", 0.15),
            ("688008", "澜起科技", 0.10),
            ("688396", "华峰测控", 0.10),
        ],
    },
    "CPO/光通信": {
        "etfs": ["515880", "515050", "512880"],
        "stocks": [
            ("300502", "新易盛", 0.25),
            ("300308", "中际旭创", 0.25),
            ("600522", "中天科技", 0.20),
            ("600487", "亨通光电", 0.20),
            ("002281", "光迅科技", 0.10),
        ],
    },
    "具身/机器人": {
        "etfs": ["562500", "562300", "159770"],
        "stocks": [
            ("300124", "汇川技术", 0.25),
            ("688169", "石头科技", 0.20),
            ("300607", "拓斯达", 0.20),
            ("002747", "埃斯顿", 0.15),
            ("688022", "瀚川智能", 0.10),
            ("300024", "机器人", 0.10),
        ],
    },
    "科技宽基": {
        "etfs": ["515030", "588000", "515750", "512720"],
        "stocks": [
            ("300750", "宁德时代", 0.20),
            ("688256", "寒武纪", 0.15),
            ("002415", "海康威视", 0.15),
            ("300059", "东方财富", 0.15),
            ("688981", "中芯国际", 0.10),
            ("002230", "科大讯飞", 0.10),
            ("603259", "药明康德", 0.05),
        ],
    },
    "银行ETF": {
        "etfs": ["512800", "515020", "512820"],
        "stocks": [
            ("601398", "工商银行", 0.25),
            ("601288", "农业银行", 0.20),
            ("601939", "建设银行", 0.20),
            ("600036", "招商银行", 0.20),
            ("601166", "兴业银行", 0.15),
        ],
    },
}

FEATURE_COLS = [
    'ret_1', 'ret_5', 'ret_10', 'ret_20',
    'mom_5', 'mom_10', 'mom_20', 'mom_60',
    'vol_5', 'vol_20', 'vol_60',
    'rsi_14', 'ma5_dev', 'ma10_dev', 'ma20_dev', 'ma60_dev',
    'boll_pct', 'vol_ratio_5_20', 'vol_ratio_5_60',
    'turn_5', 'turn_20', 'vwap_dev',
]


# ═══════ 东方财富新闻抓取 ═══════

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': '*/*',
    'Referer': 'https://so.eastmoney.com/',
}

def fetch_news(code: str, limit: int = 5) -> list[dict]:
    """从东方财富搜索API抓取个股最新新闻标题"""
    try:
        param = json.dumps({
            "uid": "",
            "keyword": code,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": 1,
                    "pageSize": limit,
                    "preTag": "",
                    "postTag": "",
                }
            }
        }, ensure_ascii=False)

        url = (
            f"https://search-api-web.eastmoney.com/search/jsonp?"
            f"cb=jQuery&param={urllib.parse.quote(param)}"
        )

        req = urllib.request.Request(url, headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=10)
        raw = resp.read().decode('utf-8', errors='ignore')

        m = re.search(r'jQuery\((.*)\)', raw, re.DOTALL)
        if not m:
            return []

        data = json.loads(m.group(1))
        cms = data.get('result', {}).get('cmsArticleWebOld', [])
        # cms可能是list（直接是文章列表）或dict（有list子键）
        if isinstance(cms, dict):
            articles = cms.get('list', [])
        elif isinstance(cms, list):
            articles = cms
        else:
            articles = []

        news = []
        for a in articles[:limit]:
            title = a.get('title', '').strip()
            # 去掉HTML标签
            title = re.sub(r'<[^>]+>', '', title)
            if title:
                news.append({
                    'title': title,
                    'date': a.get('date', '')[:10],
                    'source': a.get('mediaName', ''),
                })
        return news

    except Exception as e:
        return [{'title': f'[抓取失败: {e}]', 'date': '', 'source': ''}]


def fetch_sector_news(stocks: list[tuple], limit_per_stock: int = 3) -> list[dict]:
    """抓取板块内所有龙头股的新闻，去重后返回"""
    all_news = []
    seen_titles = set()

    for code, name, weight in stocks:
        time.sleep(0.3)  # 避免请求过快
        news = fetch_news(code, limit=limit_per_stock)
        for n in news:
            title = n['title']
            if title not in seen_titles and not title.startswith('[抓取失败'):
                seen_titles.add(title)
                all_news.append({
                    'stock': name,
                    'code': code,
                    'title': title,
                    'date': n['date'],
                    'source': n['source'],
                })

    return all_news[:10]  # 每板块最多10条


# ═══════ L1 数据加载 ═══════

def load_stock_data(code: str, lookback_days: int = 120) -> pd.DataFrame:
    end = datetime.date.today()
    start = end - datetime.timedelta(days=lookback_days)

    for fname in [f'{code}.parquet', f'sh_{code}.parquet', f'sz_{code}.parquet']:
        fpath = DATA_DIR / fname
        if fpath.exists():
            df = pd.read_parquet(fpath)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                mask = (df['date'] >= str(start)) & (df['date'] <= str(end))
                df = df[mask].copy()
            if len(df) >= 30:
                # 数据新鲜度检查：parquet末日距今>10天视为过期，走baostock实时兜底
                last_date = df['date'].max().date()
                if (end - last_date).days <= 10:
                    return df
                print(f"  [WARN] {fname} 数据过期(末日{last_date})，改用baostock实时数据", file=sys.stderr)

    try:
        import baostock as bs
        # Redirect baostock login/logout messages to stderr
        import contextlib
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                bs.login()
        bs_code = f"sh.{code}" if code.startswith('6') else f"sz.{code}"
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount,turn",
            start_date=str(start), end_date=str(end),
            frequency="d", adjustflag="2"
        )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                bs.logout()

        if not rows:
            return pd.DataFrame()

        cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn']
        df = pd.DataFrame(rows, columns=cols)
        for c in cols[1:]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"  [WARN] Failed to fetch {code}: {e}", file=sys.stderr)
        return pd.DataFrame()


# ═══════ L1 因子计算 ═══════

def compute_features(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 60:
        return {}

    close = df['close'].values
    volume = df['volume'].values
    high = df['high'].values
    low = df['low'].values
    turn = df['turn'].values if 'turn' in df.columns else np.ones(len(df))

    f = {}
    f['ret_1'] = close[-1] / close[-2] - 1 if len(close) > 1 else 0
    f['ret_5'] = close[-1] / close[-6] - 1 if len(close) > 5 else 0
    f['ret_10'] = close[-1] / close[-11] - 1 if len(close) > 10 else 0
    f['ret_20'] = close[-1] / close[-21] - 1 if len(close) > 20 else 0

    f['mom_5'] = close[-1] / close[-6] - 1 if len(close) > 5 else 0
    f['mom_10'] = close[-1] / close[-11] - 1 if len(close) > 10 else 0
    f['mom_20'] = close[-1] / close[-21] - 1 if len(close) > 20 else 0
    f['mom_60'] = close[-1] / close[-61] - 1 if len(close) > 60 else 0

    rets = np.diff(close) / close[:-1]
    f['vol_5'] = np.std(rets[-5:]) if len(rets) >= 5 else 0
    f['vol_20'] = np.std(rets[-20:]) if len(rets) >= 20 else 0
    f['vol_60'] = np.std(rets[-60:]) if len(rets) >= 60 else 0

    if len(rets) >= 14:
        gains = np.where(rets[-14:] > 0, rets[-14:], 0)
        losses = np.where(rets[-14:] < 0, -rets[-14:], 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        f['rsi_14'] = 100 - 100 / (1 + avg_gain / avg_loss) if avg_loss > 0 else 100
    else:
        f['rsi_14'] = 50

    f['ma5_dev'] = close[-1] / np.mean(close[-5:]) - 1 if len(close) >= 5 else 0
    f['ma10_dev'] = close[-1] / np.mean(close[-10:]) - 1 if len(close) >= 10 else 0
    f['ma20_dev'] = close[-1] / np.mean(close[-20:]) - 1 if len(close) >= 20 else 0
    f['ma60_dev'] = close[-1] / np.mean(close[-60:]) - 1 if len(close) >= 60 else 0

    if len(close) >= 20:
        ma20 = np.mean(close[-20:])
        std20 = np.std(close[-20:])
        f['boll_pct'] = (close[-1] - ma20) / (2 * std20) if std20 > 0 else 0
    else:
        f['boll_pct'] = 0

    vol5 = np.mean(volume[-5:]) if len(volume) >= 5 else 1
    vol20 = np.mean(volume[-20:]) if len(volume) >= 20 else 1
    vol60 = np.mean(volume[-60:]) if len(volume) >= 60 else 1
    f['vol_ratio_5_20'] = vol5 / vol20 if vol20 > 0 else 1
    f['vol_ratio_5_60'] = vol5 / vol60 if vol60 > 0 else 1

    f['turn_5'] = np.mean(turn[-5:]) if len(turn) >= 5 else 0
    f['turn_20'] = np.mean(turn[-20:]) if len(turn) >= 20 else 0

    if 'amount' in df.columns and len(df) >= 5:
        amount = df['amount'].values
        vol = df['volume'].values
        vwap_5 = np.sum(amount[-5:]) / np.sum(vol[-5:]) if np.sum(vol[-5:]) > 0 else close[-1]
        f['vwap_dev'] = close[-1] / vwap_5 - 1
    else:
        f['vwap_dev'] = 0

    return f


def predict_stock(model_dict: dict, features: dict) -> float:
    if not features:
        return 0.5
    X = pd.DataFrame([{c: features.get(c, 0) for c in FEATURE_COLS}])
    X = X.fillna(0)
    booster = model_dict['model']
    score = booster.predict(X)[0]
    return float(score)


def aggregate_etf_signal(scores: list[tuple[str, float, float]]) -> dict:
    if not scores:
        return {"ml_score": 0.5, "direction": "中性", "confidence": "低"}

    weighted_score = sum(s * w for _, s, w in scores)
    total_weight = sum(w for _, _, w in scores)
    avg = weighted_score / total_weight if total_weight > 0 else 0.5

    if avg > 0.6:
        direction = "看涨"
        confidence = "高" if avg > 0.7 else "中"
    elif avg > 0.55:
        direction = "偏多"
        confidence = "中"
    elif avg < 0.4:
        direction = "看跌"
        confidence = "高" if avg < 0.3 else "中"
    elif avg < 0.45:
        direction = "偏空"
        confidence = "中"
    else:
        direction = "中性"
        confidence = "低"

    return {"ml_score": round(avg, 3), "direction": direction, "confidence": confidence}


# ═══════ 主流程 ═══════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-news", action="store_true", help="跳过新闻抓取（调试用）")
    args = parser.parse_args()

    now = datetime.datetime.now()

    # 加载模型
    if not MODEL_PATH.exists():
        print(json.dumps({"error": "Model not found", "path": str(MODEL_PATH)}))
        sys.exit(1)

    with open(MODEL_PATH, 'rb') as f:
        model_dict = pickle.load(f)

    signals = []

    for group_name, config in ETF_STOCKS.items():
        stock_results = []
        top_stocks = []

        for code, name, weight in config["stocks"]:
            df = load_stock_data(code)
            if df.empty:
                continue
            features = compute_features(df)
            score = predict_stock(model_dict, features)
            stock_results.append((name, score, weight))
            top_stocks.append({
                "code": code, "name": name,
                "score": round(score, 3), "weight": weight
            })

        agg = aggregate_etf_signal(stock_results)

        # 抓取板块舆情
        news = []
        if not args.no_news:
            print(f"  📰 抓取{group_name}新闻...", file=sys.stderr)
            news = fetch_sector_news(config["stocks"], limit_per_stock=3)

        signals.append({
            "group": group_name,
            "etfs": config["etfs"],
            "ml_score": agg["ml_score"],
            "direction": agg["direction"],
            "confidence": agg["confidence"],
            "top_stocks": sorted(top_stocks, key=lambda x: -x["score"]),
            "news": news,  # 原始新闻，供Agent链路消费
        })

    # 整体判断
    all_scores = [s["ml_score"] for s in signals]
    avg_all = np.mean(all_scores)
    if avg_all > 0.55:
        summary = "整体偏多"
    elif avg_all < 0.45:
        summary = "整体偏空"
    else:
        summary = "整体中性"

    sorted_signals = sorted(signals, key=lambda x: -x["ml_score"])
    summary += f"，最强：{sorted_signals[0]['group']}({sorted_signals[0]['direction']})"
    summary += f"，最弱：{sorted_signals[-1]['group']}({sorted_signals[-1]['direction']})"

    result = {
        "date": now.strftime("%Y-%m-%d"),
        "generated_at": now.strftime("%H:%M"),
        "signals": signals,
        "fusion_summary": summary,
    }

    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)

    if args.save:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(output, encoding='utf-8')
        print(f"\nSaved to {CACHE_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
