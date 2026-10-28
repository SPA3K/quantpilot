#!/usr/bin/env python3
"""
A股盘中监控脚本 v4 — DSA风格升级
数据采集 + 技术指标 + 交易日历 + 结构化输出
"""
import urllib.request, json, datetime, sys, math
import os
from pathlib import Path
STATE_DIR = Path(os.environ.get("QUANTPILOT_DATA_DIR")
             or (Path.home() / ".quantpilot" / "industry"))


# ═══════ 交易日历 ═══════
def is_trading_day(dt=None):
    """判断是否A股交易日（含节假日跳过）"""
    try:
        import exchange_calendars as xcals
        xshg = xcals.get_calendar("XSHG")
        dt = dt or datetime.datetime.now()
        ts = dt.strftime("%Y-%m-%d")
        return xshg.is_session(ts)
    except Exception:
        # fallback: 简单判断工作日
        dt = dt or datetime.datetime.now()
        return dt.weekday() < 5

# ═══════ 数据源：腾讯行情 ═══════
def tencent_quote(symbols):
    """腾讯行情接口"""
    url = f'https://qt.gtimg.cn/q={",".join(symbols)}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=10)
    raw = resp.read().decode('gbk')
    results = []
    for line in raw.strip().split('\n'):
        if '~' not in line:
            continue
        parts = line.split('~')
        if len(parts) < 40:
            continue
        results.append({
            'name': parts[1],
            'code': parts[2],
            'price': parts[3],
            'prev_close': parts[4],
            'open': parts[5],
            'volume': parts[6],
            'buy1': parts[9],
            'sell1': parts[19],
            'time': parts[30],
            'chg_pct': parts[32],
            'high': parts[33],
            'low': parts[34],
            'amount': parts[37],
            'turnover': parts[38],
        })
    return results

# ═══════ 数据源：东方财富 ═══════
def eastmoney_sector_flow(pz=10, po=1):
    """板块资金流向 (po=1净流入, po=0净流出)"""
    url = f'https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz={pz}&po={po}&np=1&fltt=2&invt=2&fid=f62&fs=m:90+t:2&fields=f14,f62,f3'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://data.eastmoney.com/'})
    resp = urllib.request.urlopen(req, timeout=8)
    data = json.loads(resp.read().decode())
    items = []
    if data.get('data') and data['data'].get('diff'):
        for item in data['data']['diff']:
            items.append({
                'name': item.get('f14', ''),
                'flow': item.get('f62', 0),
                'pct': item.get('f3', 0),
            })
    return items

def eastmoney_concept_board():
    """概念板块涨跌"""
    url = 'https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=60&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:90+t:3&fields=f14,f3,f62'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://data.eastmoney.com/'})
    resp = urllib.request.urlopen(req, timeout=8)
    data = json.loads(resp.read().decode())
    items = []
    if data.get('data') and data['data'].get('diff'):
        for item in data['data']['diff']:
            items.append({
                'name': item.get('f14', ''),
                'pct': item.get('f3', 0),
                'flow': item.get('f62', 0),
            })
    return items

# ═══════ 数据源：baostock历史K线 ═══════
_bs_logged_in = False

def _bs_login():
    global _bs_logged_in
    if not _bs_logged_in:
        try:
            import baostock as bs
            bs.login()
            _bs_logged_in = True
        except:
            pass

def _bs_logout():
    global _bs_logged_in
    if _bs_logged_in:
        try:
            import baostock as bs
            bs.logout()
            _bs_logged_in = False
        except:
            pass

def get_kline(symbol, days=60):
    """获取历史K线数据（baostock）"""
    try:
        import baostock as bs
        _bs_login()
        # baostock格式: sh.512480 / sz.159995
        code = symbol.replace('sh', 'sh.').replace('sz', 'sz.')
        if '.' not in code:
            return None
        from datetime import timedelta
        start = (datetime.datetime.now() - timedelta(days=days+10)).strftime('%Y-%m-%d')
        rs = bs.query_history_k_data_plus(
            code, 'date,close,volume,high,low,open',
            start_date=start, frequency='d'
        )
        rows = []
        while rs.error_code == '0' and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None
        # 返回 dict list
        result = []
        for r in rows:
            try:
                result.append({
                    'date': r[0],
                    'close': float(r[1]),
                    'volume': float(r[2]),
                    'high': float(r[3]),
                    'low': float(r[4]),
                    'open': float(r[5]),
                })
            except (ValueError, IndexError):
                continue
        return result if result else None
    except Exception as e:
        return None

# ═══════ 技术指标计算 ═══════
def calc_ma(closes, period):
    """简单移动平均"""
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period

def calc_ema(closes, period):
    """指数移动平均"""
    if len(closes) < period:
        return None
    k = 2 / (period + 1)
    ema = closes[0]
    for c in closes[1:]:
        ema = c * k + ema * (1 - k)
    return ema

def calc_macd(closes, fast=12, slow=26, signal=9):
    """MACD (DIF, DEA, MACD柱)"""
    if len(closes) < slow + signal:
        return None, None, None
    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)
    if ema_fast is None or ema_slow is None:
        return None, None, None
    # 逐日计算DIF序列
    dif_list = []
    k_fast = 2 / (fast + 1)
    k_slow = 2 / (slow + 1)
    ema_f = closes[0]
    ema_s = closes[0]
    for c in closes:
        ema_f = c * k_fast + ema_f * (1 - k_fast)
        ema_s = c * k_slow + ema_s * (1 - k_slow)
        dif_list.append(ema_f - ema_s)
    dif = dif_list[-1]
    # DEA = DIF的EMA
    dea = dif_list[0]
    k_sig = 2 / (signal + 1)
    for d in dif_list[1:]:
        dea = d * k_sig + dea * (1 - k_sig)
    macd_bar = 2 * (dif - dea)
    return dif, dea, macd_bar

def calc_rsi(closes, period=14):
    """RSI"""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_bias(closes, period=20):
    """乖离率 BIAS = (收盘价 - MA) / MA * 100"""
    ma = calc_ma(closes, period)
    if ma is None or ma == 0:
        return None
    return (closes[-1] - ma) / ma * 100

def technical_summary(symbol):
    """单只股票/ETF的技术指标汇总"""
    data = get_kline(symbol, days=60)
    if not data:
        return None
    closes = [d['close'] for d in data]
    if len(closes) < 30:
        return None

    price = closes[-1]
    ma5 = calc_ma(closes, 5)
    ma10 = calc_ma(closes, 10)
    ma20 = calc_ma(closes, 20)
    dif, dea, macd_bar = calc_macd(closes)
    rsi = calc_rsi(closes, 14)
    bias = calc_bias(closes, 20)

    # MA排列判断
    ma_trend = "—"
    if ma5 and ma10 and ma20:
        if ma5 > ma10 > ma20:
            ma_trend = "多头排列 📈"
        elif ma5 < ma10 < ma20:
            ma_trend = "空头排列 📉"
        else:
            ma_trend = "交织"

    # MACD信号
    macd_signal = "—"
    if dif is not None and dea is not None:
        if dif > dea and dif > 0:
            macd_signal = "水上金叉 ✅"
        elif dif > dea and dif < 0:
            macd_signal = "水下金叉 ⬆️"
        elif dif < dea and dif > 0:
            macd_signal = "水上死叉 ⚠️"
        elif dif < dea and dif < 0:
            macd_signal = "水下死叉 🔻"

    # RSI判断
    rsi_tag = "—"
    if rsi is not None:
        if rsi > 80:
            rsi_tag = f"超买({rsi:.0f}) ⚠️"
        elif rsi > 70:
            rsi_tag = f"偏强({rsi:.0f})"
        elif rsi < 20:
            rsi_tag = f"超卖({rsi:.0f}) ⬇️"
        elif rsi < 30:
            rsi_tag = f"偏弱({rsi:.0f})"
        else:
            rsi_tag = f"中性({rsi:.0f})"

    # 乖离率
    bias_tag = "—"
    if bias is not None:
        if abs(bias) > 5:
            bias_tag = f"乖离过大({bias:+.1f}%) 🚫不追"
        elif abs(bias) > 3:
            bias_tag = f"乖离偏大({bias:+.1f}%)"
        else:
            bias_tag = f"正常({bias:+.1f}%)"

    return {
        'price': price,
        'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
        'ma_trend': ma_trend,
        'dif': dif, 'dea': dea, 'macd_bar': macd_bar,
        'macd_signal': macd_signal,
        'rsi': rsi, 'rsi_tag': rsi_tag,
        'bias': bias, 'bias_tag': bias_tag,
    }

# ═══════ 输出格式化 ═══════
def fmt(q):
    """格式化单个股票/ETF输出"""
    pct = float(q['chg_pct']) if q['chg_pct'] else 0
    price = q['price']
    name = q['name']
    amount = float(q['amount']) if q['amount'] else 0
    turnover = float(q['turnover']) if q['turnover'] else 0
    open_p = q['open']
    high = q['high']
    low = q['low']

    if pct > 2:
        arrow = '🟢🟢'
    elif pct > 0.5:
        arrow = '🟢'
    elif pct > -0.5:
        arrow = '🟡'
    elif pct > -2:
        arrow = '🔴'
    else:
        arrow = '🔴🔴'

    line = f"  {arrow} {name}: {price} ({pct:+.2f}%)"
    if open_p and q['prev_close']:
        gap = (float(open_p) - float(q['prev_close'])) / float(q['prev_close']) * 100
        if abs(gap) > 0.3:
            line += f" [开盘{gap:+.2f}%]"
    if high and low and float(high) > 0:
        amp = (float(high) - float(low)) / float(low) * 100
        line += f" 振幅{amp:.1f}%"
    if amount > 10000:
        line += f" 额{amount/10000:.1f}亿"
    elif amount > 0:
        line += f" 额{amount:.0f}万"
    if turnover > 0:
        line += f" 换手{turnover:.1f}%"
    return line, pct

def section(title, symbols, show_tech=False):
    """输出一个板块的数据 + 可选技术指标"""
    print(f"\n{title}")
    total_pct = []
    for q in tencent_quote(symbols):
        line, pct = fmt(q)
        print(line)
        total_pct.append(pct)
        # 技术指标（仅ETF显示，个股太多会很慢）
        if show_tech:
            code = q['code']
            # 从原始symbol提取交易所前缀
            full_sym = next((s for s in symbols if code in s), code)
            tech = technical_summary(full_sym)
            if tech:
                print(f"    MA: {tech['ma_trend']} | MACD: {tech['macd_signal']} | RSI: {tech['rsi_tag']} | BIAS: {tech['bias_tag']}")
    if total_pct:
        avg = sum(total_pct) / len(total_pct)
        print(f"  📊 板块均涨幅: {avg:+.2f}%")

# ═══════ 微博热搜 ═══════
def weibo_hot():
    """微博热搜过滤"""
    try:
        url = 'https://weibo.com/ajax/side/hotSearch'
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://weibo.com/'
        })
        resp = urllib.request.urlopen(req, timeout=8)
        data = json.loads(resp.read().decode())
        if data.get('data') and data['data'].get('realtime'):
            keywords = [
                '特朗普', '访华', '芯片', '黄仁勋', '英伟达', 'NVIDIA',
                '半导体', '中美', '关税', '贸易', '科技', 'AI', '算力',
                '机器人', '具身智能', '人形机器人', 'CPO', '光模块',
                'A股', '股市', '基金', '银行', '央行', '降准', '降息',
            ]
            related = []
            for item in data['data']['realtime'][:50]:
                word = item.get('word', '')
                num = item.get('num', 0)
                if any(k in word for k in keywords):
                    related.append(f"  🔥 {word} (热度:{num})")
            return related[:10]
    except Exception as e:
        return [f"  获取失败: {e}"]
    return ["  暂无相关热搜"]

# ═══════ 北向资金 ═══════
def northbound():
    """北向资金近3日"""
    try:
        url = 'https://push2.eastmoney.com/api/qt/kamtbs.wh?fields1=f1,f2,f3&fields2=f51,f52,f53,f54,f55,f56'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://data.eastmoney.com/'})
        resp = urllib.request.urlopen(req, timeout=8)
        data = json.loads(resp.read().decode())
        lines = []
        if data.get('data') and data['data'].get('s2n'):
            for day in data['data']['s2n'][-3:]:
                if isinstance(day, list) and len(day) >= 4:
                    lines.append(f"  {day[0]}: 沪股通{day[1]} 深股通{day[2]} 合计{day[3]}")
        return lines
    except Exception as e:
        return [f"  获取失败: {e}"]

# ═══════ 主流程 ═══════
def main():
    now = datetime.datetime.now()
    trading = is_trading_day(now)

    print(f"{'='*50}")
    print(f"A股盘中监控 v4 | {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"交易日: {'✅ 是' if trading else '❌ 否（节假日/周末）'}")
    print(f"{'='*50}")

    if not trading:
        print("\n⚠️ 今日非交易日，跳过实时行情采集。")
        print("以下为最近一个交易日的收盘数据参考：\n")

    # ━━━ 美股/全球夜盘 ━━━
    print("\n━━━ 🌙 美股隔夜/全球 ━━━")
    us = ['usNVDA', 'usTSLA', 'usAAPL', 'usAMD', 'usAVGO', 'usQCOM', 'usINTC', 'usMRVL']
    for q in tencent_quote(us):
        pct = float(q['chg_pct']) if q['chg_pct'] else 0
        arrow = '🟢' if pct > 0.5 else '🔴' if pct < -0.5 else '🟡'
        print(f"  {arrow} {q['name']}: ${q['price']} ({pct:+.2f}%)")

    print("\n  --- 美股半导体ETF ---")
    for q in tencent_quote(['usSOXX', 'usSMH']):
        pct = float(q['chg_pct']) if q['chg_pct'] else 0
        arrow = '🟢' if pct > 0.5 else '🔴' if pct < -0.5 else '🟡'
        print(f"  {arrow} {q['name']}: ${q['price']} ({pct:+.2f}%)")

    # ━━━ 港股科技 ━━━
    print("\n  --- 港股科技 ---")
    for q in tencent_quote(['hk09888', 'hk03690', 'hk09988', 'hk01810', 'hk00981', 'hk09868']):
        pct = float(q['chg_pct']) if q['chg_pct'] else 0
        arrow = '🟢' if pct > 0.5 else '🔴' if pct < -0.5 else '🟡'
        print(f"  {arrow} {q['name']}: {q['price']} ({pct:+.2f}%)")

    # ━━━ 大盘指数 ━━━
    print("\n━━━ 📈 大盘指数 ━━━")
    for q in tencent_quote(['sh000001', 'sz399001', 'sz399006', 'sh000688']):
        pct = float(q['chg_pct']) if q['chg_pct'] else 0
        amount = float(q['amount']) if q['amount'] else 0
        arrow = '📈' if pct > 0 else '📉'
        gap = ''
        if q['open'] and q['prev_close']:
            g = (float(q['open']) - float(q['prev_close'])) / float(q['prev_close']) * 100
            gap = f" [开盘{g:+.2f}%]"
        print(f"  {arrow} {q['name']}: {q['price']} ({pct:+.2f}%){gap} 额{amount/1e8:.0f}亿")

    # ━━━ 板块个股 ━━━
    section("\n━━━ 🔬 半导体/芯片个股 ━━━", [
        'sz002371', 'sh688981', 'sh688012', 'sh688008', 'sh688256',
        'sz300782', 'sh600183', 'sz002049', 'sz300661', 'sh603501',
        'sh688396', 'sz300223',
    ])

    section("\n━━━ 💡 CPO/光通信 ━━━", [
        'sz300308', 'sz300502', 'sh600487', 'sh600522',
        'sz300394', 'sz300620', 'sz002281', 'sz300570',
    ])

    section("\n━━━ 🤖 具身智能/机器人个股 ━━━", [
        'sz300124', 'sh688169', 'sz002747', 'sh603486', 'sh688160',
        'sz002527', 'sz300263', 'sz002472', 'sh603728', 'sz300607',
        'sz300970', 'sh601689',
    ])

    section("\n━━━ 📱 消费电子/苹果特斯拉链 ━━━", [
        'sz002475', 'sz300433', 'sh601231', 'sz002415', 'sz300124',
    ])

    section("\n━━━ 🏦 银行个股 ━━━", [
        'sh601398', 'sh601288', 'sh601939', 'sh601988', 'sh601328',
        'sh600036', 'sh601166', 'sh600016', 'sh600000', 'sh601818',
        'sz000001', 'sh601998',
    ])

    section("\n━━━ 🚗 新能源/汽车链 ━━━", [
        'sz300750', 'sz002594', 'sz002812', 'sh600438',
    ])

    # ━━━ ETF行情 + 技术指标 ━━━
    print("\n━━━ 📊 关键ETF（含技术指标）━━━")
    etf_groups = [
        ("🔬 芯片/半导体", ['sh512480', 'sh512760', 'sz159995', 'sh588200']),
        ("💡 CPO/通信", ['sh515880', 'sh515050', 'sh512880']),
        ("🤖 具身/机器人", ['sh562500', 'sh562300', 'sz159770']),
        ("📊 科技宽基", ['sh515030', 'sh588000', 'sh515750', 'sh512720']),
        ("🏦 银行ETF", ['sh512800', 'sh515020', 'sh512820']),
    ]
    for group_name, syms in etf_groups:
        print(f"\n  {group_name}:")
        for q in tencent_quote(syms):
            line, _ = fmt(q)
            print(line)
            # 技术指标
            full_sym = next((s for s in syms if q['code'] in s), q['code'])
            tech = technical_summary(full_sym)
            if tech:
                print(f"    MA: {tech['ma_trend']} | MACD: {tech['macd_signal']} | RSI: {tech['rsi_tag']} | BIAS: {tech['bias_tag']}")

    # ━━━ ML模型预测信号 ━━━
    print("\n━━━ 🤖 QuantPilot ML模型信号 ━━━")
    ml_cache = STATE_DIR / "ml_signal_cache.json"
    if ml_cache.exists():
        try:
            ml_data = json.loads(ml_cache.read_text(encoding='utf-8'))
            print(f"  模型: L1 LightGBM (22因子) | 生成时间: {ml_data.get('generated_at', '?')}")
            print(f"  综合判断: {ml_data.get('fusion_summary', '?')}")
            for sig in ml_data.get('signals', []):
                score = sig['ml_score']
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"  {sig['group']:12s} | {bar} {score:.3f} | {sig['direction']}({sig['confidence']})")
                # 显示最强个股
                top = sig.get('top_stocks', [{}])[0]
                if top:
                    print(f"    → 龙头: {top.get('name','')}({top.get('score',0):.3f})")
        except Exception as e:
            print(f"  读取失败: {e}")
    else:
        print("  ⚠️ 无ML信号缓存，运行 python3 ml_etf_signal.py --save 生成")

    # ━━━ 板块资金流向 ━━━
    print("\n━━━ 💰 板块主力资金 ━━━")
    print("  --- 净流入TOP10 ---")
    try:
        for item in eastmoney_sector_flow(10, 1):
            flow_str = f'{item["flow"]/1e8:.1f}亿' if isinstance(item["flow"], (int, float)) and abs(item["flow"]) > 1e6 else 'N/A'
            print(f"  {item['name']}: {item['pct']:+.2f}% 净流入{flow_str}")
    except Exception as e:
        print(f"  获取失败: {e}")

    print("\n  --- 净流出TOP5 ---")
    try:
        for item in eastmoney_sector_flow(5, 0):
            flow_str = f'{item["flow"]/1e8:.1f}亿' if isinstance(item["flow"], (int, float)) and abs(item["flow"]) > 1e6 else 'N/A'
            print(f"  {item['name']}: {item['pct']:+.2f}% 净流出{flow_str}")
    except Exception as e:
        print(f"  获取失败: {e}")

    # ━━━ 科技概念板块 ━━━
    print("\n━━━ 🔖 科技概念板块 ━━━")
    keywords = [
        '英伟达', '算力', '半导体', '芯片', 'CPO', 'HBM', '存储', '光刻',
        '碳化硅', '封装', 'AIPC', '消费电子', 'AI', '光模块', '光通信',
        '机器人', '具身智能', '人形机器人', '减速器', '伺服', '华为', '苹果', '特斯拉',
        '银行', '金融', '四大行', '股份行', '央行', '降准', '降息', '救市',
    ]
    try:
        shown = 0
        for item in eastmoney_concept_board():
            if any(k in item['name'] for k in keywords):
                flow_str = f'{item["flow"]/1e8:.1f}亿' if isinstance(item["flow"], (int, float)) and abs(item["flow"]) > 1e6 else 'N/A'
                marker = '⭐' if any(k in item['name'] for k in ['英伟达', 'CPO', '光模块', '半导体', '芯片', '算力', '机器人', '具身智能', '人形机器人', '减速器', '银行', '金融', '降准', '降息', '救市']) else '  '
                print(f"  {marker} {item['name']}: {item['pct']:+.2f}% 主力:{flow_str}")
                shown += 1
                if shown >= 18:
                    break
    except Exception as e:
        print(f"  获取失败: {e}")

    # ━━━ 微博热搜 ━━━
    print("\n━━━ 📰 微博热搜(相关) ━━━")
    for line in weibo_hot():
        print(line)

    # ━━━ 北向资金 ━━━
    print("\n━━━ 🌊 北向资金 ━━━")
    for line in northbound():
        print(line)

    # ━━━ 尾部 ━━━
    _bs_logout()  # 释放baostock连接
    print(f"\n{'='*50}")
    print(f"采集完成: {now.strftime('%H:%M:%S')} | v4含技术指标+交易日历")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
