"""
QuantPilot L3 - 舆情数据获取
使用已有API获取财经新闻和舆情数据
"""

import urllib.request
import urllib.parse
import json
import re
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SentimentDataFetcher:
    """
    舆情数据获取器
    使用微博、Bing、东方财富等公开API
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'zh-CN,zh;q=0.9',
        }
    
    # ── 微博热搜 ──────────────────────────────────────────────
    
    def fetch_weibo_hot(self, top_n: int = 30) -> List[Dict]:
        """
        获取微博热搜
        返回: [{'word': '关键词', '热度': 数值, 'url': '链接'}, ...]
        """
        try:
            url = 'https://weibo.com/ajax/side/hotSearch'
            req = urllib.request.Request(url, headers=self.headers)
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read().decode())
            
            results = []
            for item in data['data']['realtime'][:top_n]:
                results.append({
                    'word': item.get('word', ''),
                    '热度': item.get('num', 0),
                    'url': f"https://s.weibo.com/weibo?q={urllib.parse.quote(item.get('word', ''))}",
                    'category': item.get('category', ''),
                })
            
            logger.info(f"Fetched {len(results)} weibo hot topics")
            return results
        except Exception as e:
            logger.error(f"Failed to fetch weibo hot: {e}")
            return []
    
    def filter_finance_weibo(self, hot_list: List[Dict]) -> List[Dict]:
        """筛选财经相关热搜"""
        finance_keywords = [
            '股', 'A股', '基金', '涨', '跌', '牛市', '熊市', '涨停', '跌停',
            '央行', '降息', '降准', '加息', '利率', '汇率', '人民币', '美元',
            'GDP', 'CPI', 'PMI', '经济', '通胀', '通缩', '失业',
            '半导体', '芯片', '新能源', '光伏', '锂电', 'AI', '人工智能',
            '茅台', '宁德', '比亚迪', '华为', '苹果', '特斯拉',
            '证监会', '银保监', '政策', '监管', 'IPO', '减持', '增持',
        ]
        
        finance_hot = []
        for item in hot_list:
            word = item['word']
            if any(kw in word for kw in finance_keywords):
                finance_hot.append(item)
        
        logger.info(f"Filtered {len(finance_hot)} finance topics from {len(hot_list)}")
        return finance_hot
    
    # ── Bing新闻搜索 ──────────────────────────────────────────────
    
    def fetch_bing_news(self, query: str, top_n: int = 10) -> List[Dict]:
        """
        Bing新闻搜索
        返回: [{'title': '标题', 'snippet': '摘要', 'url': '链接'}, ...]
        """
        try:
            q = urllib.parse.quote(query)
            url = f'https://www.bing.com/search?q={q}+新闻&filters=ex1%3a"ez5_18776_18781"'
            req = urllib.request.Request(url, headers=self.headers)
            resp = urllib.request.urlopen(req, timeout=15)
            html = resp.read().decode('utf-8', errors='ignore')
            
            # 解析搜索结果
            results = []
            
            # 提取标题和摘要
            # Bing的HTML结构：<h2><a href="...">标题</a></h2><p>摘要</p>
            title_pattern = r'<h2><a[^>]*href="([^"]*)"[^>]*>(.*?)</a></h2>'
            snippet_pattern = r'<p[^>]*>(.*?)</p>'
            
            titles = re.findall(title_pattern, html, re.DOTALL)
            snippets = re.findall(snippet_pattern, html, re.DOTALL)
            
            for i, (url, title) in enumerate(titles[:top_n]):
                # 清理HTML标签
                title = re.sub(r'<[^>]+>', '', title).strip()
                snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ''
                
                if title:
                    results.append({
                        'title': title,
                        'snippet': snippet,
                        'url': url,
                        'query': query,
                    })
            
            logger.info(f"Fetched {len(results)} news for '{query}'")
            return results
        except Exception as e:
            logger.error(f"Failed to fetch bing news for '{query}': {e}")
            return []
    
    def fetch_stock_news(self, stock_name: str, stock_code: str = None) -> List[Dict]:
        """获取个股新闻"""
        query = stock_name
        if stock_code:
            query = f"{stock_name} {stock_code}"
        return self.fetch_bing_news(query)
    
    def fetch_market_news(self, keywords: List[str] = None) -> List[Dict]:
        """获取市场新闻"""
        if keywords is None:
            keywords = ['A股', '股市', '基金', '央行', '经济']
        
        all_news = []
        for kw in keywords:
            news = self.fetch_bing_news(kw, top_n=5)
            all_news.extend(news)
            time.sleep(1)  # 避免请求过快
        
        return all_news
    
    # ── 东方财富板块情绪 ──────────────────────────────────────────────
    
    def fetch_eastmoney_sector_sentiment(self, top_n: int = 20) -> List[Dict]:
        """
        获取东方财富板块资金流向（反映市场情绪）
        返回: [{'name': '板块名', '涨跌幅': 数值, '主力净流入': 数值}, ...]
        """
        try:
            url = (
                'https://push2.eastmoney.com/api/qt/clist/get?'
                'pn=1&pz={top_n}&po=1&np=1&fltt=2&invt=2&fid=f62&fs=m:90+t:2'
                '&fields=f12,f14,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f3'
            ).format(top_n=top_n)
            
            req = urllib.request.Request(url, headers={
                'User-Agent': self.headers['User-Agent'],
                'Referer': 'https://data.eastmoney.com/'
            })
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read().decode())
            
            results = []
            for item in data['data']['diff']:
                results.append({
                    'name': item.get('f14', ''),
                    '涨跌幅': item.get('f3', 0),
                    '主力净流入': item.get('f62', 0) / 1e8,  # 转为亿
                    '超大单净流入': item.get('f66', 0) / 1e8,
                    '大单净流入': item.get('f72', 0) / 1e8,
                })
            
            logger.info(f"Fetched {len(results)} sector sentiment data")
            return results
        except Exception as e:
            logger.error(f"Failed to fetch eastmoney sector sentiment: {e}")
            return []
    
    # ── 综合舆情获取 ──────────────────────────────────────────────
    
    def fetch_all_sentiment(self, stock_names: List[str] = None) -> Dict:
        """
        获取综合舆情数据
        返回: {
            'weibo_hot': [...],
            'finance_hot': [...],
            'market_news': [...],
            'stock_news': {...},
            'sector_sentiment': [...],
        }
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'weibo_hot': [],
            'finance_hot': [],
            'market_news': [],
            'stock_news': {},
            'sector_sentiment': [],
        }
        
        # 微博热搜
        logger.info("Fetching weibo hot topics...")
        result['weibo_hot'] = self.fetch_weibo_hot(top_n=50)
        result['finance_hot'] = self.filter_finance_weibo(result['weibo_hot'])
        
        # 市场新闻
        logger.info("Fetching market news...")
        result['market_news'] = self.fetch_market_news(['A股', '股市', '央行', '经济'])
        
        # 个股新闻
        if stock_names:
            logger.info(f"Fetching stock news for {len(stock_names)} stocks...")
            for name in stock_names[:5]:  # 限制5只股票
                result['stock_news'][name] = self.fetch_stock_news(name)
                time.sleep(1)
        
        # 板块情绪
        logger.info("Fetching sector sentiment...")
        result['sector_sentiment'] = self.fetch_eastmoney_sector_sentiment(top_n=30)
        
        return result


# ── 便捷函数 ──────────────────────────────────────────────

_fetcher = None

def get_fetcher() -> SentimentDataFetcher:
    """获取全局数据获取器实例"""
    global _fetcher
    if _fetcher is None:
        _fetcher = SentimentDataFetcher()
    return _fetcher

def fetch_weibo_finance_hot() -> List[str]:
    """获取微博财经热搜关键词列表"""
    fetcher = get_fetcher()
    hot = fetcher.fetch_weibo_hot(top_n=50)
    finance = fetcher.filter_finance_weibo(hot)
    return [item['word'] for item in finance]

def fetch_stock_sentiment(stock_name: str) -> List[str]:
    """获取个股相关新闻文本"""
    fetcher = get_fetcher()
    news = fetcher.fetch_stock_news(stock_name)
    return [f"{item['title']}。{item['snippet']}" for item in news if item.get('snippet')]


# ── 测试 ──────────────────────────────────────────────

if __name__ == '__main__':
    print("Testing sentiment data fetcher...")
    
    fetcher = SentimentDataFetcher()
    
    # 测试微博热搜
    print("\n1. Fetching weibo hot topics...")
    hot = fetcher.fetch_weibo_hot(top_n=10)
    for item in hot[:5]:
        print(f"  - {item['word']} (热度: {item['热度']})")
    
    # 测试财经筛选
    finance = fetcher.filter_finance_weibo(hot)
    print(f"\n2. Finance topics: {len(finance)}")
    for item in finance[:3]:
        print(f"  - {item['word']}")
    
    # 测试新闻搜索
    print("\n3. Fetching A股 news...")
    news = fetcher.fetch_bing_news('A股', top_n=3)
    for item in news:
        print(f"  - {item['title'][:50]}...")
    
    # 测试板块情绪
    print("\n4. Fetching sector sentiment...")
    sectors = fetcher.fetch_eastmoney_sector_sentiment(top_n=5)
    for item in sectors:
        print(f"  - {item['name']}: 涨跌幅={item['涨跌幅']:.2f}%, 主力净流入={item['主力净流入']:.2f}亿")
