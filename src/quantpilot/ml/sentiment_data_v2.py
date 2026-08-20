"""
QuantPilot L3 - 舆情数据获取（v2）
经过实测验证的免费数据源：
  1. 东方财富搜索API（JSONP）→ 个股新闻+内容
  2. 东方财富公告API → 公司公告
  3. AkShare（可选）→ 东方财富新闻封装

不依赖第三方库，纯urllib实现
"""

import json
import re
import time
import urllib.request
import urllib.parse
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 统一输出格式
# ═══════════════════════════════════════════════════════════════

class NewsItem:
    """统一新闻条目"""
    __slots__ = ['code', 'title', 'content', 'source', 'url', 'date']

    def __init__(self, code: str, title: str, content: str = '',
                 source: str = '', url: str = '', date: str = ''):
        self.code = code
        self.title = title
        self.content = content
        self.source = source
        self.url = url
        self.date = date

    def text(self) -> str:
        """合并标题+正文，用于BERT输入（截断到300字避免超512 token）"""
        parts = [self.title]
        if self.content:
            parts.append(self.content[:300])
        return '。'.join(parts)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


# ═══════════════════════════════════════════════════════════════
# 数据源1：东方财富搜索API（实测可用 ✅）
# ═══════════════════════════════════════════════════════════════

class EastMoneySearchFetcher:
    """
    东方财富搜索API — per-stock新闻，返回title+content+date
    JSONP格式，需要解析callback包装

    实测端点：search-api-web.eastmoney.com/search/jsonp
    """
    name = "eastmoney_search"

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': '*/*',
        'Referer': 'https://so.eastmoney.com/',
    }

    def fetch(self, code: str, limit: int = 20) -> List[NewsItem]:
        try:
            # JSONP搜索API，搜股票代码+关键词
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

            req = urllib.request.Request(url, headers=self.HEADERS)
            resp = urllib.request.urlopen(req, timeout=15)
            raw = resp.read().decode('utf-8', errors='ignore')

            # 解析JSONP：jQuery({...})
            json_str = re.search(r'jQuery\((.*)\)', raw)
            if not json_str:
                logger.warning(f"[em_search] {code}: JSONP parse failed")
                return []

            data = json.loads(json_str.group(1))
            articles = data.get('result', {}).get('cmsArticleWebOld', [])

            items = []
            for art in articles:
                title = art.get('title', '').strip()
                content = art.get('content', '').strip()
                # 清理HTML标签
                title = re.sub(r'<[^>]+>', '', title)
                content = re.sub(r'<[^>]+>', '', content)

                if not title:
                    continue

                items.append(NewsItem(
                    code=code,
                    title=title,
                    content=content,
                    source='eastmoney_search',
                    url=art.get('url', ''),
                    date=art.get('date', ''),
                ))

            logger.info(f"[em_search] {code}: {len(items)} news")
            return items

        except Exception as e:
            logger.error(f"[em_search] {code} failed: {e}")
            return []


# ═══════════════════════════════════════════════════════════════
# 数据源2：东方财富公告API（实测可用 ✅）
# ═══════════════════════════════════════════════════════════════

class EastMoneyAnnouncementFetcher:
    """
    东方财富公告API — 公司公告（法律意见书、业绩报告等）

    实测端点：np-anotice-stock.eastmoney.com/api/security/ann
    """
    name = "eastmoney_ann"

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://data.eastmoney.com/',
    }

    def fetch(self, code: str, limit: int = 10) -> List[NewsItem]:
        try:
            url = (
                f"https://np-anotice-stock.eastmoney.com/api/security/ann?"
                f"page_size={limit}&page_index=1&ann_type=A"
                f"&stock_list={code}&f_node=0"
            )

            req = urllib.request.Request(url, headers=self.HEADERS)
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read().decode('utf-8'))

            items = []
            ann_list = data.get('data', {}).get('list', [])
            for ann in ann_list:
                # 公告标题在columns里
                columns = ann.get('columns', [])
                title = columns[0].get('column_name', '') if columns else ''

                # 公告正文需要单独请求（这里只取标题+摘要）
                # art_code可以用来获取详情
                art_code = ann.get('art_code', '')

                # 简单标题作为内容
                if not title:
                    # 从codes里构造标题
                    codes_info = ann.get('codes', [])
                    if codes_info:
                        short_name = codes_info[0].get('short_name', code)
                        title = f"{short_name}公告"

                items.append(NewsItem(
                    code=code,
                    title=title,
                    content='',  # 公告详情需要额外请求
                    source='eastmoney_ann',
                    url=f"https://data.eastmoney.com/notices/detail/{code}/{art_code}.html",
                    date=ann.get('display_time', ''),
                ))

            logger.info(f"[em_ann] {code}: {len(items)} announcements")
            return items

        except Exception as e:
            logger.error(f"[em_ann] {code} failed: {e}")
            return []


# ═══════════════════════════════════════════════════════════════
# 数据源3：AkShare（可选，需安装）
# ═══════════════════════════════════════════════════════════════

class AkShareFetcher:
    """AkShare封装 — 需要 pip install akshare"""
    name = "akshare"

    def is_available(self) -> bool:
        try:
            import akshare
            return True
        except ImportError:
            return False

    def fetch(self, code: str, limit: int = 20) -> List[NewsItem]:
        if not self.is_available():
            return []
        try:
            import akshare as ak
            df = ak.stock_news_em(symbol=code)
            if df is None or df.empty:
                return []

            items = []
            for _, row in df.head(limit).iterrows():
                items.append(NewsItem(
                    code=code,
                    title=str(row.get('新闻标题', '')),
                    content=str(row.get('新闻内容', '')),
                    source='akshare',
                    url=str(row.get('新闻链接', '')),
                    date=str(row.get('发布时间', '')),
                ))
            logger.info(f"[akshare] {code}: {len(items)} news")
            return items
        except Exception as e:
            logger.error(f"[akshare] {code} failed: {e}")
            return []


# ═══════════════════════════════════════════════════════════════
# 统一调度器
# ═══════════════════════════════════════════════════════════════

class UnifiedSentimentFetcher:
    """
    统一舆情获取器
    按优先级尝试数据源，失败自动fallback
    """

    def __init__(self):
        self.fetchers = [
            AkShareFetcher(),                # 优先级1（需安装）
            EastMoneySearchFetcher(),        # 优先级2（免费，JSONP）
            EastMoneyAnnouncementFetcher(),  # 优先级3（免费，公告）
        ]

    def fetch_stock_news(self, code: str, limit: int = 20,
                         source: str = None) -> List[NewsItem]:
        """
        获取个股新闻，自动fallback

        Args:
            code: 股票代码，如 "300750"
            limit: 最大条数
            source: 指定数据源名（可选）
        """
        if source:
            fetcher = next((f for f in self.fetchers if f.name == source), None)
            if fetcher:
                return fetcher.fetch(code, limit)
            logger.error(f"Unknown source: {source}")
            return []

        # 自动fallback
        for fetcher in self.fetchers:
            if hasattr(fetcher, 'is_available') and not fetcher.is_available():
                continue
            items = fetcher.fetch(code, limit)
            if items:
                return items
            time.sleep(0.3)

        logger.warning(f"All sources failed for {code}")
        return []

    def fetch_multi_stocks(self, codes: List[str], limit_per_stock: int = 10,
                           delay: float = 1.5) -> Dict[str, List[NewsItem]]:
        """批量获取多只股票新闻"""
        results = {}
        for i, code in enumerate(codes):
            items = self.fetch_stock_news(code, limit_per_stock)
            results[code] = items
            if i < len(codes) - 1:
                time.sleep(delay)
            if (i + 1) % 20 == 0:
                logger.info(f"Progress: {i+1}/{len(codes)}")
        return results

    def status(self) -> Dict[str, bool]:
        """返回各数据源状态"""
        result = {}
        for f in self.fetchers:
            if hasattr(f, 'is_available'):
                result[f.name] = f.is_available()
            else:
                result[f.name] = True  # 内置源总是可用
        return result


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

_fetcher = None

def get_unified_fetcher() -> UnifiedSentimentFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = UnifiedSentimentFetcher()
    return _fetcher


# ═══════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    fetcher = UnifiedSentimentFetcher()
    print(f"\n=== 数据源状态 ===")
    for name, ok in fetcher.status().items():
        print(f"  {'✓' if ok else '✗'} {name}")

    # 测试3只股票
    test_codes = ['300750', '600519', '002594']

    for code in test_codes:
        print(f"\n{'='*50}")
        print(f"  {code}")
        print(f"{'='*50}")

        for source_name in ['eastmoney_search', 'eastmoney_ann', 'akshare']:
            print(f"\n  --- {source_name} ---")
            items = fetcher.fetch_stock_news(code, limit=5, source=source_name)
            if items:
                for item in items[:3]:
                    print(f"    [{item.date[:10]}] {item.title[:55]}")
                    if item.content:
                        print(f"      {item.content[:70]}...")
            else:
                print(f"    (no data)")
