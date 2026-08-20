"""
QuantPilot L3 情感因子 - 端到端集成脚本
用法: python scripts/run_sentiment.py [--codes 300750,600519] [--limit 10] [--save]

完整流程:
1. 从东方财富获取个股新闻
2. StructBERT情感分析
3. 生成情感因子面板
4. 可选：保存到parquet
"""

import sys
import os
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.expanduser('~/workspace/quantpilot/src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='L3 Sentiment Factor Pipeline')
    parser.add_argument('--codes', type=str, default='300750,600519,002594,000858,601318',
                        help='股票代码，逗号分隔')
    parser.add_argument('--limit', type=int, default=10,
                        help='每只股票最大新闻数')
    parser.add_argument('--save', action='store_true',
                        help='保存结果到parquet')
    parser.add_argument('--date', type=str, default=None,
                        help='日期（默认今天）')
    args = parser.parse_args()

    codes = [c.strip() for c in args.codes.split(',')]
    date = args.date or datetime.now().strftime('%Y-%m-%d')

    # ── Step 1: 获取新闻 ──
    logger.info(f"Step 1: 获取新闻 ({len(codes)} stocks, limit={args.limit})")
    from quantpilot.ml.sentiment_data_v2 import UnifiedSentimentFetcher

    fetcher = UnifiedSentimentFetcher()
    stock_news = {}
    for i, code in enumerate(codes):
        items = fetcher.fetch_stock_news(code, limit=args.limit)
        stock_news[code] = items
        logger.info(f"  {code}: {len(items)} news")
        if i < len(codes) - 1:
            time.sleep(1.5)

    total_news = sum(len(v) for v in stock_news.values())
    logger.info(f"  Total: {total_news} news from {len(stock_news)} stocks")

    # ── Step 2: 情感分析 ──
    logger.info("Step 2: StructBERT情感分析")
    from quantpilot.ml.sentiment import SentimentFactorGenerator

    generator = SentimentFactorGenerator()
    panel = generator.generate_panel(stock_news, date=date)

    # ── Step 3: 输出结果 ──
    logger.info("Step 3: 情感因子面板")
    print(f"\n{'='*70}")
    print(f"  L3 情感因子面板 ({date})")
    print(f"{'='*70}")

    if panel.empty:
        print("  (no data)")
        return

    for _, row in panel.iterrows():
        code = row['code']
        mean = row['sentiment_mean']
        std = row['sentiment_std']
        pos_r = row['positive_ratio']
        neg_r = row['negative_ratio']
        count = int(row['news_count'])

        emoji = "🟢" if mean > 0.1 else ("🔴" if mean < -0.1 else "⚪")
        print(f"\n  {emoji} {code}")
        print(f"     sentiment_mean:   {mean:+.3f}")
        print(f"     sentiment_std:    {std:.3f}")
        print(f"     sentiment_range: [{row['sentiment_min']:+.3f}, {row['sentiment_max']:+.3f}]")
        print(f"     positive_ratio:  {pos_r:.0%}")
        print(f"     negative_ratio:  {neg_r:.0%}")
        print(f"     news_count:      {count}")

    # 汇总统计
    print(f"\n{'─'*70}")
    print(f"  汇总: {len(panel)} stocks, {total_news} news")
    print(f"  均值: {panel['sentiment_mean'].mean():+.3f} ± {panel['sentiment_mean'].std():.3f}")
    print(f"  最正面: {panel.loc[panel['sentiment_mean'].idxmax(), 'code']} ({panel['sentiment_mean'].max():+.3f})")
    print(f"  最负面: {panel.loc[panel['sentiment_mean'].idxmin(), 'code']} ({panel['sentiment_mean'].min():+.3f})")

    # ── Step 4: 保存 ──
    if args.save:
        out_dir = Path.home() / 'workspace' / 'quantpilot' / 'data' / 'ml' / 'sentiment'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'sentiment_{date}.parquet'
        panel.to_parquet(out_path, index=False)
        logger.info(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
