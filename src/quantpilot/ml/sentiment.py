"""
QuantPilot L3 - 情感分析因子（修正版）
StructBERT模型加载：key映射 encoder.* → bert.*
数据源：东方财富搜索API + 公告API（免费、无需认证）
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 模型路径
MODEL_DIR = os.path.expanduser(
    '~/workspace/quantpilot/models/models/'
    'iic--nlp_structbert_sentiment-classification_chinese-base/snapshots/master'
)


class SentimentAnalyzer:
    """
    StructBERT中文情感分析器
    二分类：正面(1) / 负面(0)，输出 score = pos - neg ∈ [-1, +1]
    """

    def __init__(self, model_dir: str = None, device: str = None):
        self.model_dir = model_dir or MODEL_DIR
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load_model(self):
        """加载模型（修正StructBERT的key映射）"""
        if self._loaded:
            return

        from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

        logger.info(f"Loading StructBERT from {self.model_dir}")
        logger.info(f"Device: {self.device}")

        # StructBERT权重前缀是 encoder.*，BertForSequenceClassification 期望 bert.*
        # 需要手动映射key
        state_path = os.path.join(self.model_dir, 'pytorch_model.bin')
        state = torch.load(state_path, map_location='cpu', weights_only=True)

        new_state = {}
        for k, v in state.items():
            if k.startswith('encoder.'):
                new_key = 'bert.' + k[len('encoder.'):]
            else:
                new_key = k
            new_state[new_key] = v

        # 用config初始化空模型，再加载修正后的权重
        config = BertConfig.from_pretrained(self.model_dir)
        config.num_labels = 2
        config.id2label = {0: '负面', 1: '正面'}
        config.label2id = {'负面': 0, '正面': 1}

        self.model = BertForSequenceClassification(config)
        missing, unexpected = self.model.load_state_dict(new_state, strict=False)

        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.debug(f"Unexpected keys: {len(unexpected)}")

        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir, use_fast=False)
        self._loaded = True
        logger.info("StructBERT loaded successfully ✓")

    def analyze_text(self, text: str) -> Dict[str, float]:
        """分析单条文本情感"""
        if not self._loaded:
            self.load_model()

        inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        neg, pos = float(probs[0]), float(probs[1])
        return {
            'positive': pos,
            'negative': neg,
            'sentiment_score': pos - neg,  # -1 到 +1
        }

    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """批量分析文本"""
        if not self._loaded:
            self.load_model()

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors='pt', truncation=True,
                                    max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for prob in probs:
                neg, pos = float(prob[0]), float(prob[1])
                results.append({
                    'positive': pos,
                    'negative': neg,
                    'sentiment_score': pos - neg,
                })

        return results


class SentimentFactorGenerator:
    """
    情感因子生成器
    输入：股票代码列表 → 输出：per-stock情感因子DataFrame
    """

    def __init__(self, model_dir: str = None):
        self.analyzer = SentimentAnalyzer(model_dir=model_dir)

    def generate_factor_from_news(self, code: str, news_items: list) -> Optional[Dict]:
        """
        从新闻列表生成单只股票的情感因子

        Args:
            code: 股票代码
            news_items: NewsItem列表（来自sentiment_data_v2）

        Returns:
            因子字典，或None（无数据时）
        """
        if not news_items:
            return None

        texts = [item.text() for item in news_items]
        sentiments = self.analyzer.analyze_batch(texts)
        scores = [s['sentiment_score'] for s in sentiments]

        return {
            'code': code,
            'sentiment_mean': np.mean(scores),
            'sentiment_std': np.std(scores) if len(scores) > 1 else 0,
            'sentiment_max': np.max(scores),
            'sentiment_min': np.min(scores),
            'sentiment_range': np.max(scores) - np.min(scores),
            'news_count': len(scores),
            'positive_ratio': sum(1 for s in scores if s > 0.1) / len(scores),
            'negative_ratio': sum(1 for s in scores if s < -0.1) / len(scores),
        }

    def generate_panel(self, stock_news: Dict[str, list], date: str = None) -> pd.DataFrame:
        """
        批量生成情感因子面板

        Args:
            stock_news: {code: [NewsItem, ...], ...}
            date: 日期（默认今天）

        Returns:
            DataFrame with columns: date, code, sentiment_mean, sentiment_std, ...
        """
        if date is None:
            from datetime import datetime
            date = datetime.now().strftime('%Y-%m-%d')

        results = []
        for code, items in stock_news.items():
            factor = self.generate_factor_from_news(code, items)
            if factor:
                factor['date'] = date
                results.append(factor)

        if not results:
            return pd.DataFrame(columns=[
                'date', 'code', 'sentiment_mean', 'sentiment_std',
                'sentiment_max', 'sentiment_min', 'sentiment_range',
                'news_count', 'positive_ratio', 'negative_ratio',
            ])

        df = pd.DataFrame(results)
        cols = ['date', 'code'] + [c for c in df.columns if c not in ('date', 'code')]
        return df[cols]


# ── 便捷函数 ──────────────────────────────────────────────

_analyzer = None

def get_analyzer() -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer

def analyze_sentiment(text: str) -> float:
    """快速分析单条文本情感分数 (-1到1)"""
    return get_analyzer().analyze_text(text)['sentiment_score']

def analyze_sentiment_batch(texts: List[str]) -> List[float]:
    """快速批量分析"""
    return [r['sentiment_score'] for r in get_analyzer().analyze_batch(texts)]
