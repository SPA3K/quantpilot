"""轻量关键词情感分析 — 零外部依赖，财经新闻专用。

替代重型 NLP 模型（StructBERT/FinBERT）在内存受限环境下的空缺。
输出 ∈ [-1, +1]，>0.1 偏正面，<-0.1 偏负面。
"""

POS_WORDS = [
    "大涨", "暴涨", "超预期", "利好", "突破", "新高", "上调", "增持", "回购",
    "扩产", "获准", "签约", "中标", "合作", "发布", "升级", "创新", "增长",
    "盈利", "扭亏", "热销", "火爆", "订单", "领先", "获批", "专利", "最佳",
    "第一", "龙头", "布局", "加速", "放量", "金叉", "反弹", "修复", "看好",
    "推荐", "买入", "扩张", "翻倍", "爆发", "创纪录", "里程碑",
]

NEG_WORDS = [
    "大跌", "暴跌", "利空", "下滑", "亏损", "下调", "减持", "违规", "处罚",
    "诉讼", "召回", "事故", "裁员", "关停", "延期", "延迟", "缩水", "退市",
    "做空", "爆雷", "低于", "不及", "走弱", "破位", "死叉", "超卖", "逃逸",
    "维权", "投诉", "质量问题", "竞争加剧", "份额下降", "毛利率下降",
    "净利下降", "营收下降", "空头", "萎缩", "停滞", "恶化",
]


def text_sentiment(text: str) -> float:
    """单条文本情感打分 ∈ [-1, +1]"""
    pos = sum(1 for w in POS_WORDS if w in text)
    neg = sum(1 for w in NEG_WORDS if w in text)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


def news_sentiment(news_list: list[dict]) -> dict:
    """对新闻列表整体打分。

    Args:
        news_list: [{"title": ..., "content": ...}, ...]

    Returns:
        {"score": float, "pos": int, "neg": int, "neutral": int}
    """
    scores = [
        text_sentiment(n.get("title", "") + n.get("content", ""))
        for n in news_list
        if "error" not in n
    ]
    if not scores:
        return {"score": 0.0, "pos": 0, "neg": 0, "neutral": 0}
    pos = sum(1 for s in scores if s > 0.1)
    neg = sum(1 for s in scores if s < -0.1)
    neutral = len(scores) - pos - neg
    return {
        "score": round(sum(scores) / len(scores), 3),
        "pos": pos,
        "neg": neg,
        "neutral": neutral,
    }
