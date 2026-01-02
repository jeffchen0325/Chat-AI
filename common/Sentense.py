import re
from typing import List, Tuple

def split_paragraph(text: str, min_length: int = 4) -> List[str]:
    """简单分句：返回完整的句子列表"""
    if not text:
        return []

    # 使用正则表达式分割句子
    pattern = r'[。！？!?.;;\n]+'
    parts = re.split(pattern, text)

    # 过滤空字符串和过短的句子
    sentences = [
        sentence.strip()
        for sentence in parts
        if sentence.strip() and len(sentence.strip()) >= min_length
    ]

    return sentences
