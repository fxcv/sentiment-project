import re
import jieba

# 中文分词函数（原来你训练基线模型在用）
def chinese_tokenizer(text):
    return jieba.lcut(text)

# 长句自动切分函数
def split_text_for_sentiment(text):
    """
    按标点、连接词、空格对长句进行切分。
    如果切不出多段，就返回原句。
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # 1. 先按常见中文标点切
    parts = re.split(r"[，。；！？\n]+", text)
    parts = [p.strip() for p in parts if p.strip()]

    # 2. 再按连接词进一步切分
    connector_pattern = r"(但是|不过|然而|而且|并且|只是|可是|却)"
    refined_parts = []

    for part in parts:
        subparts = re.split(connector_pattern, part)

        temp = ""
        for sub in subparts:
            sub = sub.strip()
            if not sub:
                continue

            # 连接词单独出现时，作为新片段起点
            if sub in ["但是", "不过", "然而", "而且", "并且", "只是", "可是", "却"]:
                if temp:
                    refined_parts.append(temp.strip())
                temp = ""
            else:
                if temp:
                    temp += sub
                else:
                    temp = sub

        if temp:
            refined_parts.append(temp.strip())

    # 3. 如果还只有一段，再尝试按空格切
    if len(refined_parts) <= 1:
        space_parts = re.split(r"\s+", text)
        space_parts = [p.strip() for p in space_parts if p.strip()]
        if len(space_parts) > 1:
            refined_parts = space_parts

    # 4. 如果仍然只有一段，就返回原句
    if len(refined_parts) <= 1:
        return [text]

    return refined_parts