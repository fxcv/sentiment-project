import os
import pickle
import requests
import pandas as pd
import streamlit as st
import torch
from urllib.parse import urlparse, parse_qs
from transformers import BertTokenizer, BertForSequenceClassification
from text_utils import split_text_for_sentiment

# =============================
# 页面配置
# =============================
st.set_page_config(
    page_title="中文文本情感分析系统",
    page_icon="📊",
    layout="centered"
)

# =============================
# 轻量样式
# =============================
st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
.info-chip {
    display: inline-block;
    background: #f5f7fb;
    border: 1px solid #e4e9f2;
    color: #4f5f78;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 13px;
    margin-right: 8px;
    margin-bottom: 8px;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #e8edf5;
    border-radius: 10px;
    overflow: hidden;
}
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# =============================
# 路径自动选择
# =============================
def resolve_model_dir(preferred: str, fallback: str) -> str:
    return preferred if os.path.exists(preferred) else fallback

bert_model_3class_path = "models/bert_model_final_3class_v2"
bert_model_7class_path = "models/bert_model_final_7class"
# =============================
# 加载三分类基线模型
# =============================
with open("models/model.pkl", "rb") as f:
    baseline_model_3class = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer_3class = pickle.load(f)

# =============================
# 加载三分类 BERT 模型
# =============================
bert_tokenizer_3class = BertTokenizer.from_pretrained(bert_model_3class_path)
bert_model_3class = BertForSequenceClassification.from_pretrained(bert_model_3class_path)

# =============================
# 加载7分类基线模型
# =============================
with open("models/merged_baseline_7class_model.pkl", "rb") as f:
    baseline_model_7class = pickle.load(f)

with open("models/merged_baseline_7class_vectorizer.pkl", "rb") as f:
    vectorizer_7class = pickle.load(f)

# =============================
# 加载7分类 BERT 模型
# =============================
bert_tokenizer_7class = BertTokenizer.from_pretrained(bert_model_7class_path)
bert_model_7class = BertForSequenceClassification.from_pretrained(bert_model_7class_path)

# =============================
# 标签映射
# =============================
sentiment_label_map = {
    0: "负面",
    1: "正面",
    2: "中性"
}

fine_label_map = {
    0: "喜悦",
    1: "喜欢",
    2: "愤怒",
    3: "悲伤",
    4: "恐惧",
    5: "厌恶",
    6: "惊讶"
}

# =============================
# 阈值
# =============================
COARSE_CONF_THRESHOLD = 0.55
COARSE_MARGIN_THRESHOLD = 0.10

# 二级更严格
FINE_THRESHOLD = 0.65
MARGIN_THRESHOLD = 0.10

# =============================
# 工具函数
# =============================
def get_confidence_info(proba, conf_threshold=0.60, margin_threshold=0.15):
    sorted_probs = sorted(proba, reverse=True)
    max_prob = sorted_probs[0]
    second_prob = sorted_probs[1]
    margin = max_prob - second_prob
    is_low_conf = (max_prob < conf_threshold) or (margin < margin_threshold)
    return {
        "confidence": max_prob,
        "margin": margin,
        "is_low_conf": is_low_conf
    }

def predict_3class(text, use_bert: bool):
    if not use_bert:
        text_features = vectorizer_3class.transform([text])
        pred = baseline_model_3class.predict(text_features)[0]
        proba = baseline_model_3class.predict_proba(text_features)[0]
    else:
        inputs = bert_tokenizer_3class(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = bert_model_3class(**inputs)
            logits = outputs.logits
            proba = torch.softmax(logits, dim=1).squeeze().tolist()
            pred = torch.argmax(logits, dim=1).item()

    return {
        "pred": pred,
        "label": sentiment_label_map[pred],
        "proba": proba,
        "conf": get_confidence_info(proba, COARSE_CONF_THRESHOLD, COARSE_MARGIN_THRESHOLD)
    }

def predict_7class(text, use_bert: bool):
    if not use_bert:
        text_features = vectorizer_7class.transform([text])
        pred = baseline_model_7class.predict(text_features)[0]
        proba = baseline_model_7class.predict_proba(text_features)[0]
    else:
        inputs = bert_tokenizer_7class(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = bert_model_7class(**inputs)
            logits = outputs.logits
            proba = torch.softmax(logits, dim=1).squeeze().tolist()
            pred = torch.argmax(logits, dim=1).item()

    fine_label = fine_label_map[pred]
    conf = get_confidence_info(proba, FINE_THRESHOLD, MARGIN_THRESHOLD)
    fine_result = fine_label if not conf["is_low_conf"] else "暂未细分"

    return {
        "pred": pred,
        "fine_label_raw": fine_label,
        "fine_result": fine_result,
        "proba": proba,
        "conf": conf
    }

def predict_joint(text, use_bert: bool):
    coarse_result = predict_3class(text, use_bert)
    fine_result = predict_7class(text, use_bert)
    return {
        "coarse_label": coarse_result["label"],
        "coarse_proba": coarse_result["proba"],
        "coarse_conf": coarse_result["conf"],
        "fine_label_raw": fine_result["fine_label_raw"],
        "fine_result": fine_result["fine_result"],
        "fine_proba": fine_result["proba"],
        "fine_conf": fine_result["conf"]
    }

def get_input_stats(text: str):
    chars = len(text)
    lines = [line for line in text.split("\n") if line.strip()]
    line_count = len(lines)

    if line_count > 1:
        mode = "批量输入"
    else:
        segments = split_text_for_sentiment(text.strip()) if text.strip() else []
        if len(segments) > 1:
            mode = "长句拆分"
        else:
            mode = "单句分析"
    return chars, line_count, mode

def style_result_df(df: pd.DataFrame):
    def color_emotion(val):
        if val == "正面":
            return "background-color: #eaf7ef; color: #1f7a45; font-weight: 700;"
        if val == "负面":
            return "background-color: #fdecec; color: #b13a3a; font-weight: 700;"
        if val == "中性":
            return "background-color: #eef2f7; color: #4b6078; font-weight: 700;"
        return ""

    def color_fine(val):
        if val == "暂未细分":
            return "background-color: #f5f7fb; color: #73839a; font-weight: 700;"
        return "background-color: #eef4ff; color: #2b5fb8; font-weight: 700;"

    return (
        df.style
        .map(color_emotion, subset=["一级情感"])
        .map(color_fine, subset=["二级情绪"])
        .format({
            "一级置信度": "{:.4f}",
            "二级置信度": "{:.4f}"
        })
    )

# =============================
# YouTube 工具函数
# =============================
def load_youtube_api_key():
    if "YOUTUBE_API_KEY" in st.secrets:
        return st.secrets["YOUTUBE_API_KEY"]

    filepath = "youtube_api_key.txt"
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            key = f.read().strip()
        if key and "把你的API_KEY填这里" not in key:
            return key

    raise ValueError("未找到有效的 YouTube API Key")

def extract_video_id(url: str) -> str:
    url = url.strip()

    # 1) 短链接
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0].split("/")[0]

    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    path = parsed.path.strip("/")

    # 2) 标准 watch 链接
    if "youtube.com" in netloc:
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            return qs["v"][0]

        # 3) shorts 链接
        if path.startswith("shorts/"):
            parts = path.split("/")
            if len(parts) >= 2 and parts[1]:
                return parts[1]

        # 4) embed 链接
        if path.startswith("embed/"):
            parts = path.split("/")
            if len(parts) >= 2 and parts[1]:
                return parts[1]

        # 5) live 链接（顺手兼容）
        if path.startswith("live/"):
            parts = path.split("/")
            if len(parts) >= 2 and parts[1]:
                return parts[1]

    raise ValueError("无法识别视频链接，请输入标准 YouTube 视频链接。")

def fetch_youtube_comments(video_id: str, api_key: str, max_comments: int = 100):
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        remaining = max_comments - len(comments)
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100 if remaining >= 100 else remaining,
            "textFormat": "plainText",
            "key": api_key
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        resp = requests.get(url, params=params, timeout=20)

        if resp.status_code != 200:
            raise RuntimeError(f"YouTube API 请求失败：{resp.text}")

        data = resp.json()

        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["id"],
                "author": snippet.get("authorDisplayName", ""),
                "text": snippet.get("textDisplay", ""),
                "like_count": snippet.get("likeCount", 0),
                "published_at": snippet.get("publishedAt", "")
            })

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return comments

# =============================
# 页面头部
# =============================
st.title("中文文本情感分析系统")
st.caption("支持普通文本分析与 YouTube 评论区氛围分析。")

if "demo_text" not in st.session_state:
    st.session_state.demo_text = ""

# =============================
# 模式选择
# =============================
analysis_mode = st.selectbox(
    "分析模式",
    ["文本分析", "YouTube 评论区氛围分析"]
)

# =============================
# 模型配置
# =============================
st.subheader("模型配置")
m1, m2 = st.columns([1, 0.25], gap="large")

with m1:
    model_choice = st.selectbox(
        "模型类型",
        ["BERT模型", "基线模型（TF-IDF + 逻辑回归）"]
    )
with m2:
    st.write("")
    st.write("")
    clear_input_btn = st.button("清空文本", use_container_width=True)

if clear_input_btn:
    st.session_state.demo_text = ""

use_bert = (model_choice == "BERT模型")

st.divider()

# =============================
# 文本分析模式
# =============================
if analysis_mode == "文本分析":
    st.subheader("文本输入")
    st.caption("支持单句分析、长句拆分分析和多句批量分析。")

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    if r1c1.button("😊 喜悦示例", use_container_width=True):
        st.session_state.demo_text = "今天真的太开心了"
    if r1c2.button("😍 喜欢示例", use_container_width=True):
        st.session_state.demo_text = "我很喜欢这个产品"
    if r1c3.button("😠 愤怒示例", use_container_width=True):
        st.session_state.demo_text = "这简直让我气炸了"
    if r1c4.button("😢 悲伤示例", use_container_width=True):
        st.session_state.demo_text = "听到这个消息我很难过"

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    if r2c1.button("😨 恐惧示例", use_container_width=True):
        st.session_state.demo_text = "我有点害怕"
    if r2c2.button("🤢 厌恶示例", use_container_width=True):
        st.session_state.demo_text = "这东西让我很恶心"
    if r2c3.button("😲 惊讶示例", use_container_width=True):
        st.session_state.demo_text = "太突然了，我都惊了"
    if r2c4.button("😐 中性示例", use_container_width=True):
        st.session_state.demo_text = "一般般"

    if st.button("📋 批量示例", use_container_width=True):
        st.session_state.demo_text = (
            "无语了\n"
            "烦死了\n"
            "太差劲了\n"
            "今天真的太开心了\n"
            "我很喜欢这个产品\n"
            "这简直让我气炸了\n"
            "太突然了，我都惊了\n"
            "一般般\n"
            "还行吧"
        )

    user_input = st.text_area(
        "请输入待分析文本",
        value=st.session_state.demo_text,
        height=240,
        placeholder="可输入一句话，也可输入多行文本（每行一句）"
    )

    chars, line_count, mode = get_input_stats(user_input)
    st.markdown(
        f"""
        <span class="info-chip">字符数：{chars}</span>
        <span class="info-chip">行数：{line_count}</span>
        <span class="info-chip">模式：{mode}</span>
        """,
        unsafe_allow_html=True
    )

    run_btn = st.button("开始分析", use_container_width=True)

    st.divider()

    if run_btn:
        if not user_input.strip():
            st.warning("请输入文本后再进行分析。")
        else:
            lines = [line.strip() for line in user_input.split("\n") if line.strip()]

            if len(lines) > 1:
                st.subheader("批量分析结果")

                results = []
                for i, line in enumerate(lines, start=1):
                    result = predict_joint(line, use_bert)
                    results.append({
                        "序号": i,
                        "文本": line,
                        "一级情感": result["coarse_label"],
                        "二级情绪": result["fine_result"],
                        "一级置信度": result["coarse_conf"]["confidence"],
                        "二级置信度": result["fine_conf"]["confidence"]
                    })

                result_df = pd.DataFrame(results)
                st.dataframe(style_result_df(result_df), use_container_width=True, height=430)

                csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="下载批量分析结果 CSV",
                    data=csv_data,
                    file_name="joint_analysis_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            else:
                text = lines[0]
                segments = split_text_for_sentiment(text)

                if len(segments) > 1:
                    st.subheader("长句拆分分析结果")
                    st.caption(f"原始文本：{text}")

                    results = []
                    for i, seg in enumerate(segments, start=1):
                        result = predict_joint(seg, use_bert)
                        results.append({
                            "序号": i,
                            "片段": seg,
                            "一级情感": result["coarse_label"],
                            "二级情绪": result["fine_result"],
                            "一级置信度": result["coarse_conf"]["confidence"],
                            "二级置信度": result["fine_conf"]["confidence"]
                        })

                    segment_df = pd.DataFrame(results)
                    st.dataframe(style_result_df(segment_df), use_container_width=True, height=400)

                    csv_data = segment_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="下载拆分分析结果 CSV",
                        data=csv_data,
                        file_name="split_joint_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                else:
                    result = predict_joint(text, use_bert)

                    st.subheader("结果概览")
                    c1, c2 = st.columns(2)

                    with c1:
                        st.metric("一级情感", result["coarse_label"], f"置信度 {result['coarse_conf']['confidence']:.4f}")
                    with c2:
                        st.metric("二级情绪", result["fine_result"], f"置信度 {result['fine_conf']['confidence']:.4f}")

                    st.caption(f"输入文本：{text}")

                    t1, t2, t3 = st.tabs(["总体结果", "一级情感概率", "二级情绪概率"])

                    with t1:
                        a1, a2 = st.columns(2)
                        with a1:
                            st.metric("一级情感置信度", f"{result['coarse_conf']['confidence']*100:.2f}%")
                            st.progress(float(result["coarse_conf"]["confidence"]))
                        with a2:
                            st.metric("二级情绪置信度", f"{result['fine_conf']['confidence']*100:.2f}%")
                            st.progress(float(result["fine_conf"]["confidence"]))

                    with t2:
                        st.write(f"负面概率：{result['coarse_proba'][0]:.4f}")
                        st.write(f"正面概率：{result['coarse_proba'][1]:.4f}")
                        st.write(f"中性概率：{result['coarse_proba'][2]:.4f}")
                        st.progress(float(result["coarse_proba"][0]), text="负面")
                        st.progress(float(result["coarse_proba"][1]), text="正面")
                        st.progress(float(result["coarse_proba"][2]), text="中性")

                    with t3:
                        for i, prob in enumerate(result["fine_proba"]):
                            st.write(f"{fine_label_map[i]}：{prob:.4f}")
                        st.write(f"七分类原始预测：{result['fine_label_raw']}")
                        for i, prob in enumerate(result["fine_proba"]):
                            st.progress(float(prob), text=fine_label_map[i])

# =============================
# YouTube 评论分析模式
# =============================
else:
    st.subheader("YouTube 评论区氛围分析")
    st.caption("输入 YouTube 视频链接，系统将自动获取评论并分析评论区氛围。")

    youtube_url = st.text_input("请输入 YouTube 视频链接")
    max_comments = st.slider("评论抓取数量", min_value=20, max_value=200, value=100, step=20)

    run_youtube_btn = st.button("开始分析 YouTube 评论", use_container_width=True)

    st.divider()

    if run_youtube_btn:
        if not youtube_url.strip():
            st.warning("请输入 YouTube 视频链接。")
        else:
            try:
                api_key = load_youtube_api_key()
                video_id = extract_video_id(youtube_url)
                comments = fetch_youtube_comments(video_id, api_key, max_comments=max_comments)

                if not comments:
                    st.warning("没有获取到评论，可能视频关闭了评论或评论为空。")
                else:
                    rows = []
                    for c in comments:
                        text = c["text"].strip()
                        if not text:
                            continue

                        result = predict_joint(text, use_bert)

                        rows.append({
                            "comment_id": c["comment_id"],
                            "author": c["author"],
                            "text": text,
                            "一级情感": result["coarse_label"],
                            "二级情绪": result["fine_result"],
                            "一级置信度": round(result["coarse_conf"]["confidence"], 4),
                            "二级置信度": round(result["fine_conf"]["confidence"], 4),
                            "like_count": c["like_count"],
                            "published_at": c["published_at"]
                        })

                    df = pd.DataFrame(rows)

                    st.subheader("评论区氛围统计")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("评论总数", len(df))
                    c2.metric("正面评论数", int((df["一级情感"] == "正面").sum()))
                    c3.metric("负面评论数", int((df["一级情感"] == "负面").sum()))

                    st.write("一级情感分布：")
                    coarse_stats = df["一级情感"].value_counts().rename_axis("一级情感").reset_index(name="数量")
                    st.dataframe(coarse_stats, use_container_width=True)

                    st.write("二级情绪分布：")
                    fine_stats = df["二级情绪"].value_counts().rename_axis("二级情绪").reset_index(name="数量")
                    st.dataframe(fine_stats, use_container_width=True)

                    st.subheader("评论分析结果")
                    st.dataframe(df, use_container_width=True, height=450)

                    csv_data = df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="下载评论分析结果 CSV",
                        data=csv_data,
                        file_name="youtube_comment_sentiment_result.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"分析失败：{e}")