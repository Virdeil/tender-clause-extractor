# -*- coding: utf-8 -*-
"""
网页版（上传任意招标文件）：上传 PDF/TXT → 原地建索引 → 问答

启动：
    streamlit run 09_上传版.py

依赖：
    pip install streamlit pypdf
"""

import io
import os
import re
import time

import streamlit as st
import pypdf

import rag_core


st.set_page_config(page_title="招标文件智能问答", page_icon="📄", layout="wide")

MAX_PAGES_WARN = 200      # 超过这个页数提示会慢
MIN_SENTENCES = 20        # 少于这个句子数，认为提取不完整
MIN_TEXT_CHARS = 100      # 全文少于这么多字，判为扫描件，回退到视觉 OCR
MAX_OCR_PAGES = 40        # 视觉 OCR 最多处理多少页（按页计费，防止烧钱）

VLM_URL = "https://api.siliconflow.cn/v1/chat/completions"
VLM_MODEL = "Qwen/Qwen3-VL-32B-Instruct"

OCR_PROMPT = (
    "请逐字转录这一页的全部文字。\n"
    "如果是表格，请按「条目：内容」逐行输出，保留数字、单位、百分比和勾选项。\n"
    "不要总结，不要解释，不要添加原文没有的内容。"
)


# ==================== 从上传文件解析出「页」====================

def clean_text(text):
    """清洗 PDF 提取的文本：去哈希串、去所有空白。"""
    text = re.sub(r"[0-9a-zA-Z]{16,}", " ", text)
    return re.sub(r"\s+", "", text)


def parse_pages(file_bytes, filename):
    """返回 [(页码, 该页正文), ...]

    PDF 天然分页 → 页码就是元数据，不靠模型猜。
    TXT 没有页的概念 → 整篇作为 1 页。
    """
    name = filename.lower()

    if name.endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        out = []
        for i, page in enumerate(reader.pages, 1):
            body = clean_text(page.extract_text() or "")
            if body:
                out.append((i, body))
        return out

    if name.endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        body = "".join(lines)
        return [(1, body)] if body else []

    return []


def ocr_page(image_bytes):
    """用视觉模型把一页图片转录成文字（扫描件的兜底方案）。"""
    import base64
    import requests

    b64 = base64.b64encode(image_bytes).decode()
    resp = requests.post(
        VLM_URL,
        headers={"Authorization": f"Bearer {os.environ.get('SF_API_KEY', '')}",
                 "Content-Type": "application/json"},
        json={
            "model": VLM_MODEL,
            "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": OCR_PROMPT},
            ]}],
        },
        timeout=180,
    )
    if resp.status_code != 200:
        return ""
    return resp.json()["choices"][0]["message"]["content"]


def ocr_pdf(file_bytes, progress=None, max_pages=MAX_OCR_PAGES):
    """把扫描件 PDF 逐页渲染成图片，再逐页转录成文字。

    返回 [(页码, 正文), ...] —— 结构和 parse_pages 一致，
    所以下游的切分、向量、检索、页码引用全都不用改。
    """
    import fitz

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    n = min(len(doc), max_pages)
    out = []

    for i in range(n):
        pix = doc[i].get_pixmap(dpi=150)
        img = pix.tobytes("jpeg", jpg_quality=85)
        text = ocr_page(img)
        body = clean_text(text)
        if body:
            out.append((i + 1, body))
        if progress:
            progress((i + 1) / n, f"视觉识别第 {i + 1}/{n} 页")

    doc.close()
    return out


def build_index(pages, offset=0):
    """页 → 句子块（供检索）＋ 整页（供生成）"""
    sentences, page_map = [], {}

    for pno, body in pages:
        printed = pno + offset
        page_map[f"upload|{pno}"] = {"file": "upload", "page": pno, "printed": printed, "text": body}
        for s in rag_core.split_sentences(body):
            sentences.append({"file": "upload", "page": pno, "printed": printed, "text": s})

    return sentences, page_map


# ==================== 页面 ====================

st.title("📄 招标文件智能问答")
st.caption("上传任意一份招标文件（PDF / TXT），现场建索引后即可提问，答案带出处页码")

if not os.environ.get("SF_API_KEY"):
    st.error("未检测到环境变量 SF_API_KEY，请先设置后再启动。")
    st.stop()

with st.sidebar:
    st.header("① 上传文件")
    up = st.file_uploader("招标文件（PDF / TXT）", type=["pdf", "txt"])

    st.divider()
    st.subheader("检索流程")
    st.caption(
        "1. 粗排：余弦 + BM25(jieba) → RRF，捞 20 句\n\n"
        "2. 精排：bge-reranker 重打分，留 5 句\n\n"
        "3. 扩展：命中句子 → 所在整页\n\n"
        "4. 生成：只依据资料回答，必须带出处"
    )
    st.divider()
    st.caption("提示：50 页约 30 秒，115 页约 60 秒。")


if up is None:
    st.info("👈 先在左侧上传一份招标文件")
    st.stop()


# ==================== 建索引（按文件内容缓存）====================

@st.cache_resource(show_spinner=False)
def prepare(file_bytes, filename):
    """返回 (sentences, page_map, bm25, n_pages, n_chars, cost, method)。

    method: "text"（直接提取）或 "vision"（扫描件，用视觉模型 OCR）
    提取失败时 sentences 为 None，不浪费一次向量化调用。
    """
    t0 = time.time()

    pages = parse_pages(file_bytes, filename)
    n_chars = sum(len(b) for _, b in pages)

    # 文字太少 → 判为扫描件 → 回退到视觉 OCR
    method = "text"
    if n_chars < MIN_TEXT_CHARS and filename.lower().endswith(".pdf"):
        method = "vision"
        bar = st.progress(0.0, text="扫描件：启动视觉识别...")
        pages = ocr_pdf(file_bytes, progress=lambda r, t: bar.progress(min(r, 1.0), text=t))
        bar.empty()
        n_chars = sum(len(b) for _, b in pages)

    if not pages:
        return None, None, None, 0, 0, 0.0, method

    # 测页码偏移（页脚是图片，只能视觉读）；失败则按 0 处理
    offset = 0
    if filename.lower().endswith(".pdf") and len(pages) > 5:
        offset = rag_core.detect_page_offset(file_bytes)

    sentences, page_map = build_index(pages, offset)
    if not sentences:
        return None, None, None, len(pages), n_chars, 0.0, method

    bar = st.progress(0.0, text="计算向量...")
    BATCH = 16
    for i in range(0, len(sentences), BATCH):
        batch = sentences[i:i + BATCH]
        vecs = rag_core.embed_texts([s["text"] for s in batch])
        for s, v in zip(batch, vecs):
            s["vector"] = v
        bar.progress(min((i + BATCH) / len(sentences), 1.0))
    bar.empty()

    bm25 = rag_core.build(sentences)
    return sentences, page_map, bm25, len(pages), n_chars, time.time() - t0, method


with st.spinner("解析文件..."):
    sentences, page_map, bm25, n_pages, n_chars, cost, method = prepare(up.getvalue(), up.name)


# ==================== 提取结果自检（产品细节）====================

if n_pages == 0:
    st.error("**这个文件读不出任何内容。**")
    st.markdown(
        "已经试过两条路：\n\n"
        "- **文本提取**：结果为空\n"
        "- **视觉识别**（Qwen3-VL）：也没识别出文字\n\n"
        "常见原因：PDF 加密、图片分辨率过低、或根本不是文档。\n"
        "建议：用 WPS / Adobe 打开看一眼能不能正常显示。"
    )
    st.stop()

if sentences is None:
    st.error(f"提取到 {n_pages} 页，但没切出有效句子。文件可能几乎全是图片或表格。")
    st.stop()

if n_pages > MAX_PAGES_WARN:
    st.warning(f"⚠️ 这份文件有 {n_pages} 页，建索引会比较久（{cost:.0f} 秒已用）。")

if len(sentences) < MIN_SENTENCES:
    st.warning(
        f"⚠️ 只切出 {len(sentences)} 个句子块（{n_pages} 页 / {n_chars} 字），"
        f"内容偏少或提取不完整，问答效果可能不佳。"
    )

if method == "vision":
    st.info(f"🖼 这是一份**扫描件**（文本提取不到内容），已用视觉模型逐页识别。"
            f"识别页数上限 {MAX_OCR_PAGES} 页。")

st.success(
    f"已建索引：**{n_pages}** 页 · **{n_chars:,}** 字 → **{len(sentences)}** 个句子块，"
    f"耗时 **{cost:.1f}** 秒" + ("（视觉识别）" if method == "vision" else "")
)


# ==================== 问答 ====================

query = st.text_input("输入你的问题", placeholder="例如：投标保证金需要交多少？")

if query:
    with st.spinner("检索并生成中..."):
        cand = rag_core.retrieve(query, sentences, bm25, k=20)
        hits = rag_core.rerank(query, cand, top_n=5)
        page_hits = rag_core.expand_to_pages(hits, page_map)
        answer, ti, to = rag_core.generate(query, page_hits)

    st.markdown("### 回答")
    st.markdown(
        f"<div style='background:#F4F7FB;padding:16px;border-radius:8px;"
        f"border-left:4px solid #1F3A5F'>{answer}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"token：输入 {ti} / 输出 {to}")

    with st.expander(
        f"查看检索过程（粗排 {len(cand)} 句 → 精排 {len(hits)} 句 → 喂入 {len(page_hits)} 页）"
    ):
        st.markdown("**精排后的句子**")
        for i, (c, score) in enumerate(hits, 1):
            st.markdown(
                f"`{i}` **P{c['page']}** · 相关度 `{score:.3f}`  \n{c['text'][:120]}"
            )
        st.markdown("**喂给模型的整页**")
        for p, _ in page_hits:
            st.markdown(f"- 第 **{p['page']}** 页（{len(p['text'])} 字）")
