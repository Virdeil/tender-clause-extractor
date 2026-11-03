# -*- coding: utf-8 -*-
"""
网页版：招标文件智能问答

启动：
    streamlit run 08_网页版.py

依赖：
    pip install streamlit
"""

import os
import json

import streamlit as st

import rag_core


# ==================== 页面配置 ====================

st.set_page_config(
    page_title="招标文件智能问答",
    page_icon="📄",
    layout="wide",
)


# ==================== 缓存加载（只跑一次）====================

@st.cache_resource(show_spinner="正在加载索引...")
def load_resources():
    chunks = json.load(open("index.json", encoding="utf-8"))
    pages = rag_core.load_pages("pages.json")
    bm25 = rag_core.build(chunks)
    return chunks, pages, bm25


def check_env():
    if not os.environ.get("SF_API_KEY"):
        st.error("未检测到环境变量 SF_API_KEY，请先在系统里设置后再启动。")
        st.stop()
    for f in ("index.json", "pages.json"):
        if not os.path.exists(f):
            st.error(f"缺少 {f}，请先运行 03_建索引.py")
            st.stop()


# ==================== 侧边栏 ====================

check_env()
chunks, pages, bm25 = load_resources()

with st.sidebar:
    st.header("系统信息")
    st.metric("检索单元（句子）", f"{len(chunks)}")
    st.metric("文档页数", f"{len(pages)}")

    st.divider()
    st.subheader("检索流程")
    st.caption(
        "1. 粗排：余弦相似度 + BM25(jieba) → RRF 融合，捞 20 句\n\n"
        "2. 精排：bge-reranker 重打分，留 5 句\n\n"
        "3. 扩展：命中句子 → 所在整页（页码来自元数据）\n\n"
        "4. 生成：只依据资料回答，必须带出处"
    )

    st.divider()
    st.subheader("试试这些问题")
    samples = [
        "投标人需要具备什么资格？",
        "本项目的采购预算是多少？",
        "未在规定时间内提交最后报价，后果是什么？",
        "预付款的支付比例是多少？",
        "本工程的结构设计使用年限是多少年？",
    ]
    for i, s in enumerate(samples):
        if st.button(s, key=f"s{i}", use_container_width=True):
            st.session_state["query"] = s


# ==================== 主界面 ====================

st.title("📄 招标文件智能问答")
st.caption("115 页政府采购磋商文件 ｜ 答案带出处页码 ｜ 无框架手写 RAG")

query = st.text_input(
    "输入你的问题",
    value=st.session_state.get("query", ""),
    placeholder="例如：投标保证金需要交多少？",
)

if query:
    with st.spinner("检索并生成中..."):
        # ① 粗排
        cand = rag_core.retrieve(query, chunks, bm25, k=20)
        # ② 精排
        hits = rag_core.rerank(query, cand, top_n=5)
        # ③ 扩展到整页
        page_hits = rag_core.expand_to_pages(hits, pages)
        # ④ 生成
        answer, ti, to = rag_core.generate(query, page_hits)

    st.markdown("### 回答")
    st.markdown(
        f"<div style='background:#F4F7FB;padding:16px;border-radius:8px;"
        f"border-left:4px solid #1F3A5F'>{answer}</div>",
        unsafe_allow_html=True,
    )

    st.caption(f"token：输入 {ti} / 输出 {to}")

    # 检索过程（面试时展开这个，最能说明你懂）
    with st.expander(f"查看检索过程（粗排 {len(cand)} 句 → 精排 {len(hits)} 句 → 喂入 {len(page_hits)} 页）"):
        st.markdown("**精排后的句子**")
        for i, (c, score) in enumerate(hits, 1):
            st.markdown(
                f"`{i}` **P{c['page']}** · 相关度 `{score:.3f}`  \n"
                f"{c['text'][:120]}"
            )

        st.markdown("**喂给模型的整页**")
        for p, _ in page_hits:
            st.markdown(f"- 第 **{p['page']}** 页（{len(p['text'])} 字）· {p['file']}")
