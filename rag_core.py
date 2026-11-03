# -*- coding: utf-8 -*-
"""
RAG 核心模块：分词、BM25、向量、RRF 融合、检索、生成

被 06_RAG问答.py 和 07_评测.py 共同引用。
把检索和生成都抽出来的理由：问答、评测都要用，不该复制三遍。
"""

import os
import re
import math
from collections import Counter

import requests

try:
    import jieba
    jieba.setLogLevel(20)
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

API_KEY = os.environ.get("SF_API_KEY", "")
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "BAAI/bge-m3"

RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

LLM_URL = "https://api.siliconflow.cn/v1/chat/completions"
LLM_MODEL = "deepseek-ai/DeepSeek-V3"

K_RRF = 60
BM25_K1 = 1.5
BM25_B = 0.75

STOP = set("的了和与及或在为是有无不将对其这那等以对从被把而但很更重要需要应当可以本项目".split())


# ==================== 分词 ====================

def tokenize(text):
    if HAS_JIEBA:
        return [w for w in jieba.lcut(text)
                if len(w) >= 2
                and re.match(r"^[\u4e00-\u9fa5A-Za-z0-9]+$", w)
                and w not in STOP]
    s = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", text)
    return [s[i:i + 2] for i in range(len(s) - 1)] if len(s) >= 2 else []


# ==================== 文本切分 ====================

def split_sentences(text, max_len=90, min_len=12):
    """把一段正文切成句子级小块，用于检索。"""
    raw = re.split(r"(?<=[。；！？])|\n", text)
    out = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        if len(s) <= max_len:
            out.append(s)
            continue
        buf = ""
        for piece in re.split(r"(?<=[，、,])", s):
            if len(buf) + len(piece) <= max_len:
                buf += piece
            else:
                if buf:
                    out.append(buf.strip())
                buf = piece
        if buf.strip():
            out.append(buf.strip())
    return [x for x in out if len(x) >= min_len]


# ==================== BM25 ====================

class BM25:
    def __init__(self, docs_tokens):
        self.docs = docs_tokens
        self.N = len(docs_tokens)
        self.avgdl = sum(len(d) for d in docs_tokens) / max(self.N, 1)
        self.tf = [Counter(d) for d in docs_tokens]

        df = Counter()
        for f in self.tf:
            for t in f:
                df[t] += 1
        self.idf = {t: math.log(1 + (self.N - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query_tokens):
        out = []
        for i in range(self.N):
            s = 0.0
            dl = len(self.docs[i])
            for t in query_tokens:
                f = self.tf[i].get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * (f * (BM25_K1 + 1)) / (
                        f + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avgdl))
            out.append(s)
        return out


# ==================== 向量 ====================

def embed_texts(texts):
    r = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()["data"]
    data.sort(key=lambda d: d["index"])
    return [d["embedding"] for d in data]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ==================== RRF 融合 ====================

def rank_of(scores):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return {i: r + 1 for r, i in enumerate(order)}


def rrf_fuse(vec_scores, bm_scores, k=K_RRF):
    rv, rb = rank_of(vec_scores), rank_of(bm_scores)
    return [1.0 / (k + rv[i]) + 1.0 / (k + rb[i]) for i in range(len(vec_scores))]


# ==================== 对外接口 ====================

def build(chunks):
    """建 BM25 索引。chunks 需含 text 字段。"""
    return BM25([tokenize(c["text"]) for c in chunks])


def retrieve(query, chunks, bm25, k=3):
    """混合检索，返回 [(chunk, rrf分数), ...]，按相关度降序。"""
    qv = embed_texts([query])[0]
    vec_scores = [cosine(qv, c["vector"]) for c in chunks]
    bm_scores = bm25.scores(tokenize(query))
    fused = rrf_fuse(vec_scores, bm_scores)
    top = sorted(range(len(chunks)), key=lambda i: -fused[i])[:k]
    return [(chunks[i], fused[i]) for i in top]


# ==================== Rerank（精排）====================

def rerank(query, candidates, top_n=5):
    """对候选句子重新打分。

    向量检索是「双塔」：query 和文档各编码各的，快但粗。
    Rerank 是「交互式」：query 和文档一起进模型，慢但准。
    所以两阶段：粗排捞 20 条 → 精排取 5 条。

    candidates: [(chunk, score), ...]
    """
    if not candidates:
        return []

    docs = [c["text"] for c, _ in candidates]

    resp = requests.post(
        RERANK_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": RERANK_MODEL, "query": query, "documents": docs, "top_n": top_n},
        timeout=60,
    )
    if resp.status_code in (401, 402, 403, 404):
        # rerank 不可用时退回原顺序，保证主流程不中断
        print(f"    [rerank 不可用 {resp.status_code}]，退回粗排结果")
        return candidates[:top_n]
    resp.raise_for_status()

    results = resp.json()["results"]
    results.sort(key=lambda r: -r["relevance_score"])
    return [(candidates[r["index"]][0], r["relevance_score"]) for r in results]


# ==================== Small-to-Big ====================

def load_pages(path="pages.json"):
    """载入整页文本。"""
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def expand_to_pages(hits, pages, max_pages=3):
    """把命中的句子扩展成它们所在的整页，按页去重。

    检索用句子（准），生成用整页（全）—— 这就是 small-to-big。
    页码依然是元数据，不会漂。
    """
    seen, out = set(), []
    for c, score in hits:
        key = f"{c['file']}|{c['page']}"
        if key in seen:
            continue
        seen.add(key)
        p = pages.get(key)
        if p:
            out.append((p, score))
        if len(out) >= max_pages:
            break
    return out


# ==================== 生成 ====================

SYSTEM_PROMPT = """你是招投标文件助手。只依据用户提供的【资料】回答问题。

规则：
1. 只使用资料中的信息，禁止用常识或推测补充
2. 每条结论后必须标注出处，格式：（依据第 3 页）
3. 页码只能来自资料中标注的页号，不得改写、推算
4. 资料中没有相关信息时，只回答："文件中未提及。"

5. 不同文件对同一概念的叫法不同，下列视为同一概念，按此理解：
   · 保证金：投标保证金 / 磋商保证金 / 谈判保证金 / 响应保证金
   · 有效期：投标有效期 / 磋商有效期 / 响应文件有效期
   · 工期：工期 / 服务期 / 合同履行期限 / 交货期 / 交付期
   · 限价：最高限价 / 最高投标限价 / 招标控制价 / 预算金额
   · 资质：资质 / 资格要求 / 特定资格条件

6. 若资料中只有原则性表述、没有具体数值或结论，回答"文件中未给出具体内容"，
   不要拿原则性条款硬凑答案（例如问"多少年"而资料只写"合理使用年限"，属未给出）

7. 注意否定式答案：资料中写"不收取 / 不再收取 / 无需缴纳 / 不要求"时，
   这是明确的答案，必须据此回答（例如回答"本项目不收取投标保证金"），
   不得回答"未提及"。
5. 先给一句结论，再给依据原文，总长不超过 150 字"""


def build_context(hits):
    """把检索结果拼成带页码的资料块。页码在这里被显式写进提示词。"""
    parts = []
    for i, (c, _score) in enumerate(hits, 1):
        parts.append(f"【资料{i}｜第 {c['page']} 页】\n{c['text']}")
    return "\n\n".join(parts)


def generate(query, hits):
    """根据检索结果生成回答。返回 (文本, prompt_tokens, completion_tokens)。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【资料】\n{build_context(hits)}\n\n【问题】\n{query}"},
    ]
    resp = requests.post(
        LLM_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": LLM_MODEL, "messages": messages, "temperature": 0},
        timeout=120,
    )
    if resp.status_code in (401, 402, 403, 404):
        raise SystemExit(f"[错误 {resp.status_code}] 配置或余额问题，停止")
    resp.raise_for_status()

    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
