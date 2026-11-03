# -*- coding: utf-8 -*-
"""
手写 RAG 第三步：混合检索（向量 + BM25，用 RRF 融合）

两次修正记录：
  1) 2-gram 分词 → 改用 jieba。2-gram 会产生「要具」这类跨词垃圾组合，
     IDF 极高，制造假匹配（实测把违约金条款顶到第 1 名）。
  2) 归一化加权 → 改用 RRF。两种检索的分数量纲不可比，一个离群值就能翻盘；
     RRF 只看排名，对量纲免疫。

依赖：pip install jieba
"""

import os
import re
import json
import math
from collections import Counter

import requests

try:
    import jieba
    jieba.setLogLevel(20)
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    print("警告：未安装 jieba，退回 2-gram（效果较差）")
    print("      pip install jieba -i https://pypi.tuna.tsinghua.edu.cn/simple\n")


# ==================== 配置 ====================

API_KEY = os.environ.get("SF_API_KEY", "")
if not API_KEY:
    raise SystemExit("没有读到环境变量 SF_API_KEY")

EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "BAAI/bge-m3"

TOP_K = 3
K_RRF = 60        # RRF 平滑常数，60 是论文里的经验值
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

def embed_one(text):
    r = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ==================== RRF 融合 ====================

def rank_of(scores):
    """返回 {文档下标: 名次}，名次从 1 开始。"""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return {i: r + 1 for r, i in enumerate(order)}


def rrf_fuse(vec_scores, bm_scores, k=K_RRF):
    """倒数排名融合：score = 1/(k+向量名次) + 1/(k+BM25名次)

    只看名次、不看分数，所以两种检索的量纲差异完全不影响结果。
    """
    rv = rank_of(vec_scores)
    rb = rank_of(bm_scores)
    return [1.0 / (k + rv[i]) + 1.0 / (k + rb[i]) for i in range(len(vec_scores))]


# ==================== 主流程 ====================

def main():
    if not os.path.exists("index.json"):
        print("找不到 index.json，请先运行 03_建索引.py")
        return

    chunks = json.load(open("index.json", encoding="utf-8"))
    print(f"已加载 {len(chunks)} 块索引")

    print("构建 BM25 索引（分词中）...")
    docs_tokens = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25(docs_tokens)
    print(f"   完成。平均块长 {bm25.avgdl:.0f} 词，词表 {len(bm25.idf)} 个\n")

    while True:
        query = input("问题（直接回车退出）：").strip()
        if not query:
            break

        qt = tokenize(query)
        qv = embed_one(query)

        vec_scores = [cosine(qv, c["vector"]) for c in chunks]
        bm_scores = bm25.scores(qt)
        fused = rrf_fuse(vec_scores, bm_scores)

        rv, rb = rank_of(vec_scores), rank_of(bm_scores)

        print("=" * 74)
        print(f"问题：{query}")
        print(f"分词：{' / '.join(qt)}\n")

        print(f"  【融合结果（RRF）】")
        for rank, i in enumerate(sorted(range(len(chunks)), key=lambda i: -fused[i])[:TOP_K], 1):
            c = chunks[i]
            print(f"   {rank}. {c['file'][:14]:<16} P{c['page']:<4} "
                  f"RRF={fused[i]:.5f}  (向量#{rv[i]}  BM25#{rb[i]})")
            print(f"      {c['text'][:56].replace(chr(10), ' ')}")
        print()

        print(f"  【纯向量前 {TOP_K}】")
        for rank, i in enumerate(sorted(range(len(chunks)), key=lambda i: -vec_scores[i])[:TOP_K], 1):
            print(f"   {rank}. P{chunks[i]['page']:<4} {vec_scores[i]:.4f}")
        print()

        print(f"  【纯 BM25 前 {TOP_K}】")
        for rank, i in enumerate(sorted(range(len(chunks)), key=lambda i: -bm_scores[i])[:TOP_K], 1):
            print(f"   {rank}. P{chunks[i]['page']:<4} {bm_scores[i]:.2f}")
        print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
