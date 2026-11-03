# -*- coding: utf-8 -*-
"""
手写 RAG 第二步：检索
给一个问题，从 index.json 里找出最相关的 3 块，并显示页码。

这一步不调用大模型——只用数学（余弦相似度）找「意思最接近」的段落。
"""

import os
import json
import math

import requests


# ==================== 配置 ====================

API_KEY = os.environ.get("SF_API_KEY", "")
if not API_KEY:
    raise SystemExit("没有读到环境变量 SF_API_KEY")

EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "BAAI/bge-m3"

TOP_K = 3        # 取最相关的几块


# ==================== 工具函数 ====================

def embed_one(text):
    """把一句话变成向量。"""
    resp = requests.post(
        EMBED_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def cosine(a, b):
    """余弦相似度：两个向量夹角的余弦值，越接近 1 越相似。

    公式：点积 / (长度a × 长度b)
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def search(query, chunks, k=TOP_K):
    """把问题变成向量，和每一块算相似度，返回前 k 名。"""
    qv = embed_one(query)

    scored = []
    for c in chunks:
        s = cosine(qv, c["vector"])
        scored.append((s, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


# ==================== 主流程 ====================

def main():
    # 1. 读索引
    if not os.path.exists("index.json"):
        print("找不到 index.json，请先运行 03_建索引.py")
        return

    with open("index.json", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"已加载 {len(chunks)} 块索引\n")

    # 2. 循环提问
    while True:
        query = input("问题（直接回车退出）：").strip()
        if not query:
            break

        results = search(query, chunks)

        print(f"\n{'=' * 60}")
        for rank, (score, c) in enumerate(results, 1):
            print(f"[{rank}] 相似度 {score:.4f} ｜ {c['file']} ｜ 第 {c['page']} 页")
            body = c["text"].replace("\n", " ")
            print(f"    {body[:120]}...")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
