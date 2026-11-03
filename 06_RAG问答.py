# -*- coding: utf-8 -*-
"""
完整 RAG 问答：检索 → 拼上下文 → 大模型生成 → 带出处回答

和扣子版的区别：
  页码来自切分时记录的元数据，不是模型"说"的 → 出处可靠

运行前需先跑 03_建索引.py。
"""

import os
import json

import rag_core


RETRIEVE_K = 20    # 粗排召回多少句
RERANK_K = 5       # 精排保留多少句


def main():
    if not os.environ.get("SF_API_KEY"):
        raise SystemExit("没有读到环境变量 SF_API_KEY")

    if not os.path.exists("index.json"):
        print("找不到 index.json，请先运行 03_建索引.py")
        return

    chunks = json.load(open("index.json", encoding="utf-8"))
    pages = rag_core.load_pages("pages.json")
    print(f"已加载 {len(chunks)} 个句子块、{len(pages)} 页全文")
    print("构建 BM25 索引（分词中）...")
    bm25 = rag_core.build(chunks)
    print("就绪。\n")

    while True:
        query = input("问题（直接回车退出）：").strip()
        if not query:
            break

        # ① 检索：用句子找（准）
        cand = rag_core.retrieve(query, chunks, bm25, k=RETRIEVE_K)
        hits = rag_core.rerank(query, cand, top_n=RERANK_K)

        # ② 扩展：取命中句子所在的整页（全）
        page_hits = rag_core.expand_to_pages(hits, pages)

        print("\n" + "=" * 74)
        print(f"粗排 {len(cand)} 句 → 精排 {len(hits)} 句：")
        for i, (c, score) in enumerate(hits, 1):
            print(f"  {i}. P{c['page']:<4} {c['text'][:48]}")

        print(f"\n喂给模型的整页：{[p['page'] for p, _ in page_hits]}")

        # ③ 生成
        text, ti, to = rag_core.generate(query, page_hits)

        print("\n回答：")
        for line in text.strip().splitlines():
            print("  " + line)

        print(f"\n  （token 输入 {ti} / 输出 {to}）")
        print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
