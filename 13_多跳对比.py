# -*- coding: utf-8 -*-
"""
多跳题对比评测：同一批题，分别用 RAG（固定流程）和 Agent（自主流程）回答

目的：验证「什么题型上 Agent 才有价值」。
  RAG   —— 固定：检索一次 → 喂 5 句所在整页 → 生成
  Agent —— 自主：自己决定查不查、查几次、要不要读整页

运行前：已跑过 03_建索引.py
"""

import os
import re
import json
import time

import rag_core
import agent_core


TRAP_MARKERS = ["未提及", "未给出", "没有", "无法", "无相关", "未提供", "未明确"]


def extract_pages(text):
    return [int(n) for n in re.findall(r"第\s*(\d+)\s*页", text)]


def rag_answer(q, chunks, bm25, pages):
    """固定流程版"""
    cand = rag_core.retrieve(q, chunks, bm25, k=20)
    hits = rag_core.rerank(q, cand, top_n=5)
    page_hits = rag_core.expand_to_pages(hits, pages)
    text, ti, to = rag_core.generate(q, page_hits)
    return text, {"in": ti, "out": to}


def main():
    if not os.environ.get("SF_API_KEY"):
        raise SystemExit("没有读到环境变量 SF_API_KEY")

    cases = json.load(open("eval_multihop.json", encoding="utf-8"))
    chunks = json.load(open("index.json", encoding="utf-8"))
    pages_raw = rag_core.load_pages("pages.json")
    pages = {f"f|{v['page']}": v for v in pages_raw.values()}
    bm25 = rag_core.build(chunks)

    print(f"{len(cases)} 道多跳题 / {len(chunks)} 句索引\n")

    rag_score = agent_score = 0
    rag_cost = agent_cost = 0.0
    rag_steps = agent_steps = 0

    for c in cases:
        print("=" * 78)
        print(f"[{c['id']}] {c['q']}")
        print(f"      设计意图：{c['why']}")
        print(f"      标准页：P{c['page']}")

        # ---------- RAG ----------
        t0 = time.time()
        r_text, r_cost = rag_answer(c["q"], chunks, bm25, pages)
        r_pages = extract_pages(r_text)
        r_ok = any(k in r_text for k in c["kw"])
        r_page_ok = c["page"] in r_pages
        rag_score += 1 if (r_ok and r_page_ok) else 0
        r_money = r_cost["in"] * 0.000002 + r_cost["out"] * 0.000008
        rag_cost += r_money
        rag_steps += 1

        print(f"\n  ── RAG（固定流程，1 次检索）  {time.time()-t0:.0f}s")
        print(f"     引用页 {r_pages} ｜ 关键词 {'√' if r_ok else 'X'} ｜ 页码 {'√' if r_page_ok else 'X'}")
        print(f"     {r_text.strip()[:400].replace(chr(10), ' ')}")

        # ---------- Agent ----------
        t0 = time.time()
        a_text, a_cost = agent_core.ask(c["q"], chunks, bm25, pages, verbose=False)
        a_pages = extract_pages(a_text)
        a_ok = any(k in a_text for k in c["kw"])
        a_page_ok = c["page"] in a_pages
        agent_score += 1 if (a_ok and a_page_ok) else 0
        a_money = agent_core.estimate_cost(a_cost)
        agent_cost += a_money
        agent_steps += a_cost["steps"]

        print(f"\n  ── Agent（自主流程，{a_cost['steps']} 轮，工具 {a_cost['tools']}）  {time.time()-t0:.0f}s")
        print(f"     引用页 {a_pages} ｜ 关键词 {'√' if a_ok else 'X'} ｜ 页码 {'√' if a_page_ok else 'X'}")
        print(f"     {a_text.strip()[:400].replace(chr(10), ' ')}")
        print()

    n = len(cases)
    print("=" * 78)
    print("汇总（判定标准：关键词命中 且 页码命中）")
    print(f"  RAG    答对 {rag_score}/{n} ｜ 平均轮数 {rag_steps/n:.2f} ｜ 总费用 {rag_cost:.4f} 元")
    print(f"  Agent  答对 {agent_score}/{n} ｜ 平均轮数 {agent_steps/n:.2f} ｜ 总费用 {agent_cost:.4f} 元")


if __name__ == "__main__":
    main()
