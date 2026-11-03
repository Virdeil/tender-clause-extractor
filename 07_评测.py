# -*- coding: utf-8 -*-
"""
自动化评测：22 题跑一遍，统计答案命中率、页码命中率、拒答正确率

评什么：
  答案命中  — 回答里是否出现预期关键词（粗筛，最终仍需人看）
  页码命中  — 回答里标注的出处，是否包含标准页
  拒答正确  — 陷阱题（文件里没有）是否回答"未提及"

运行前：已跑过 03_建索引.py，且 eval_set.json 存在。
"""

import os
import re
import json
import csv
import time

import rag_core


RETRIEVE_K = 20    # 粗排召回多少句
RERANK_K = 5       # 精排保留多少句
OUT_CSV = "评测结果.csv"
TRAP_MARKERS = ["未提及", "未给出", "没有", "无法", "不包含", "无相关", "不存在", "未提供", "未明确"]


def extract_pages(text):
    """从回答里提取所有「第 X 页」中的页码。"""
    return [int(n) for n in re.findall(r"第\s*(\d+)\s*页", text)]


def main():
    if not os.environ.get("SF_API_KEY"):
        raise SystemExit("没有读到环境变量 SF_API_KEY")
    if not os.path.exists("index.json"):
        print("找不到 index.json，请先运行 03_建索引.py")
        return

    cases = json.load(open("eval_set.json", encoding="utf-8"))
    chunks = json.load(open("index.json", encoding="utf-8"))
    pages = rag_core.load_pages("pages.json")

    print(f"载入 {len(cases)} 道题、{len(chunks)} 个句子块、{len(pages)} 页")
    print("构建 BM25...")
    bm25 = rag_core.build(chunks)
    print("开始评测（每题约 5-15 秒）\n")

    rows = []
    ans_hit = page_hit = trap_ok = 0
    n_knowledge = sum(1 for c in cases if not c.get("trap"))
    t_start = time.time()

    for c in cases:
        t0 = time.time()
        cand = rag_core.retrieve(c["q"], chunks, bm25, k=RETRIEVE_K)
        hits = rag_core.rerank(c["q"], cand, top_n=RERANK_K)
        page_hits = rag_core.expand_to_pages(hits, pages)
        try:
            text, _, _ = rag_core.generate(c["q"], page_hits)
        except SystemExit as e:
            print(e)
            break

        pages_in_answer = extract_pages(text)
        got_pages = [p["page"] for p, _ in page_hits]

        if c.get("trap"):
            ok_ans = any(m in text for m in TRAP_MARKERS)
            ok_page = None
            if ok_ans:
                trap_ok += 1
        else:
            ok_ans = any(k in text for k in c["kw"]) if c["kw"] else None
            ok_page = (c["page"] in pages_in_answer) if pages_in_answer else False
            if ok_ans:
                ans_hit += 1
            if ok_page:
                page_hit += 1

        mark_a = "✓" if ok_ans else ("✗" if ok_ans is False else "?")
        mark_p = "-" if ok_page is None else ("✓" if ok_page else "✗")
        print(f"[{c['id']:>2}] 答案{mark_a} 页码{mark_p} | "
              f"检索页{got_pages} 标注页{pages_in_answer} ({time.time()-t0:.0f}s)")

        rows.append({
            "id": c["id"],
            "问题": c["q"],
            "标准页": c["page"],
            "检索页": "/".join(map(str, got_pages)),
            "标注页": "/".join(map(str, pages_in_answer)),
            "答案命中": mark_a,
            "页码命中": mark_p,
            "回答": text.replace("\n", " "),
        })

    n = len(rows)
    print("\n" + "=" * 62)
    print(f"题目数            {n}")
    print(f"答案命中          {ans_hit}/{n_knowledge} = {ans_hit / max(n_knowledge,1) * 100:.1f}%")
    print(f"页码命中          {page_hit}/{n_knowledge} = {page_hit / max(n_knowledge,1) * 100:.1f}%")
    print(f"陷阱拒答正确      {trap_ok}/2")
    print(f"总耗时            {time.time() - t_start:.0f} 秒")
    print("=" * 62)

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"明细已写入 {OUT_CSV}")


if __name__ == "__main__":
    main()
