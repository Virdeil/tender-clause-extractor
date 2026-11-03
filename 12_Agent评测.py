# -*- coding: utf-8 -*-
"""
Agent 版评测：同一套 22 题，对比 RAG 与 Agent 的准确率 / 轮数 / 成本

运行前：已跑过 03_建索引.py
"""

import os
import re
import json
import csv
import time

import rag_core
import agent_core


OUT_CSV = "评测结果_Agent.csv"
TRAP_MARKERS = ["未提及", "未给出", "没有", "无法", "不包含", "无相关",
                "不存在", "未提供", "未明确", "未在文件", "未明确给出",
                "没有给出", "未载明", "无具体"]


def extract_pages(text):
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
    pages = {f"f|{v['page']}": v for v in pages.values()}

    print(f"载入 {len(cases)} 题 / {len(chunks)} 句 / {len(pages)} 页")
    bm25 = rag_core.build(chunks)

    # 页码映射：eval_set 记的是 PDF 页码，回答里引的是印刷页码
    page_to_printed = {}
    for _c in chunks:
        page_to_printed[_c["page"]] = _c.get("printed", _c["page"])
    print("开始评测（每题 3-10 秒）\n")

    rows = []
    ans_hit = page_hit = trap_ok = 0
    n_know = sum(1 for c in cases if not c.get("trap"))
    total_in = total_out = 0
    total_steps = 0
    t_start = time.time()

    for c in cases:
        t0 = time.time()
        try:
            text, cost = agent_core.ask(c["q"], chunks, bm25, pages, verbose=False)
        except SystemExit as e:
            print(e)
            break

        pages_in_ans = extract_pages(text)
        total_in += cost["in"]
        total_out += cost["out"]
        total_steps += cost["steps"]

        if c.get("trap"):
            ok_ans = any(m in text for m in TRAP_MARKERS)
            ok_page = None
            if ok_ans:
                trap_ok += 1
        else:
            ok_ans = any(k in text for k in c["kw"]) if c["kw"] else None
            ok_page = (c["page"] in pages_in_ans) if pages_in_ans else False
            if ok_ans:
                ans_hit += 1
            if ok_page:
                page_hit += 1

        mark_a = "√" if ok_ans else ("X" if ok_ans is False else "?")
        mark_p = "-" if ok_page is None else ("√" if ok_page else "X")
        tools = "+".join(cost["tools"]) or "无工具"
        print(f"[{c['id']:>2}] 答案{mark_a} 页码{mark_p} | {cost['steps']}轮 {tools:<18s} | "
              f"{cost['in']:>5} token | {time.time()-t0:.0f}s")

        rows.append({
            "id": c["id"], "问题": c["q"], "标准页": c["page"],
            "标注页": "/".join(map(str, pages_in_ans)),
            "轮数": cost["steps"], "工具": tools,
            "输入token": cost["in"], "输出token": cost["out"],
            "答案命中": mark_a, "页码命中": mark_p,
            "回答": text.replace("\n", " "),
        })

    n = len(rows)
    money = total_in * agent_core.PRICE_IN + total_out * agent_core.PRICE_OUT

    print("\n" + "=" * 70)
    print(f"题目数          {n}")
    print(f"答案命中        {ans_hit}/{n_know} = {ans_hit/max(n_know,1)*100:.1f}%")
    print(f"页码命中        {page_hit}/{n_know} = {page_hit/max(n_know,1)*100:.1f}%")
    print(f"陷阱拒答        {trap_ok}/2")
    print(f"平均轮数        {total_steps/max(n,1):.2f}")
    print(f"总 token        输入 {total_in} / 输出 {total_out}")
    print(f"总费用          {money:.4f} 元（平均每题 {money/max(n,1):.4f} 元）")
    print(f"总耗时          {time.time()-t_start:.0f} 秒")
    print("=" * 70)

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"明细已写入 {OUT_CSV}")


if __name__ == "__main__":
    main()
