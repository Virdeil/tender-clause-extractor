# -*- coding: utf-8 -*-
"""
Agent 命令行版：让模型自己决定「要不要查、查什么」

启动：
    python 11_Agent.py

运行前需先跑过 03_建索引.py。

核心逻辑在 agent_core.py（评测脚本 12_Agent评测.py 也用它）。
"""

import os
import json
import time

import rag_core
import agent_core


def main():
    if not os.environ.get("SF_API_KEY"):
        raise SystemExit("没有读到环境变量 SF_API_KEY")
    if not os.path.exists("index.json"):
        print("找不到 index.json，请先运行 03_建索引.py")
        return

    chunks = json.load(open("index.json", encoding="utf-8"))
    pages = rag_core.load_pages("pages.json")
    pages = {f"f|{v['page']}": v for v in pages.values()}

    print(f"载入 {len(chunks)} 句 / {len(pages)} 页")
    bm25 = rag_core.build(chunks)
    print("就绪。\n")

    while True:
        q = input("问题（回车退出）：").strip()
        if not q:
            break

        t0 = time.time()
        print()
        ans, cost = agent_core.ask(q, chunks, bm25, pages, verbose=True)

        print("\n回答：")
        for line in ans.strip().splitlines():
            print("  " + line)

        money = agent_core.estimate_cost(cost)
        print(f"\n  ── 成本报告 " + "─" * 40)
        print(f"     轮数 {cost['steps']}　工具 {cost['tools']}　"
              f"输入 {cost['in']} / 输出 {cost['out']} token")
        print(f"     估算 {money:.4f} 元　预算占用 "
              f"{(cost['in'] + cost['out']) / agent_core.MAX_TOTAL_TOKENS * 100:.0f}%")
        print(f"     用时 {time.time() - t0:.1f} 秒")
        print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
