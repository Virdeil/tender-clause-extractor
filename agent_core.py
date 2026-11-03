# -*- coding: utf-8 -*-
"""
Agent 核心：工具定义、工具执行、自主循环

被 11_Agent.py（命令行）和 12_Agent评测.py（批量评测）共同引用。
和 rag_core 的分工：
  rag_core   → 检索/生成的基础能力（固定流程）
  agent_core → 让模型自己决定何时调用这些能力（自主流程）
"""

import os
import json

import requests

import rag_core

API_KEY = os.environ.get("SF_API_KEY", "")
CHAT_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V3"

MAX_STEPS = 6
MAX_TOTAL_TOKENS = 15000
PRICE_IN = 2.0 / 1_000_000
PRICE_OUT = 8.0 / 1_000_000


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "在招标文件中检索相关条款。返回若干命中句子及页码。"
                           "当你不知道该看哪几页时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词。用 2-4 个词组，不要用完整句子、"
                                       "不要带「本项目」「请问」这类前缀。"
                                       "例如：预算 最高限价；保证金；资格 资质",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "读取指定页码的全文。当你知道在哪一页、需要看细节时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "description": "页码，例如 5"}
                },
                "required": ["page"],
            },
        },
    },
]


SYSTEM = """你是招投标文件助手。你可以调用工具查阅一份 115 页的政府采购磋商文件。

工作方式：
1. 先判断这个问题需不需要查文件。闲聊类问题直接回答；事实类问题必须先查。
2. 不确定在哪一页时，先用 search 检索；需要看细节时，用 read_page 读整页。
3. 可以多次调用工具，直到有足够信息。
4. 回答必须标注出处页码，格式：（依据第 5 页）。
5. 文件里确实没有的，回答"文件中未提及"。
6. 若资料中只有原则性表述、没有具体数值，回答"文件中未给出具体内容"。
7. 注意否定式答案：资料写"不收取 / 不再收取 / 无需缴纳"时，这是明确答案，不是"未提及"。
8. 不要凭常识回答文件内容相关的问题。"""


def do_search(query, chunks, bm25):
    """返回命中的句子 + 页码（精而短）。整页内容留给 read_page。"""
    cand = rag_core.retrieve(query, chunks, bm25, k=20)
    hits = rag_core.rerank(query, cand, top_n=6)
    if not hits:
        return "没有检索到相关内容。"
    lines = [f"共命中 {len(hits)} 条（需要细节可用 read_page）："]
    for c, score in hits:
        lines.append(f"· 第 {c['page']} 页（相关度 {score:.2f}）：{c['text']}")
    return "\n".join(lines)


def do_read_page(page_no, pages):
    p = pages.get(f"f|{page_no}") or pages.get(f"upload|{page_no}")
    if p:
        return f"【第 {page_no} 页】{p['text']}"
    for v in pages.values():
        if v["page"] == page_no:
            return f"【第 {page_no} 页】{v['text']}"
    return f"没有第 {page_no} 页。"


def ask(question, chunks, bm25, pages, verbose=False):
    """自主问答。返回 (回答文本, 成本字典)。

    成本字典含：in / out / steps / force_stop / tools（调用了哪些工具）
    """
    cost = {"in": 0, "out": 0, "steps": 0, "force_stop": False, "tools": []}

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    for step in range(1, MAX_STEPS + 1):
        over_budget = (cost["in"] + cost["out"]) >= MAX_TOTAL_TOKENS
        if over_budget and not cost["force_stop"]:
            cost["force_stop"] = True
            if verbose:
                print(f"  [预算告警] 已用 {cost['in'] + cost['out']} token，强制收尾")
            messages.append({"role": "user",
                             "content": "已超出预算。请立即基于现有信息作答，不要再调用工具。"})

        resp = requests.post(
            CHAT_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "none" if over_budget else "auto",
                "temperature": 0,
            },
            timeout=120,
        )
        if resp.status_code in (401, 402, 403, 404):
            raise SystemExit(f"[错误 {resp.status_code}] 检查 Key / 余额 / 模型名")
        resp.raise_for_status()

        data = resp.json()
        usage = data.get("usage", {}) or {}
        cost["in"] += usage.get("prompt_tokens", 0)
        cost["out"] += usage.get("completion_tokens", 0)
        cost["steps"] += 1

        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        content = (msg.get("content") or "").strip()

        if not tool_calls:
            if content:
                if verbose:
                    print(f"  [第 {step} 轮] 直接回答")
                return content, cost

            # 兜底：模型返回空内容（Agent 常见毛病）
            if verbose:
                print(f"  [第 {step} 轮] 空内容，提示它必须作答")
            messages.append({"role": "user",
                             "content": "请基于已检索到的内容直接回答上面的问题，"
                                        "必须给出结论并标注页码；若确实没有，回答‘文件中未提及’。"})
            continue

        messages.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                args = {}
            cost["tools"].append(fn)

            if verbose:
                if fn == "search":
                    print(f"  [第 {step} 轮] search：{args.get('query', '')}")
                else:
                    print(f"  [第 {step} 轮] read_page：第 {args.get('page', '')} 页")

            if fn == "search":
                result = do_search(args.get("query", ""), chunks, bm25)
            elif fn == "read_page":
                result = do_read_page(int(args.get("page", 0)), pages)
            else:
                result = f"未知工具：{fn}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    return "（超过最大轮数，未得到最终回答）", cost


def estimate_cost(cost):
    return cost["in"] * PRICE_IN + cost["out"] * PRICE_OUT
