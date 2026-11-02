# -*- coding: utf-8 -*-
"""
Day 11：批量处理 + 长文本分段 + 结果去重

改动点（相比上一版）：
1. 长文本自动分段（每段 4000 字），避免超出模型输入长度
2. 多段结果合并，按条款号去重
3. 打印 token 用量，方便观察成本
"""

import os
import json
import time

import requests
from openpyxl import Workbook


# ==================== 配置区 ====================

# 从环境变量读 key，代码里不出现明文（防止上传 GitHub 后泄露）
API_KEY = os.environ.get("SF_API_KEY", "")
if not API_KEY:
    raise SystemExit("没有读到环境变量 SF_API_KEY，请先设置后再运行。")

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V3"

INPUT_DIR = "批量测试"
OUTPUT_FILE = "条款汇总.xlsx"

CHUNK_SIZE = 4000    # 每段字符数。太长会被截断，太短会切断条款


# ==================== 提示词 ====================

PROMPT = """从下面这段政府采购磋商文件中，逐条抽出所有编号为 26.3.x 的条款。

要求：
1. 输出 JSON 数组，每项包含两个字段：clause_no（条款号）、text（原文）
2. text 必须是原文照抄，不要改写、不要总结
3. 只输出 JSON，第一个字符必须是 [，最后一个字符必须是 ]
4. 如果这段文字里没有 26.3.x 的条款，输出空数组 []

文件内容：
{text}"""


# ==================== 工具函数 ====================

def split_text(text, size=CHUNK_SIZE):
    """把长文本切成若干段，每段不超过 size 字符。

    优先在换行处切，避免把一句话劈成两半。
    """
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + size, n)

        # 如果不是最后一段，往后找最近的换行，在那儿断开
        if end < n:
            nl = text.rfind("\n", start, end)
            if nl > start + size // 2:      # 别切得太短
                end = nl + 1

        chunks.append(text[start:end])
        start = end

    return chunks


def call_api(text, retry=3):
    """调用一次 API，返回 (条文列表, 输入token, 输出token)。"""
    for attempt in range(1, retry + 1):
        try:
            resp = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
                    "temperature": 0,
                },
                timeout=180,
            )

            if resp.status_code == 200:
                pass
            elif resp.status_code in (401, 402, 403, 404):
                reasons = {
                    401: "API Key 不正确或已失效",
                    402: "账户余额不足",
                    403: "没有权限访问该模型",
                    404: "接口地址或模型名不存在",
                }
                print(f"    [错误 {resp.status_code}] {reasons.get(resp.status_code)}")
                raise SystemExit(1)
            else:
                resp.raise_for_status()

            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            in_tok = usage.get("prompt_tokens", 0)
            out_tok = usage.get("completion_tokens", 0)

            start, end = answer.find("["), answer.rfind("]")
            if start < 0 or end <= start:
                return [], in_tok, out_tok

            return json.loads(answer[start:end + 1]), in_tok, out_tok

        except requests.exceptions.RequestException as e:
            print(f"    第 {attempt} 次失败（网络）：{type(e).__name__}")
            if attempt < retry:
                time.sleep(3)

    return [], 0, 0


def extract_from_file(text):
    """一个文件 → 分段 → 逐段抽取 → 合并去重。"""
    chunks = split_text(text)
    merged = {}
    tok_in = tok_out = 0

    for i, chunk in enumerate(chunks, 1):
        clauses, ti, to = call_api(chunk)
        tok_in += ti
        tok_out += to

        for c in clauses:
            no = c.get("clause_no", "").strip()
            if not no:
                continue
            # 去重：同一个条款号只保留第一次出现的
            if no not in merged:
                merged[no] = c.get("text", "")

        print(f"    第 {i}/{len(chunks)} 段（{len(chunk)} 字）→ {len(clauses)} 条")

    return merged, tok_in, tok_out


# ==================== 主流程 ====================

def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"找不到文件夹：{INPUT_DIR}")
        return

    files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.endswith(".txt")]
    print(f"共发现 {len(files)} 个文件\n")

    wb = Workbook()
    ws = wb.active
    ws.title = "条款汇总"
    ws.append(["文件名", "条款号", "原文"])

    total = 0
    summary = []

    for i, fn in enumerate(files, 1):
        path = os.path.join(INPUT_DIR, fn)
        with open(path, encoding="utf-8") as f:
            text = f.read()

        print(f"[{i}/{len(files)}] {fn}（{len(text)} 字）")

        t0 = time.time()
        merged, ti, to = extract_from_file(text)
        cost = time.time() - t0

        print(f"    → 去重后 {len(merged)} 条，耗时 {cost:.1f}s，"
              f"token 输入 {ti} / 输出 {to}")

        # 按条款号排序，输出更整齐
        def sort_key(k):
            parts = k.replace("26.3.", "").split(".")
            return int(parts[-1]) if parts[-1].isdigit() else 999

        for no in sorted(merged, key=sort_key):
            ws.append([fn, no, merged[no]])
            total += 1

        summary.append((fn, len(merged)))
        print()

    wb.save(OUTPUT_FILE)

    print("=" * 48)
    for fn, n in summary:
        print(f"{fn:<26} {n:>3} 条")
    print("-" * 48)
    print(f"合计 {total} 条，已写入 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
