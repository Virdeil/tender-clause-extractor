# -*- coding: utf-8 -*-
"""
跨文件泛化冒烟测试

思路：
  没有标准答案，就用「引用支撑率」——它引用的那一页，是否真的包含问题的关键词。
  这是跨文件通用的客观下界指标。

用法：
  python 10_泛化测试.py
"""

import io
import os
import re
import sys
import time

import pypdf
import requests

sys.path.insert(0, r"C:\Users\admin\Desktop\python练习")
import rag_core


KEY = os.environ.get("SF_API_KEY", "")

FILES = [
    (r"C:\Users\admin\Desktop\泛化测试素材\河北_磋商文件.pdf", "河北 71 页"),
    (r"C:\Users\admin\Desktop\泛化测试素材\东西湖_招标文件.pdf", "东西湖 48 页"),
]

# (问题, 该问题在正确页面上必然出现的词)
QUESTIONS = [
    ("本项目的采购预算是多少？", "预算"),
    ("本项目的最高限价是多少？", "限价"),
    ("工期或交付期限是多久？", "工期"),
    ("投标保证金需要交多少？", "保证金"),
    ("响应文件提交的截止时间是什么时候？", "截止"),
    ("供应商需要具备什么资质？", "资质"),
    ("本项目是否接受联合体投标？", "联合体"),
    ("磋商有效期是多少天？", "有效期"),
]


def clean_text(t):
    """清洗 PDF 提取的文本。

    两个真实问题：
      1. 中文常带行内空格（"工 期"），不清理会导致关键词匹配和分词全错
      2. 部分 PDF 嵌了哈希串 / 水印（e1ad0753ac744a35b1e31b8d…），
         会插进句子中间把语义截断
    """
    # 去掉连续 16 位以上的字母数字串（哈希、水印编号）
    t = re.sub(r"[0-9a-zA-Z]{16,}", " ", t)
    # 去掉所有空白
    t = re.sub(r"\s+", "", t)
    return t


def parse_pages(path):
    r = pypdf.PdfReader(path)
    out = []
    for i, pg in enumerate(r.pages, 1):
        body = clean_text(pg.extract_text() or "")
        if body:
            out.append((i, body))
    return out


def build(pages):
    sentences, page_map = [], {}
    for pno, body in pages:
        page_map[f"f|{pno}"] = {"file": "f", "page": pno, "text": body}
        for s in rag_core.split_sentences(body):
            sentences.append({"file": "f", "page": pno, "text": s})
    return sentences, page_map


def embed_all(sentences):
    BATCH = 16
    for i in range(0, len(sentences), BATCH):
        batch = sentences[i:i + BATCH]
        vecs = rag_core.embed_texts([s["text"] for s in batch])
        for s, v in zip(batch, vecs):
            s["vector"] = v


def main():
    if not KEY:
        print("没有 SF_API_KEY")
        return

    grand_total = grand_ground = 0

    for path, label in FILES:
        if not os.path.exists(path):
            print(f"跳过（不存在）：{path}")
            continue

        print("=" * 78)
        print(f"文件：{label}")
        t0 = time.time()
        pages = parse_pages(path)
        sentences, page_map = build(pages)
        print(f"  解析：{len(pages)} 页 → {len(sentences)} 个句子块")
        embed_all(sentences)
        bm25 = rag_core.build(sentences)
        print(f"  建索引完成，用时 {time.time() - t0:.0f} 秒\n")

        grounded = 0
        for q, kw in QUESTIONS:
            cand = rag_core.retrieve(q, sentences, bm25, k=20)
            hits = rag_core.rerank(q, cand, top_n=5)
            page_hits = rag_core.expand_to_pages(hits, page_map)
            ans, ti, to = rag_core.generate(q, page_hits)

            cited = [int(n) for n in re.findall(r"第\s*(\d+)\s*页", ans)]

            # 引用支撑：它引用的页里，有没有那个关键词
            ok = False
            for p in cited:
                body = page_map.get(f"f|{p}", {}).get("text", "")
                if kw in body:
                    ok = True
                    break

            # 拒答也算一种状态，单独标注
            refused = any(m in ans for m in ["未提及", "未给出", "没有", "无法", "未提供"])
            if ok:
                grounded += 1

            mark = "OK" if ok else ("拒答" if refused else "XX")
            print(f"  [{mark:>2}] {q}")
            print(f"        引用页 {cited or '无'}（判定词：{kw}）")
            print(f"        {ans[:100].replace(chr(10), ' ')}")
            print()

        total = len(QUESTIONS)
        grand_total += total
        grand_ground += grounded
        print(f"  >>> 引用支撑率：{grounded}/{total} = {grounded/total*100:.0f}%\n")

    print("=" * 78)
    print(f"总体引用支撑率：{grand_ground}/{grand_total} = {grand_ground/max(grand_total,1)*100:.0f}%")


if __name__ == "__main__":
    main()
