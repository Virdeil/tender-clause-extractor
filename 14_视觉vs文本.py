# -*- coding: utf-8 -*-
"""
视觉 vs 文本：同一页表格，两种路线谁答得对

文本路线：PDF → 文本 → 句子 → 检索 → 生成（表格结构已丢）
视觉路线：PDF → 页面图片 → 视觉模型直接读

依赖：pip install pymupdf requests
"""

import os
import io
import json
import base64

import fitz
import requests

import rag_core


PDF = r"C:\Users\admin\Downloads\fbdee8c9f7164881a7a348f8d3d9a590.pdf"
PAGE_NO = 8                      # 供应商须知资料表（页脚印的是 6）

VLM_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
VLM_URL = "https://api.siliconflow.cn/v1/chat/completions"

API_KEY = os.environ.get("SF_API_KEY", "")


# 表格题：答案都在 P8 的资料表里，且都是"某行某列"
QUESTIONS = [
    ("本项目非主体、非关键性工作是否允许分包？", "不允许"),
    ("响应文件的有效期是多少天？从哪天起算？", "60 日历天，自提交响应文件的截止之日起算"),
    ("中标后需要递交几份纸质响应文件？", "三份"),
    ("对磋商文件有疑问，最晚什么时候提出？", "2026 年 10 月 16 日 12:00 前"),
]


def render_page(pdf_path, pno, dpi=140, quality=85):
    """把某一页渲染成 JPEG 字节流"""
    doc = fitz.open(pdf_path)
    page = doc[pno - 1]
    pix = page.get_pixmap(dpi=dpi)
    data = pix.tobytes("jpeg", jpg_quality=quality)
    doc.close()
    return data


def ask_vlm(image_bytes, question):
    """把图片和问题一起发给视觉模型"""
    b64 = base64.b64encode(image_bytes).decode()
    resp = requests.post(
        VLM_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": VLM_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text",
                     "text": f"下面这份招标文件这一页的表格里，回答：{question}\n"
                             f"只回答表里写的内容，找不到就回答“未提及”。"},
                ],
            }],
            "temperature": 0,
        },
        timeout=180,
    )
    if resp.status_code != 200:
        return f"[错误 {resp.status_code}] {resp.text[:200]}"
    d = resp.json()
    return d["choices"][0]["message"]["content"]


def ask_text_rag(question, chunks, bm25, pages):
    cand = rag_core.retrieve(question, chunks, bm25, k=20)
    hits = rag_core.rerank(question, cand, top_n=5)
    ph = rag_core.expand_to_pages(hits, pages)
    text, _, _ = rag_core.generate(question, ph)
    return text


def main():
    if not API_KEY:
        raise SystemExit("没有 SF_API_KEY")
    for f in ("index.json", "pages.json"):
        if not os.path.exists(f):
            raise SystemExit(f"缺少 {f}，先跑 03_建索引.py")

    print(f"渲染 PDF 第 {PAGE_NO} 页...")
    img = render_page(PDF, PAGE_NO)
    print(f"图片 {len(img)/1024:.0f} KB\n")

    chunks = json.load(open("index.json", encoding="utf-8"))
    pages = rag_core.load_pages("pages.json")
    bm25 = rag_core.build(chunks)

    text_ok = vlm_ok = 0

    for q, gold in QUESTIONS:
        print("=" * 76)
        print(f"问：{q}")
        print(f"标准：{gold}")

        t = ask_text_rag(q, chunks, bm25, pages)
        v = ask_vlm(img, q)

        # 简单判定：答案里是否出现标准答案的关键片段
        key = gold.replace(" ", "")[:4]
        t_hit = key in t.replace(" ", "")
        v_hit = key in v.replace(" ", "")
        text_ok += t_hit
        vlm_ok += v_hit

        print(f"\n  【文本 RAG】{'√' if t_hit else 'X'}")
        print(f"  {t.strip()[:220].replace(chr(10), ' ')}")
        print(f"\n  【视觉模型】{'√' if v_hit else 'X'}")
        print(f"  {v.strip()[:220].replace(chr(10), ' ')}")
        print()

    print("=" * 76)
    print(f"文本 RAG  {text_ok}/{len(QUESTIONS)}")
    print(f"视觉模型  {vlm_ok}/{len(QUESTIONS)}")


if __name__ == "__main__":
    main()
