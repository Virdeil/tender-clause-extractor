# -*- coding: utf-8 -*-
"""
建索引（Small-to-Big 版）

产出两个文件：
  index.json   —— 句子级小块，用于「检索」（语义纯净，命得准）
  pages.json   —— 整页全文，用于「生成」（上下文完整，答得全）

为什么要两个粒度：
  按页切 → 一页混多个主题，向量是「平均脸」，查保证金会命中讲下载方式的页
  按句切 → 检索准，但单句太短，模型答不全
  两者结合 → 用句子找，用整页答
"""

import os
import re
import json
import time

import requests

import rag_core


# ==================== 配置 ====================

API_KEY = os.environ.get("SF_API_KEY", "")
if not API_KEY:
    raise SystemExit("没有读到环境变量 SF_API_KEY")

EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "BAAI/bge-m3"

SRC_DIR = r"C:\Users\admin\Desktop\招标文件分章节"
# 原始 PDF：用来测「印刷页码 - PDF 页码」的偏移。
# 本项目这份 PDF 的页脚是图片，文本层抽不到，只能靠视觉模型读。
SRC_PDF = r"C:\Users\admin\Downloads\fbdee8c9f7164881a7a348f8d3d9a590.pdf"
OUT_INDEX = "index.json"
OUT_PAGES = "pages.json"

BATCH = 16
SENT_MAX = 90        # 单个检索单元的最大字数
SENT_MIN = 12        # 太短的句子没意义，丢弃


# ==================== 切分 ====================

def split_sentences(text, max_len=SENT_MAX):
    """把一段正文切成句子级小块。先按句末标点切，过长的再按逗号补刀。"""
    raw = re.split(r"(?<=[。；！？])|\n", text)
    out = []

    for s in raw:
        s = s.strip()
        if not s:
            continue
        if len(s) <= max_len:
            out.append(s)
            continue

        buf = ""
        for piece in re.split(r"(?<=[，、,])", s):
            if len(buf) + len(piece) <= max_len:
                buf += piece
            else:
                if buf:
                    out.append(buf.strip())
                buf = piece
        if buf.strip():
            out.append(buf.strip())

    return [x for x in out if len(x) >= SENT_MIN]


def build(src_dir, offset=0):
    """返回 (句子块列表, 整页字典)"""
    sentences, pages = [], {}

    for fn in sorted(os.listdir(src_dir)):
        if not fn.endswith(".txt") or fn.startswith(("00_", "10_", "99_")):
            continue

        with open(os.path.join(src_dir, fn), encoding="utf-8") as f:
            lines = f.read().splitlines()

        page, buf = None, []

        def flush():
            if page is None or not buf:
                return
            body = "".join(x.strip() for x in buf).strip()
            if not body:
                return
            printed = page + offset
            pages[f"{fn}|{page}"] = {"file": fn, "page": page,
                                     "printed": printed, "text": body}
            for s in split_sentences(body):
                sentences.append({"file": fn, "page": page,
                                  "printed": printed, "text": s})

        for ln in lines:
            s = ln.strip()
            if s.startswith("#"):
                continue
            if s.startswith("【PDF 第") and "页】" in s:
                flush()
                buf = []
                page = int(s.replace("【PDF 第", "").replace("页】", "").strip())
                continue
            if page is not None and s:
                buf.append(s)

        flush()

    return sentences, pages


# ==================== 向量化 ====================

def embed(texts):
    for attempt in range(1, 4):
        try:
            resp = requests.post(
                EMBED_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": EMBED_MODEL, "input": texts},
                timeout=180,
            )
            if resp.status_code in (401, 402, 403, 404):
                print(f"    [错误 {resp.status_code}] 配置问题，停止")
                raise SystemExit(1)
            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda d: d["index"])
            return [d["embedding"] for d in data]
        except requests.exceptions.RequestException as e:
            print(f"    第 {attempt} 次失败：{type(e).__name__}")
            if attempt < 3:
                time.sleep(3)
    raise SystemExit("向量化失败")


# ==================== 主流程 ====================

def main():
    offset = 0
    if os.path.exists(SRC_PDF):
        print("0) 测量页码偏移（视觉读页脚）...")
        offset = rag_core.detect_page_offset(open(SRC_PDF, "rb").read())
        print(f"   印刷页码 = PDF 页码 {offset:+d}")
    else:
        print("0) 未找到原始 PDF，页码偏移按 0 处理")

    print("\n1) 切分（句子级 + 整页）...")
    sentences, pages = build(SRC_DIR, offset)
    print(f"   句子块 {len(sentences)} 个 ｜ 整页 {len(pages)} 页")
    lens = [len(s["text"]) for s in sentences]
    print(f"   句子长度：最短 {min(lens)}，最长 {max(lens)}，平均 {sum(lens)//len(lens)}")

    print("\n2) 计算向量（句子块）...")
    t0 = time.time()
    for i in range(0, len(sentences), BATCH):
        batch = sentences[i:i + BATCH]
        vecs = embed([s["text"] for s in batch])
        for s, v in zip(batch, vecs):
            s["vector"] = v
        print(f"   {i + len(batch)}/{len(sentences)}", end="\r")
    print(f"\n   耗时 {time.time() - t0:.1f} 秒")

    print("\n3) 保存...")
    with open(OUT_INDEX, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False)
    with open(OUT_PAGES, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False)

    print(f"   {OUT_INDEX}: {os.path.getsize(OUT_INDEX)/1024:.0f} KB")
    print(f"   {OUT_PAGES}: {os.path.getsize(OUT_PAGES)/1024:.0f} KB")


if __name__ == "__main__":
    main()
