# -*- coding: utf-8 -*-
"""
第一个脚本：读一段招标文件 → 调大模型 API → 输出结构化条款

运行前：
1. 已在系统里设置环境变量 SF_API_KEY（存放 API Key，不写进代码）
2. 已安装 requests：pip install requests

设环境变量（只需一次，在 PowerShell 里跑）：
   [Environment]::SetEnvironmentVariable("SF_API_KEY", "sk-你的key", "User")
然后**关掉终端重开**，环境变量才生效。
"""

import os
import json
import requests

# ========== 配置区 ==========

# 从环境变量读 key，代码里不出现明文
API_KEY = os.environ.get("SF_API_KEY", "")
if not API_KEY:
    raise SystemExit(
        "没有读到环境变量 SF_API_KEY。\n"
        "请先在 PowerShell 里执行：\n"
        '  [Environment]::SetEnvironmentVariable("SF_API_KEY","sk-你的key","User")\n'
        "然后关掉终端，重新打开再运行。"
    )

# 硅基流动的接口地址（国内可直连）
API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# 模型名。如果报错说模型不存在，就去 siliconflow.cn 的「模型广场」换一个
MODEL = "deepseek-ai/DeepSeek-V3"

# ========== 1. 读文件 ==========

# with open(...) as f：打开文件，用完自动关
with open("sample.txt", encoding="utf-8") as f:
    text = f.read()

print(f"读入字符数：{len(text)}")


# ========== 2. 拼提示词 ==========

# f"""...""" 是格式化字符串，{text} 会被替换成上面读到的内容
prompt = f"""从下面这段政府采购磋商文件中，逐条抽出所有编号为 26.3.x 的条款。

要求：
1. 输出 JSON 数组，每项包含两个字段：clause_no（条款号）、text（原文）
2. text 必须是原文照抄，不要改写、不要总结
3. 只输出 JSON，第一个字符必须是 [，最后一个字符必须是 ]

文件内容：
{text}"""


# ========== 3. 调用 API ==========

resp = requests.post(
    API_URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,        # 0 = 尽量稳定
    },
    timeout=120,
)

# 如果状态码不是 200，这里会主动抛错，方便看到问题
resp.raise_for_status()
data = resp.json()


# ========== 4. 取出模型的回复 ==========

answer = data["choices"][0]["message"]["content"]

print("\n=== 模型原始输出 ===")
print(answer[:500])


# ========== 5. 解析 JSON ==========

# 兜底：模型有时套 ```json 代码块或多说一句，先截取第一个 [ 到最后一个 ]
start = answer.find("[")
end = answer.rfind("]")
if start >= 0 and end > start:
    answer = answer[start:end + 1]

items = json.loads(answer)

print(f"\n=== 共抽出 {len(items)} 条 ===")
for it in items:
    print(f'{it["clause_no"]:>8} | {it["text"][:35]}...')

print("\n完成。这就是「程序调用大模型」的全部。")
