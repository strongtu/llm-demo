import sys
import requests

# 复杂prompt的参考
# 从Langchain到ReAct，在大模型时代下全新的应用开发核心
# https://developer.aliyun.com/article/1241363
# LangChain干货（2）：ReAct框架，推理与行动的协同！
# https://zhuanlan.zhihu.com/p/661495311
# 
# LangChain新手教程： 使用 Memory 功能，打造有记忆的聊天机器人！
# https://zhuanlan.zhihu.com/p/642223972
#
# 使用“大模型”驱动文字冒险游戏的尝试
# https://zhuanlan.zhihu.com/p/659493577
#
# NLP（廿一）：从 RAG 到 Self-RAG —— LLM 的知识增强
# https://zhuanlan.zhihu.com/p/661465330
# https://blog.csdn.net/sinat_37574187/article/details/133148372

FILE_NAME = "llama_job_summarize.txt"
#获取命令行参数
if (len(sys.argv) > 1):
    FILE_NAME = sys.argv[1]

#打开文件读取文本内容到job
with open(FILE_NAME, 'r', encoding='utf-8') as f:
    job = f.read()

# 调用接口
URL = "http://127.0.0.1:8000/api/v1/summarize"
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer <PASSWORD>'
}

data = {
    "job": job
}

response = requests.post(URL, json=data, headers=headers)

print(response.text)

URL = 'http://localhost:8000/v1/chat/completions'
headers = {
	'accept': 'application/json',
	'Content-Type': 'application/json'
}
data = {
  "messages": [
    {
      "content": job,
      "role": "system"
    },
  ]
}

response = requests.post(URL, headers=headers, json=data)
j = response.json()
c =  j['choices'][0]['message']['content']
print(c)
