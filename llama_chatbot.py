# 本地部署GPU的llamacpp
# https://studyinglover.com/2023/08/06/GPU%E9%83%A8%E7%BD%B2llama-cpp-python(llama.cpp%E9%80%9A%E7%94%A8)/
# llama python部署
# https://zhuanlan.zhihu.com/p/646578831
# llama.cpp本地编译部署
# https://github.com/ggerganov/llama.cpp
# Knowledge
# https://zhuanlan.zhihu.com/p/646578831

from llama_cpp import Llama
from llama_prompt import *

ai_name = "Bob:"
user_name = "User:"

llm = Llama(model_path="../llama.cpp/models/chinese-alpaca-2-13b-16k.Q4_K.gguf", max_tokens=512, n_gpu_layers=128, n_threads=6, n_ctx=8192, n_batch=512, verbose=True)
#llm = Llama(model_path="../llama.cpp/models/vicuna-13b-v1.5-16k.Q4_K_M.gguf", max_tokens=512, n_gpu_layers=128, n_threads=6, n_ctx=4096, n_batch=512, verbose=True)


def generate_prompt(text, past_messages_path="chat_history.txt", character_json_path="instruction.txt"):
    chat_history = []
    instruction = ""

    with open(character_json_path, 'r', encoding='utf-8') as f:
        instruction = f.read()
    with open(past_messages_path, 'r', encoding='utf-8') as f:
        chat_history_lines = f.readlines()
        chat_history_lines.reverse()

    for message_text in chat_history_lines:
        message_text = message_text.strip()
        if message_text == "\n" or message_text == "":
            continue
        chat_history.insert(0, message_text)
        if len(chat_history) >= 10:
            break

    return format_prompt(instruction, chat_history, text)

past_messages_path = "chat_history.txt"
while True:
    text = input(user_name)
    if text == "exit":
        break
    prompt = generate_prompt(user_name + text, past_messages_path)
    output = llm(prompt, max_tokens=4000, echo=False)
    response = output["choices"][0]["text"]
    response = response.removeprefix(ai_name)
    print(ai_name + response)

    #write the last response to a file
    with open("last_response.txt", 'w', encoding='utf-8') as f:
        f.write(prompt)
        f.write(response)
    #add to chat history
    with open(past_messages_path, 'a', encoding='utf-8') as f:
        f.write(user_name + text)
        f.write("\n" + ai_name + response + "\n")
