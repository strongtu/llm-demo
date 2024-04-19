ai_name = "Bob:"
user_name = "User:"


# 按照待###指引的形式
def format_prompt(instruction, chat_history, text):
    prompt = instruction
    s = "\n".join(chat_history)
    prompt = prompt.replace("{history}", s)
    prompt = prompt.replace("{input}", text)
    return prompt


# 裸文本prompt
def format_prompt2(instruction, chat_history, text):
    s = "\n".join(chat_history)
    prompt =  "".join([
                instruction,
                "\n",
                s,
                "\n",
                text
    ])
    return prompt


# llama标准<INST>的prompt拼接
# <s>
#   [INST] 
#       <<SYS>>instruction<</SYS>>
#      {user_message}
#   [/INST]
#   {bot_reply}
# </s>
# <s>
#   [INST]{user_message}[/INST]
#   {bot_reply}
# </s>
#
#TEMPLATE = (
#     "[INST] <<SYS>>\n"
#     "{system_prompt}\n"
#     "<</SYS>>\n\n"
#     "{instruction} [/INST]"
# )
def format_prompt3(instruction, chat_history, text):
    formatted_chat_history = []
    temp = {}

    for message_text in chat_history:
        if message_text.startswith(user_name):
            if temp:
                formatted_chat_history.append(temp)
                temp = {}
            temp["User:"] = message_text.removeprefix(user_name)
        elif message_text.startswith(ai_name):
            temp["ChatBot:"] = message_text.removeprefix(ai_name)
    if temp:
        formatted_chat_history.append(temp)

    temp = {"User:": text.removeprefix(user_name)}
    formatted_chat_history.append(temp)

    add_ins = True
    prompt = ""
    for chat in formatted_chat_history:
        prompt = prompt + "<s>"
        if user_name in chat and add_ins:
            add_ins = False
            prompt = prompt + "[INST]" + " <<SYS>>\n" + instruction + "\n<</SYS>>\n"
            prompt = prompt + chat["User:"] + "[/INST]"
        elif user_name in chat:
            prompt = prompt + "[INST]\n" + chat["User:"] + "[/INST]"
        if ai_name in chat:
            prompt = prompt + chat["ChatBot:"]
        prompt = prompt + "</s>\n"

    last_s_index = prompt.rfind("</s>")
    if last_s_index != -1:
        prompt = prompt[:last_s_index] + prompt[last_s_index + 4:]
    return prompt
