from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

ai_name = "Bob:"
user_name = "User:"

llm = ChatOpenAI(openai_api_key = "EMPTY", openai_api_base = "http://localhost:8000/v1", max_tokens=128)


def generate_prompt(text, past_messages_path="chat_history2.txt", character_json_path="instruction2.txt"):
    messages = []
    chat_history = []
  
    with open(character_json_path, 'r', encoding='utf-8') as f:
        instruction = f.read()
    with open(past_messages_path, 'r', encoding='utf-8') as f:
        chat_history_lines = f.readlines()
        chat_history_lines.reverse()

    for message_text in chat_history_lines:
        message_text = message_text.strip()
        if message_text == "\n" or message_text == "":
            continue
        if message_text.startswith(user_name):
            chat_history.insert(0, HumanMessage(content=message_text.removeprefix(user_name)))
        if message_text.startswith(ai_name):
            chat_history.insert(0, AIMessage(content=message_text.removeprefix(ai_name)))
        if len(chat_history) >= 10:
            break
    messages.append(SystemMessage(content=instruction))
    if len(chat_history) <= 8:
        messages.append(HumanMessage(content="你好bob", example=True))
        messages.append(AIMessage(content="你好，我是Bob，你是谁？", example=True))
    for message in chat_history:
        messages.append(message)
    messages.append(HumanMessage(content=text))
    return messages

past_messages_path = "chat_history2.txt"
while True:
    input_text = input("User:")
    if input_text == "exit":
        break
    else:
        messages = generate_prompt(input_text)
        r = llm(messages)
        print(r.content)

        #add to chat history
        with open(past_messages_path, 'a', encoding='utf-8') as f:
            f.write(user_name + input_text)
            f.write("\n" + ai_name + r.content + "\n")
