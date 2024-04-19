from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# 使用OpenAI的Web API接口(可以使用Llama本地模拟服务)
llm = ChatOpenAI(openai_api_key = "EMPTY", openai_api_base = "http://localhost:8000/v1", max_tokens=1024, verbose=True)

print("simple use=========================================================")
response = llm.predict("What is an elephant?")
print(response)

print("use with prompt template + Chain===================================")
template = """
Question: {question}
Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What is the capital of France?"
response = llm_chain.run(question)
print(response)

print("use openai api=====================================================")
messages = [
    SystemMessage(content="""You are a zoologist, you will answer my questions about animals.
You only answer question in English and will never repeat
In following chat history, you are the "ChatBot" and Don't add "ChatBot:" with your reply
Respond to these messages as your character would:"""),
    HumanMessage(content="what is an elephant？"),
    AIMessage(content="it is a kind of animal"),
    HumanMessage(content="how long can it live"),
    AIMessage(content="The average lifespan of an elephant in the wild is between 50 and 70 years, while in captivity they can live up to 80 or more years. However, some factors such as habitat loss, poaching, and human interactions can negatively affect their lifespan"),
    HumanMessage(content="Is it delicious?")
]
response = llm(messages)
print(response.content)

print("use with chat and history=============================================")
template = """
"You are a zoologist, you will answer my questions about animals.
You only answer question in English and will never repeat
In following chat history, you are the "ChatBot" and Don't add "ChatBot:" with your reply
Respond to these messages as your character would:

{history}
User: {input}
ChatBot:"""

prompt = PromptTemplate(template=template, input_variables=["history", "input"])
memory = ConversationBufferMemory()
llm_chain = ConversationChain(
    llm = llm,
    prompt = prompt,
    memory = memory,
    verbose=True
)
result = llm_chain("what is an elephant？")
print(result['response'])

result = llm_chain("how long can it live")
print(result['response'])
