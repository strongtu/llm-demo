import os
import glob
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_rag_fix import FixOpenAIEmbeddings

# langchain 默认RAG的prompt template
# 工作逻辑是，知识库chunk和问题都进行embedding，然后检索相近的4个chunk，插入到prompt中作为context
template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:
"""

# 从目录中加载所有txt，切割为块embedding后存储在VDB中
directory = './knowledge'
text_files = glob.glob(os.path.join(directory, '*.txt'))
texts = []

for txt_file in text_files:
    with open(txt_file, encoding="utf-8") as f:
        kn_text = f.read()     
        # 文档分割（分块）
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts.extend(text_splitter.split_text(kn_text))

os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"

# 调用openai Embedding，并且使用FAISS
embeddings = FixOpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embedding=embeddings)

# 检索，使用LangChain的RetrievalQA链来初始化qa对象
llm = OpenAI()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

while True:
    input_text = input("User:")
    if input_text == "exit":
        break
    else:
        r1 = llm(input_text)
        print("no RAG:", r1)
        r2 = qa.run(input_text)
        print("RAG:", r2)
