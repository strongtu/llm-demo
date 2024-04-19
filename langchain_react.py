import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

os.environ["SERPAPI_API_KEY"] = '3fa06ebd49fc9b59d477fc3ddb36183fcab53b41ddf180cc1aeca09521ee1cc5'

llm = OpenAI(api_key = "EMPTY", base_url = "http://localhost:8000/v1")

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？")
