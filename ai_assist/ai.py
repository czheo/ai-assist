import os
from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import load_tools
from langchain.tools.python.tool import PythonREPLTool
from langchain.utilities import BashProcess
from dotenv import load_dotenv


def setup():
    load_dotenv(os.path.expanduser("~/.env"))
    search = GoogleSearchAPIWrapper(k=10)
    bash = BashProcess()
    tools_llm = OpenAI()
    memory = ConversationSummaryBufferMemory(llm=tools_llm, max_token_limit=100, memory_key="chat_history", return_messages=True)
    tools = load_tools(["llm-math"], llm=tools_llm)
    tools.extend([
        Tool(
            name="google search",
            func=search.run,
            description="useful when you need to ask with search"
        ),
        PythonREPLTool(),
        Tool(
            name="bash",
            func=bash.run,
            description="useful when you need to run a shell command to interact with your local machine. the raw output of the command will be returned."
        ),
    ])

    llm = ChatOpenAI(temperature=0.9)
    return initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)


def interact(agent_chain):
    print("""Welcome to the AI.
I do math, search, run python/bash and more.
Type 'exit' to quit.""")
    while True:
        user_input = input('[USER]<< ').strip()
        if user_input in ("exit", ":q", "quit"):
            break
        try:
            response = agent_chain.run(user_input)
            print('[AI]>>', response)
        except Exception as e:
            print("ERROR: ", e)


def cli():
    agent_chain = setup()
    interact(agent_chain)
