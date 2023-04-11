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


GOOGLE_SEARCH_LIMIT = 10
MEMORY_MAX_TOKEN_LIMIT = 300


def setup():
    load_dotenv(os.path.expanduser("~/.env"))
    search = GoogleSearchAPIWrapper(k=GOOGLE_SEARCH_LIMIT)
    bash = BashProcess()
    tools_llm = OpenAI()
    memory = ConversationSummaryBufferMemory(llm=tools_llm, max_token_limit=MEMORY_MAX_TOKEN_LIMIT, memory_key="chat_history", return_messages=True)
    tools = load_tools(["llm-math"], llm=tools_llm)
    tools.extend([
        Tool(
            name="google search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world. the input should be a single search term."
        ),
        PythonREPLTool(),
        Tool(
            name="bash",
            func=bash.run,
            description="useful when you need to run a shell command to interact with the local machine. the input should be a bash command. the raw output of the command will be returned."
        ),
    ])

    llm = ChatOpenAI(temperature=0.7)
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
