import os
from argparse import ArgumentParser
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
from termcolor import colored


GOOGLE_SEARCH_LIMIT = 10
MEMORY_MAX_TOKEN_LIMIT = 300


def setup(model_name, verbose):
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

    llm = ChatOpenAI(temperature=0.7, model_name=model_name)
    return initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=verbose, memory=memory)


def interact(agent_chain):
    print(colored("""Welcome to the AI.
I do math, search, run python/bash and more.
Type 'exit' to quit.""", color = 'green'))
    while True:
        user_input = input('[USER]<< ').strip()
        if user_input in ("exit", ":q", "quit"):
            break
        try:
            response = agent_chain.run(user_input)
            print(colored('[AI]>> ' + response, color = 'green'))
        except Exception as e:
            print(colored("ERROR: \n" + e, color = 'red'))


def cli():
    parser = ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model name: https://platform.openai.com/docs/models/")
    args = parser.parse_args()
    agent_chain = setup(args.model, args.verbose)
    interact(agent_chain)
