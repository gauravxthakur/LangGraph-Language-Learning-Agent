from dotenv import load_dotenv
from typing import Typed_Dict, Annotated, Optional, TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from agent.tools import get_n_random_words

load_dotenv()

# The agent state: which is like the short-term memory of the agent
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    
# Tools: these are additional capabilites that an agent can use to achieve its goal
local_tools = [
    get_n_random_words,
]

# The assistant function: this acts like the central planner of the agent, allowing the LLM
# to decompose a problem, evaluate the steps already carried out, and select which
# tools to use

def assistant(state: AgentState):
    
    sys_msg = SystemMessage(content=f"""
        You are a helpful language learning assistant.
        
        The use is going to give you a command.
        """)
    
    tools = local_tools
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    llm_with_tools = llm.bind_tools(tools)
    
    return{
        "messages" : [llm_with_tools.invoke([sys_msg] + state["messages"])],
    }