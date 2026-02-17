from dotenv import load_dotenv
from typing import Typed_Dict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages

from agent.tools import get_n_random_words # tool that has to be built


load_dotenv() 