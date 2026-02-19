import asyncio
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display
from agent.tools import get_n_random_words, get_n_random_words_by_difficulty_level

load_dotenv()

# The agent state: which is like the short-term memory of the agent
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    source_language: Optional[str]
    number_of_words: Optional[int]
    word_difficulty: Optional[str]
    
# Tools: these are additional capabilites that an agent can use to achieve its goal
local_tools = [
    get_n_random_words,
    get_n_random_words_by_difficulty_level,
]

# Currently the below async function is unecessary and returns a copy of local tools, but will be essential for MCP integration
# where we'll need async loading of remote tools from MCP servers
async def setup_tools():
    return [*local_tools] # list unpacking to return a copy of the list


# The assistant function: this acts like the central planner of the agent, allowing the LLM
# to decompose a problem, evaluate the steps already carried out, and select which
# tools to use

def assistant(state: AgentState):
    
    textual_description_of_tools = """
    
    def get_n_random_words(language: str, n: int) -> list:
    
    Retrieve a specified number of random words from a language-specific word list.
    
    Args:
        language (str): The language code (e.g., 'spanish', 'french') to determine 
                       which word list to use. Must correspond to a directory 
                       in the 'data' folder.
        n (int): The number of random words to retrieve. Must be less than or 
                equal to the total number of words in the word list.
    
    Returns:
        list: A list of randomly selected words from the specified language's 
              word list. Each word is returned as a string.
    
    Raises:
        FileNotFoundError: If the word list file doesn't exist for the specified 
                          language.
        KeyError: If the word list file structure is invalid.
        ValueError: If n is larger than the available words in the list.
    
    Note:
        The function expects word lists to be stored in JSON format at:
        'data/{language}/word-list-cleaned.json'
        Each entry in the JSON should have a 'word' key containing the word.
    
    
    
    def get_n_random_words_by_difficulty_level(language:str,
                                           difficulty_level:str,
                                           n:int) -> list:
    
    Retrieve a specified number of random words from a language-specific word list,
    filtered by difficulty level.
 
    Args:
        language (str): The language code (e.g., 'spanish', 'french') to determine 
                       which word list to use. Must correspond to a directory 
                       in the 'data' folder.
        difficulty_level (str): The difficulty level to filter words by. Must match
                               the 'difficulty_level' field in the word list entries.
                               The only valid values are "beginner", "intermediate", "advanced".
        n (int): The number of random words to retrieve. Must be less than or 
                equal to the total number of words available for the specified 
                difficulty level.
 
    Returns:
        list: A list of randomly selected words from the specified language's 
              word list that match the given difficulty level. Each word is 
              returned as a string.
 
    Raises:
        FileNotFoundError: If the word list file doesn't exist for the specified 
                          language.
        KeyError: If the word list file structure is invalid or if 'difficulty_level'
                 field is missing from word entries.
        ValueError: If n is larger than the available words for the specified 
                   difficulty level, or if no words match the difficulty level.
 
    Note:
        The function expects word lists to be stored in JSON format at:
        'data/{language}/word-list-cleaned.json'
        Each entry in the JSON should have a 'word' key containing the word and 
        a 'word_difficulty' key containing the difficulty classification.
    """
    
    sys_msg = SystemMessage(content=f"""
        You are a helpful language learning assistant. You have access to the foolwing tools {textual_description_of_tools}
        
        The user is going to give you a command.
        
        Your job is to check:
        1. Which source language the user wants words from.
        2. How many words they want.
        3. Whether they want words of a specific difficulty or just random words.
        
        Here are some example workflows:
        input: Get 20 random words in Spanish
        source language: Spanish
        number of words: 20
        
        input: Get 10 hard words in German
        source language: German
        number of words: 10
        word difficulty: advanced
        """)
    
    tools = local_tools
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    llm_with_tools = llm.bind_tools(tools)
    
    return{
        "messages" : [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "source_language": state["source_language"],
        "number_of_words": state["number_of_words"],
        "word_difficulty": state["word_difficulty"]
    }
    
    
async def build_graph():    # Turns the simple assitant function into a LangGraph agent
    """Build the state graph with properly initialized tools"""
    tools = await setup_tools()
    assistant.tools = tools
    
    builder = StateGraph(AgentState)
    
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    
    app = builder.compile()
    
    # Generate PNG image of the graph
    image_data = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(image_data)
        
        
    return app



async def main():
    """ Main async function to run the application"""
    react_graph = await build_graph()
    
    user_prompt = "Please get 10 average difficulty words in German."
    
    messages = [HumanMessage(content=user_prompt)]
    
    # Use ainvoke instead of invoke for async execution
    result = await react_graph.ainvoke({
        "messages": messages,
        "source_language": None,
        "number_of_words": None,
        "word_difficulty": None,
    })
    
    print(f"Final messages: {result['messages'][-1].content}")
    
    
if __name__ == "__main__":
    asyncio.run(main())