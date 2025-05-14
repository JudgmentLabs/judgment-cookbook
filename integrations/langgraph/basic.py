from typing import Annotated, List
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from judgeval.common.tracer import Tracer
from judgeval.integrations.langgraph import JudgevalCallbackHandler
import os
from dotenv import load_dotenv
from judgeval.scorers import AnswerRelevancyScorer, ExecutionOrderScorer, AnswerCorrectnessScorer
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import ToolOrderScorer

load_dotenv()

PROJECT_NAME = "LangGraphBasic"

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"), project_name=PROJECT_NAME)
handler = JudgevalCallbackHandler(judgment)
client = JudgmentClient()
# REPLACE THIS WITH YOUR OWN TOOLS
def search_restaurants(location: str, cuisine: str) -> str:
    """Search for restaurants in a location with specific cuisine"""
    ans = f"Top 3 {cuisine} restaurants in {location}: 1. Le Gourmet 2. Spice Palace 3. Carbones"
    return ans

# REPLACE THIS WITH YOUR OWN TOOLS
def check_opening_hours(restaurant: str) -> str:
    """Check opening hours for a specific restaurant"""
    ans = f"{restaurant} hours: Mon-Sun 11AM-10PM"
    return ans

# REPLACE THIS WITH YOUR OWN TOOLS
def get_menu_items(restaurant: str) -> str:
    """Get popular menu items for a restaurant"""
    ans = f"{restaurant} popular dishes: 1. Chef's Special 2. Seafood Platter 3. Vegan Delight"
    example = Example(
        input="Get popular menu items for a restaurant",
        actual_output=ans
    )
    judgment.async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=1)],
        example=example,
        model="gpt-4.1"
    )
    return ans 

def run_agent(prompt: str):
    tools = [
        TavilySearchResults(max_results=2),
        check_opening_hours,
        get_menu_items,
        search_restaurants,
    ]

    llm = ChatOpenAI(model="gpt-4.1")

    graph_builder = StateGraph(State)

    def assistant(state: State):
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)
    
    graph_builder.add_node("assistant", assistant)
    graph_builder.add_node("tools", tool_node)
    
    graph_builder.set_entry_point("assistant")
    graph_builder.add_conditional_edges(
        "assistant",
        lambda state: "tools" if state["messages"][-1].tool_calls else END
    )
    graph_builder.add_edge("tools", "assistant")
    
    graph = graph_builder.compile()

    config_with_callbacks = {"callbacks": [handler]}

    result = graph.invoke({
        "messages": [HumanMessage(content=prompt)]
    }, config_with_callbacks)

    return result, handler

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "test.yaml")

    client.assert_test(
        test_file=yaml_path,
        scorers=[ToolOrderScorer()],
        function=run_agent,
        tracer=handler,
        override=True
    )