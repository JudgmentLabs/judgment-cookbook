from typing import Annotated, List
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from judgeval.common.tracer import Tracer
from judgeval.integrations.langgraph import JudgevalCallbackHandler, set_global_handler
import os
from judgeval.scorers import AnswerRelevancyScorer, ExecutionOrderScorer, AnswerCorrectnessScorer
from judgeval import JudgmentClient
from judgeval.data import Example

PROJECT_NAME = "LangGraphBasic"

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"), project_name=PROJECT_NAME)

# REPLACE THIS WITH YOUR OWN TOOLS
def search_restaurants(location: str, cuisine: str, state: State) -> str:
    """Search for restaurants in a location with specific cuisine"""
    ans = f"Top 3 {cuisine} restaurants in {location}: 1. Le Gourmet 2. Spice Palace 3. Carbones"
    example = Example(
        input="Search for restaurants in a location with specific cuisine",
        actual_output=ans
    )
    judgment.async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=1)],
        example=example,
        model="gpt-4o-mini"
    )
    return ans

# REPLACE THIS WITH YOUR OWN TOOLS
def check_opening_hours(restaurant: str, state: State) -> str:
    """Check opening hours for a specific restaurant"""
    ans = f"{restaurant} hours: Mon-Sun 11AM-10PM"
    example = Example(
        input="Check opening hours for a specific restaurant",
        actual_output=ans,
        expected_output=ans
    )
    judgment.async_evaluate(
        scorers=[AnswerCorrectnessScorer(threshold=1)],
        example=example,
        model="gpt-4o-mini"
    )
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
        model="gpt-4o-mini"
    )
    return ans 

# @judgment.observe(name="run_agent", span_type="Main Function")
def run_agent(prompt: str):
    tools = [
        TavilySearchResults(max_results=2),
        check_opening_hours,
        get_menu_items,
        search_restaurants,
    ]

    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

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

    handler = JudgevalCallbackHandler(judgment)
    set_global_handler(handler)

    result = graph.invoke({
        "messages": [HumanMessage(content=prompt)]
    })

    return result, handler

if __name__ == "__main__":
    result, handler = run_agent("Find me a good Italian restaurant in Manhattan. Check their opening hours and most popular dishes.")
    # print(result)
    
    print("Executed Node-Tools:")
    print(handler.executed_node_tools)