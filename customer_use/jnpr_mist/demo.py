from typing import Annotated, List
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from judgeval.common.tracer import Tracer, wrap, JudgevalCallbackHandler
import os
from judgeval.data import Example
from judgeval.data.datasets import EvalDataset
from judgeval.scorers import AnswerRelevancyScorer, ExecutionOrderScorer, AnswerCorrectnessScorer
from judgeval import JudgmentClient

PROJECT_NAME = "JNPR_MIST_LANGGRAPH"

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"), project_name=PROJECT_NAME)


@judgment.observe(name="search_restaurants", span_type="tool")
def search_restaurants(location: str, cuisine: str, state: State) -> str:
    """Search for restaurants in a location with specific cuisine"""
    ans = f"Top 3 {cuisine} restaurants in {location}: 1. Le Gourmet 2. Spice Palace 3. Carbones"
    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=1)],
        input="Search for restaurants in a location with specific cuisine",
        actual_output=ans,
        model="gpt-4o-mini"
    )
    return ans

@judgment.observe(name="check_opening_hours", span_type="tool")
def check_opening_hours(restaurant: str, state: State) -> str:
    """Check opening hours for a specific restaurant"""
    ans = f"{restaurant} hours: Mon-Sun 11AM-10PM"
    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerCorrectnessScorer(threshold=1)],
        input="Check opening hours for a specific restaurant",
        actual_output=ans,
        expected_output=ans,
        model="gpt-4o-mini"
    )
    return ans

@judgment.observe(name="get_menu_items", span_type="tool")
def get_menu_items(restaurant: str) -> str:
    """Get popular menu items for a restaurant"""
    ans = f"{restaurant} popular dishes: 1. Chef's Special 2. Seafood Platter 3. Vegan Delight"
    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=1)],
        input="Get popular menu items for a restaurant",
        actual_output=ans,
        model="gpt-4o-mini"
    )
    return ans 


@judgment.observe(span_type="Run Agent", overwrite=True)
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

    handler = JudgevalCallbackHandler(judgment.get_current_trace())

    result = graph.invoke({
        "messages": [HumanMessage(content=prompt)]
    }, config=dict(callbacks=[handler]))

    print("\nFinal Result:")
    for msg in result["messages"]:
        print(f"{type(msg).__name__}: {msg.content}")
    
    return handler
    

def test_eval_dataset():
    dataset = EvalDataset()

    # Helper to configure tests with YAML
    dataset.add_from_yaml(os.path.join(os.path.dirname(__file__), "test.yaml"))
    
    for example in dataset.examples:
        # Run your agent here
        handler = run_agent(example.input)
        example.actual_output = handler.executed_node_tools

    client = JudgmentClient()
    client.run_evaluation(
        examples=dataset.examples,
        scorers=[ExecutionOrderScorer(threshold=1, should_consider_ordering=True)],
        model="gpt-4o-mini",
        project_name=PROJECT_NAME,
        eval_run_name="mist-demo-examples"
    )


if __name__ == "__main__":
    test_eval_dataset()