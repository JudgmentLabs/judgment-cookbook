import os
import re
import requests
import asyncio
import json
from typing import TypedDict, List
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient
from langchain.schema import Document
from openai import OpenAI

from judgeval.common.tracer import Tracer, wrap
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer

load_dotenv()

client = wrap(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
judgment = Tracer(
    api_key=os.getenv("JUDGMENT_API_KEY"),
    organization_id=os.getenv("JUDGMENT_ORG_ID"),
    project_name="anime_chatbot"
)

# Setup Chroma and embeddings
chroma_client = chromadb.Client()
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)
collection = chroma_client.get_or_create_collection(
    name="anime_data",
    embedding_function=embedding_fn
)

# Define the shape of our state
class ChatState(TypedDict):
    query: str
    refined_query: str
    retrieved_info: List[str]  # Retrieved text from Chroma, Jikan, or web
    final_answer: str          # Final answer to show the user
    next_node: str             # Data source decision for routing
    attempt_count: int         # Number of attempts made
    retry_flag: bool           # Flag indicating whether to retry
    node_decision: str         # Stores the chosen data source

# Node Functions
    
@judgment.observe(span_type="LLM decision")
def decision_node(state: ChatState) -> ChatState:
    """
    Select the best data source for answering an anime query and refine the query if necessary.
    
    If a previous attempt failed (retry_flag True), include feedback in the prompt.
    The LLM returns a JSON with keys 'chosen_node' and 'refined_query'.
    """
    state["attempt_count"] += 1
    query = state["query"]
    refined_query = state["refined_query"]

    feedback = ""
    if state.get("retry_flag", False):
        feedback = (f"Previously, you chose to search {state['node_decision']} using refined_query: {refined_query} failed. "
                    f"Please do not choose {state['node_decision']} and use a different query. Previous attempt feedback: {state['final_answer']}. "
                    "Incorporate some of these keywords to refine the query.")

    prompt = (
        "You have three available data sources:\n"
        "1. 'vector': A Chroma vector database populated with the top 300 anime information.\n"
        "2. 'jikan': The Jikan API that returns detailed information about a specific anime (for summarization tasks).\n"
        "3. 'web': A web search tool that returns recent anime news articles.\n\n"
        "Based on the user's query, decide which data source is most likely to return useful results. "
        "If the query might not yield good results from that source, provide a refined version of the query "
        "that is more specific or likely to produce results.\n\n"
        "For example, the query 'recommend me an anime that prominently swords' is poor because irrelevant keywords like "
        "'recommend' may be considered, especially when querying the vector database. "
        "If a user asks for suggestions or similar anime based on thematic content, choose 'vector'. "
        "Also, overly detailed queries for the Jikan API may yield poor results due to extraneous wording.\n\n"
        f"User Query: {query}\n\n"
        f"{feedback}\n\n"
        "Return your answer in JSON format with exactly two keys:\n"
        "  - 'chosen_node': one of 'vector', 'jikan', or 'web'\n"
        "  - 'refined_query': the refined version of the user's query\n\n"
        "Ensure the JSON is valid and contains only these two keys."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that selects the best data source for answering anime queries and refines queries when necessary."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        chosen_node = parsed.get("chosen_node", "web").strip().lower()
        refined_query = parsed.get("refined_query", query).strip()
    except Exception as e:
        print(f"DecisionNode: Error parsing LLM response: {e}")
        chosen_node = "web"
        refined_query = query

    state["next_node"] = chosen_node
    state["refined_query"] = refined_query
    state["node_decision"] = chosen_node
    print(f"DecisionNode: Chosen node: {chosen_node}, Refined query: {refined_query}")

    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        input=prompt,
        actual_output=content,
        model="gpt-4",
    )

    return state

@judgment.observe(span_type="retriever")
def anime_vector_node(state: ChatState) -> ChatState:
    """
    Perform a similarity search using the Chroma vector store and populate retrieved_info.
    """
    query = state["query"]
    # print("AnimeRecommendationNode: vector search for recommendations.")
    try:
        results = collection.query(query_texts=[query], n_results=3)
        docs = results.get("documents", [[]])[0]
        if not docs:
            state["retrieved_info"] = ["No similar anime found for your request."]
        else:
            state["retrieved_info"] = [f"RECOMMEND DOC: {d[:300]}" for d in docs]
    except Exception as e:
        state["retrieved_info"] = [f"Error retrieving from DB: {e}"]
    

    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        input=query,
        actual_output=state["retrieved_info"],
        model="gpt-4",
    )
    return state

@judgment.observe(span_type="API call")
def anime_jikan_node(state: ChatState) -> ChatState:
    """
    Fetch detailed anime information from the Jikan API using the query.
    Considers the first 10 results and populates retrieved_info.
    """
    query = state["query"]
    # print("AnimeDetailNode: fetching info from Jikan.")
    url = f"https://api.jikan.moe/v4/anime?q={query}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        anime_data = data.get("data", [])
        if not anime_data:
            state["retrieved_info"] = [f"No anime details found for '{query}'."]
            return state

        results = []
        for anime in anime_data[:10]:
            title = anime.get("title", "Unknown Title")
            synopsis = anime.get("synopsis", "No synopsis available.")
            combined = f"Title: {title}\nSynopsis: {synopsis}"
            results.append(combined)
        state["retrieved_info"] = results
    except Exception as e:
        state["retrieved_info"] = [f"Error fetching details for '{query}': {str(e)}"]
    
    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        input=query,
        actual_output=state["retrieved_info"],
        model="gpt-4",
    )
    return state

@judgment.observe(span_type="web search")
def anime_web_node(state: ChatState) -> ChatState:
    """
    Fetch recent anime news articles using the Tavily web search tool.
    """
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

    query = state.get("refined_query", state["query"])
    try:
        web_search_tool = TavilySearchResults(k=10, tavily_api_key=TAVILY_API_KEY)
        results = web_search_tool.invoke({"query": query})
        docs = []
        for result in results:
            content = result.get("content", "")
            title = result.get("title", "No Title")
            docs.append(f"{title}: {content[:500]}")
        if not docs:
            state["retrieved_info"] = [f"No news articles found for '{query}'."]
        else:
            state["retrieved_info"] = docs
        
        judgment.get_current_trace().async_evaluate(
            scorers=[AnswerRelevancyScorer(threshold=0.5)],
            input=query,
            actual_output=state["retrieved_info"],
            model="gpt-4",
        )
    except Exception as e:
        state["retrieved_info"] = [f"Error retrieving news: {e}"]
    return state

@judgment.observe(span_type="LLM evaluation")
def finalize_answer_node(state: ChatState) -> ChatState:
    """
    Evaluate the retrieved information using GPT and determine if it is sufficient.
    If sufficient, return a comprehensive answer. Otherwise, return suggested keywords.
    
    Expects a JSON output with keys:
      - "status": "sufficient" or "insufficient"
      - "final_answer": (if sufficient)
      - "keywords": (if insufficient)
    """
    MAX_ATTEMPTS = 2
    query = state["query"]
    retrieved_info = state["retrieved_info"]
    
    prompt = (
        "You are a helpful assistant tasked with evaluating retrieved information for an anime query. "
        "Please provide a comprehensive final answer to the query using the information provided. "
        "If the retrieved information is insufficient, please also provide a comma-separated list of keywords "
        "that, if added to the query, would yield better results.\n\n"
        f"User Query: {query}\n\n"
        "Retrieved Information:\n" + "\n".join(retrieved_info) + "\n\n"
        "Return your answer in JSON format with exactly the following keys:\n"
        '  "status": either "sufficient" or "insufficient",\n'
        '  "final_answer": "your comprehensive answer using the retrieved information",\n'
        '  "keywords": "a comma-separated list of suggested keywords if the information is insufficient" (optional).\n'
        "Ensure the JSON is valid and contains only these keys."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that evaluates retrieved information and provides comprehensive answers."
                },
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content.strip()

        parsed = json.loads(content)
        status = parsed.get("status", "sufficient").lower()
        final_answer = parsed.get("final_answer", "")
        if status == "insufficient" and state.get("attempt_count", 0) < MAX_ATTEMPTS:
            state["final_answer"] = f"Retrieved information insufficient. Suggested keywords: {parsed.get('keywords', '')}\n"
            state["retry_flag"] = True
        else:
            state["final_answer"] = final_answer
            state["retry_flag"] = False
    except Exception as e:
        state["final_answer"] = f"Error generating final answer: {e}"
        state["retry_flag"] = True

    judgment.get_current_trace().async_evaluate(
        scorers=[FaithfulnessScorer(threshold=0.5), AnswerRelevancyScorer(threshold=0.5)],
        input=prompt,
        actual_output=content,
        retrieval_context=retrieved_info,
        model="gpt-4",
    )

    return state

# Build the Graph
graph_builder = StateGraph(ChatState)
graph_builder.add_node("decision", decision_node)
graph_builder.add_node("vector", anime_vector_node)
graph_builder.add_node("jikan", anime_jikan_node)
graph_builder.add_node("web", anime_web_node)
graph_builder.add_node("finalize", finalize_answer_node)

# Graph edges
graph_builder.add_edge(START, "decision")

def route_from_decision(state: ChatState) -> str:
    return state["next_node"]

graph_builder.add_conditional_edges(
    "decision",
    route_from_decision,
    {"vector": "vector", "jikan": "jikan", "web": "web"}
)

def route_from_finalize(state: ChatState) -> str:
    max_attempts = 2
    if state.get("retry_flag", False) and state.get("attempt_count", 0) < max_attempts:
        print("Final answer unsatisfactory. Retrying with updated query...")
        return "decision"
    else:
        return END

graph_builder.add_conditional_edges(
    "finalize",
    route_from_finalize,
    {"decision": "decision", END: END}
)

graph_builder.add_edge("vector", "finalize")
graph_builder.add_edge("jikan", "finalize")
graph_builder.add_edge("web", "finalize")
graph_builder.add_edge("finalize", END)

memory_saver = MemorySaver()
graph = graph_builder.compile(checkpointer=memory_saver)

def fetch_top_anime(total=350):
    """
    Fetch top anime from the Jikan API.
    """
    base_url = "https://api.jikan.moe/v4/top/anime"
    per_page = 25
    pages = (total + per_page - 1) // per_page
    all_data = []
    for p in range(1, pages + 1):
        try:
            params = {"page": p, "limit": per_page}
            r = requests.get(base_url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            anime_list = data.get("data", [])
            if not anime_list:
                break
            all_data.extend(anime_list)
        except Exception as e:
            # print(f"Error fetching top anime page {p}: {e}")
            pass
    return all_data

def populate_vector_db(coll, anime_list):
    """
    Populate the Chroma collection with detailed anime information.
    """
    docs, metas, ids = [], [], []
    seen_ids = set()
    for item in anime_list:
        mal_id = item.get("mal_id")
        if mal_id in seen_ids:
            continue
        seen_ids.add(mal_id)
        title = item.get("title", "")
        synopsis = item.get("synopsis", "")
        type_ = item.get("type", "Unknown")
        episodes = item.get("episodes", "N/A")
        score = item.get("score", "N/A")
        rank = item.get("rank", "N/A")
        popularity = item.get("popularity", "N/A")

        if rank is None:
            rank = "N/A"
        if episodes is None:
            episodes = "N/A"
        
        genres = []
        if "genres" in item and isinstance(item["genres"], list):
            genres = [g.get("name", "") for g in item["genres"] if g.get("name")]
        genres_str = ", ".join(genres) if genres else "N/A"
        
        combined = (
            f"Title: {title}\n"
            f"Synopsis: {synopsis}\n"
            f"Type: {type_}\n"
            f"Episodes: {episodes}\n"
            f"Score: {score}\n"
            f"Rank: {rank}\n"
            f"Popularity: {popularity}\n"
            f"Genres: {genres_str}"
        )
        docs.append(combined)
        meta = {
            "title": title,
            "mal_id": mal_id,
            "type": type_,
            "episodes": episodes,
            "score": score,
            "rank": rank,
            "popularity": popularity,
            "genres": genres_str,
        }
        metas.append(meta)
        ids.append(str(mal_id))
    if docs:
        try:
            coll.add(documents=docs, metadatas=metas, ids=ids)
            print(f"Populated {len(docs)} anime items into Chroma collection.")
        except Exception as e:
            print("Error inserting into Chroma:", e)
    else:
        print("No anime records to add.")


@judgment.observe(span_type="Main Function", overwrite=True)
async def main():
    top_anime = fetch_top_anime(total=350)
    populate_vector_db(collection, top_anime)

    print("=== Basic LangGraph Anime Chatbot ===")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        init_state: ChatState = {
            "query": user_input,
            "refined_query": user_input,
            "retrieved_info": [],
            "final_answer": "",
            "next_node": "",
            "attempt_count": 0,
            "retry_flag": False,
            "node_decision": ""
        }

        results = graph.invoke(
            init_state,
            config={"configurable": {"thread_id": "my_unique_conversation_id"}}
        )
        final_answer = results["final_answer"]
        print("Assistant:", final_answer, "\n")

if __name__ == "__main__":
    asyncio.run(main())
