import json
import os
from openai import OpenAI
from tavily import TavilyClient
import requests
from subagent.agent_base import AgentBase

SYSTEM_PROMPT_RESEARCH = '''
You are a research assistant. Your goal is to answer the user's query based on information from web searches and news sources.
You have access to the following tools:
- web_search: Use this tool to find relevant information online. You must provide a "search_term" parameter.
- news_search: Use this tool to find recent news articles on a specific topic. You must provide a "topic" parameter and optionally a "days" parameter (integer).

The current memory state might provide useful context:
<BEGIN MEMORY {memory_context} END MEMORY>

Based on the user's research query and the memory context, call the appropriate tool with the best possible search parameters to find the answer.
You will be given the results, and then you must synthesize them into a concise and informative answer to the original user query.
'''


class ResearchAgent(AgentBase):
    """Handles the orchestration of the research sub-task using Tavily."""

    def __init__(self, client: OpenAI, model: str, memory):
        """Initialize the ResearchAgent.

        Args:
            client: The OpenAI client instance.
            model: The model name to use for LLM calls.
            memory: The memory object (from TravelAgent) for context.
        """
        super().__init__(client, model)
        self.memory = memory

        try:
            self.tavily_client = TavilyClient(
                api_key=os.getenv("TAVILY_API_KEY"))
        except Exception as e:
            print(
                f"Error initializing TavilyClient: {e}. Make sure TAVILY_API_KEY is set.")
            self.tavily_client = None

        self.newsapi_key = os.getenv("NEWSAPI_KEY")
        if not self.newsapi_key:
            print("Warning: NEWSAPI_KEY environment variable not set")

        all_tools = self._load_tools_from_json()

        self.web_search_tool_def = next(
            (tool for tool in all_tools if tool.get('name') == 'web_search'),
            None
        )

        self.news_search_tool_def = next(
            (tool for tool in all_tools if tool.get('name') == 'news_search'),
            None
        )

        if not self.web_search_tool_def:
            print("Warning: web_search tool not found in toolkit.json")

        if not self.news_search_tool_def:
            print("Warning: news_search tool not found in toolkit.json")
            
        self.function_map = {
            "web_search": self._execute_tavily_search,
            "news_search": self._execute_news_search
        }

    def _execute_tavily_search(self, args_dict):
        """Executes a search using the Tavily client."""
        if not self.tavily_client:
            return "Error: Tavily client not initialized. Check API key."

        if not args_dict or not isinstance(args_dict, dict):
            return "Error: Invalid arguments format for web_search"

        query = args_dict.get("search_term")
        if not query:
            return "Error: 'search_term' parameter is required for web_search."

        print(f"--- ResearchAgent: Executing Tavily Search for: {query} ---")
        try:
            print(f"🔍 Running web_search with query: '{query}'")
            response = self.tavily_client.search(
                query=query, search_depth="basic", max_results=5)

            results = response.get("results", [])
            formatted_results = "\n".join([
                f"Title: {res.get('title', 'N/A')}\nURL: {res.get('url', 'N/A')}\nSnippet: {res.get('content', 'N/A')}\n---"
                for res in results
            ])
            if not formatted_results:
                print(f"⚠️ No results found for query '{query}'")
                return "No results found by Tavily."
            print(f"✅ web_search complete - Found {len(results)} results")
            return formatted_results

        except Exception as e:
            print(f"❌ Error during Tavily search for '{query}': {e}")
            return f"Error performing web search via Tavily: {e}"

    def _execute_news_search(self, args_dict):
        """Executes a news search using the NewsAPI."""
        if not self.newsapi_key:
            return "Error: NewsAPI key not initialized. Check NEWSAPI_KEY environment variable."

        if not args_dict or not isinstance(args_dict, dict):
            return "Error: Invalid arguments format for news_search"

        topic = args_dict.get("topic")
        if not topic:
            return "Error: 'topic' parameter is required for news_search."

        try:
            days = int(args_dict.get("days", 7))
            if days <= 0:
                days = 7  
        except (ValueError, TypeError):
            days = 7
            print(
                f"⚠️ Invalid 'days' parameter: {args_dict.get('days')}. Using default: 7")

        print(f"--- ResearchAgent: Executing News Search for: {topic} ---")
        try:
            from datetime import datetime, timedelta
            from_date = (datetime.now() - timedelta(days=days)
                         ).strftime('%Y-%m-%d')

            print(
                f"📰 Running news_search with topic: '{topic}', looking back {days} days")
            url = f"https://newsapi.org/v2/everything"
            params = {
                "q": topic,
                "from": from_date,
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": self.newsapi_key,
                "pageSize": 10
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            articles = data.get("articles", [])
            if not articles:
                print(f"⚠️ No news articles found for topic '{topic}'")
                return f"No news articles found for topic '{topic}' in the last {days} days."

            formatted_results = "\n".join([
                f"Title: {article.get('title', 'N/A')}\nSource: {article.get('source', {}).get('name', 'N/A')}\n"
                f"Published: {article.get('publishedAt', 'N/A')}\nURL: {article.get('url', 'N/A')}\n"
                f"Description: {article.get('description', 'N/A')}\n---"
                for article in articles
            ])

            print(f"✅ news_search complete - Found {len(articles)} articles")
            return formatted_results

        except Exception as e:
            print(f"❌ Error during NewsAPI search for '{topic}': {e}")
            return f"Error performing news search via NewsAPI: {e}"

    def _process_tool_calls(self, response, messages):
        """Processes tool calls within the ResearchAgent context."""
        processed_calls = False
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'type') and item.type == "function_call":
                    tool_name = getattr(item, 'name', 'unknown_tool')

                    print(f"\n📋 EXECUTING TOOL CALL: {tool_name}")
                    print(f"📋 ARGUMENTS: {getattr(item, 'arguments', 'None')}")

                    if tool_name in self.function_map:
                        processed_calls = True
                        messages.append(item)

                        args_dict = self._parse_args(item.arguments)

                        if args_dict:
                            try:
                                result = self.function_map[tool_name](
                                    args_dict)
                            except Exception as e:
                                error_msg = f"Error executing {tool_name}: {str(e)}"
                                print(f"❌ {error_msg}")
                                result = error_msg
                        else:
                            error_msg = f"Error: Invalid arguments format for {tool_name}"
                            print(f"❌ {error_msg}")
                            result = error_msg

                        result_summary = str(result)
                        if len(result_summary) > 200:
                            result_summary = result_summary[:200] + \
                                "... [truncated]"
                        print(f"📋 RESULT: {result_summary}")

                        messages.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": str(result)
                        })
                    else:
                        print(
                            f"❌ ResearchAgent Warning: Unexpected tool call requested: {tool_name}")
                        messages.append(item)
                        messages.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": f"Error: Tool '{tool_name}' is not available. Available tools are: {', '.join(self.function_map.keys())}"
                        })
        return processed_calls

    def perform_research(self, query: str):
        """Orchestrates the research process: request search, execute, synthesize."""
        print(f"--- ResearchAgent: Starting research for: {query} ---")
        research_context = self.memory.get_all_memory()

        research_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_RESEARCH.format(
                memory_context=research_context)},
            {"role": "user", "content": query}
        ]

        available_tools = []
        if self.web_search_tool_def:
            available_tools.append(self.web_search_tool_def)
        if self.news_search_tool_def:
            available_tools.append(self.news_search_tool_def)

        if not available_tools:
            print(
                "Warning: No tools available. Research may not work properly.")

        print("--- ResearchAgent: Requesting tool calls --- ")
        search_request_response = self._call_model(
            research_messages,
            available_tools
        )

        print("--- ResearchAgent: Processing potential tool calls --- ")
        processed_search = self._process_tool_calls(
            search_request_response, research_messages)

        if not processed_search:
            print("--- ResearchAgent: Model did not request any tool calls. ---")

        print("--- ResearchAgent: Synthesizing final answer --- ")
        research_messages.append({
            "role": "user",
            "content": "Based on the search results above, please provide a concise answer to my original research query."
        })
        final_research_response = self._call_model(research_messages, tools=[])

        final_text = getattr(final_research_response, 'output_text', None)
        if final_text:
            print("--- ResearchAgent: Returning synthesized answer ---")
            return final_text
        else:
            print("--- ResearchAgent: Error - No final text generated --- ")
            return "Sorry, I could perform the search but encountered an error synthesizing the final answer."

    def _parse_args(self, args_str):
        """Parse arguments from a string to a dictionary."""
        try:
            return json.loads(args_str)
        except json.JSONDecodeError:
            print(f"Error parsing arguments: {args_str}")
            return None
