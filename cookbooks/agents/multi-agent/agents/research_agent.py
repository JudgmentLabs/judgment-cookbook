import json
from subagent import AgentBase, ToolsMixin
from tools import ResearchToolsMixin
from judgeval.scorers import FaithfulnessScorer
from judgeval.data import Example
from tools.common import judgment

SYSTEM_PROMPT = '''
You are a research agent specialized in gathering and analyzing information. Your job is to research topics thoroughly using the following tools:

web_search: Search the web for information on a given query
  - query: The search query to look up
  - NOTE: If web_search returns an error or empty result, retry with a modified query

extract_facts: Extract key facts and statistics from text
  - text: The text to analyze and extract facts from

summarize_sources: Summarize information from multiple sources
  - sources: List of source texts to summarize

evaluate_completeness: Assess if gathered information is sufficient
  - information: The information gathered so far
  - original_query: The original research question

YOU MUST use these tools in sequence for every research query:
1. ALWAYS start with web_search to get initial information
   - If web_search fails:
    a. retry with modified query
2. ALWAYS use extract_facts on the search results
3. ALWAYS use evaluate_completeness to check if you have enough information
4. If evaluation shows INSUFFICIENT:
   - Look at the NEXT_SEARCH suggestions
   - Do another web_search with those suggestions
   - Repeat steps 2-3
5. Once evaluation shows SUFFICIENT:
   - Use summarize_sources to create the final summary

IMPORTANT:
- NEVER output tool calls in your final summary
- ALWAYS execute tools to get actual results
- ALWAYS analyze the results before proceeding
- ALWAYS provide a proper research summary at the end
- ALWAYS handle web_search errors by retrying with modified queries
- If web_search consistently fails, note this in your final summary

Format your tool calls as:
<tool>
{"name": "tool_name", "args": {"parameter": "value"}}
</tool>

Example of proper tool sequence with error handling:
1. First tool call:
<tool>
{"name": "web_search", "args": {"query": "your search query"}}
</tool>

2. If web_search fails, retry with modified query:
<tool>
{"name": "web_search", "args": {"query": "modified search query"}}
</tool>

3. After getting results, extract facts:
<tool>
{"name": "extract_facts", "args": {"text": "search results text"}}
</tool>

4. Evaluate completeness:
<tool>
{"name": "evaluate_completeness", "args": {"information": "extracted facts", "original_query": "original query"}}
</tool>

5. If sufficient, summarize:
<tool>
{"name": "summarize_sources", "args": {"sources": ["extracted facts"]}}
</tool>

Remember: 
- Execute tools in sequence
- Handle web_search errors by retrying with modified queries
- Analyze results before proceeding
- Provide a proper research summary at the end
- Note any persistent search failures in your final summary
'''

@judgment.identify(identifier="name", track_state=True)
class ResearchAgent(ToolsMixin, ResearchToolsMixin, AgentBase):
    def __init__(self, client=None, model="gpt-4.1", name="ResearchAgent"):
        super().__init__(client, model)
        self.name = name
        self.function_map = {
            "web_search": self.web_search,
            "extract_facts": self.extract_facts,
            "summarize_sources": self.summarize_sources,
            "evaluate_completeness": self.evaluate_completeness
        }

    def _execute_function(self, function_name: str, args: dict) -> str:
        """Execute a function with the given arguments and return string result."""
        if function_name not in self.function_map:
            return f"Error: Unknown function '{function_name}'"
        
        try:
            func = self.function_map[function_name]
            # Call function directly with arguments (no dict wrapper)
            if function_name == "web_search":
                result = func(args.get("query", ""), args.get("num_results", 5))
            elif function_name == "extract_facts":
                result = func(args.get("text", ""))
            elif function_name == "summarize_sources":
                result = func(args.get("sources", []))
            elif function_name == "evaluate_completeness":
                result = func(args.get("information", ""), args.get("original_query", ""))
            else:
                return f"Error: Unsupported function '{function_name}'"
            
            return str(result)
        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"
        
    @judgment.observe(span_type="function")
    def process_request(self, research_query):
        print(f"\n[RESEARCH AGENT] Starting research on: {research_query[:100]}...")
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": research_query}
        ]
        
        max_steps = 20
        step = 0
        
        while step < max_steps:
            # Get response from model
            response = self._call_model(messages, [])  # No tools needed for this approach
            
            # Extract content from response
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Add assistant response to messages
            messages.append({"role": "assistant", "content": content})
            
            # Check for tool call
            tool_name, tool_args = self._parse_tool_call(content)
            
            if tool_name:
                # Execute the tool
                print(f"\n[RESEARCH] 🔍 Tool call: {tool_name}({tool_args})")
                result = self._execute_function(tool_name, tool_args)
                
                # Show evaluation results for completeness checks
                if tool_name == "evaluate_completeness":
                    print(f"\n[RESEARCH] 📊 Evaluation Result:")
                    print(f"{result}")
                    if "INSUFFICIENT" in result:
                        print(f"[RESEARCH] 🔄 Information gap detected - will search for more data")
                    else:
                        print(f"[RESEARCH] ✅ Information appears sufficient")
                else:
                    print(f"\n[RESEARCH] 📝 Tool result:")
                    print(f"{result}")
                
                # Add tool result as user message
                tool_response = f"<result>{result}</result>"
                messages.append({"role": "user", "content": tool_response})
                
                step += 1
            else:
                # No tool call found, this should be the final answer
                print(f"\n[RESEARCH] ✅ Research complete. Final summary:")
                print(f"{content.strip()}")
                
                example = Example(
                    input=research_query,
                    actual_output=content.strip(),
                    retrieval_context=[str(s) for s in messages]
                )
                judgment.async_evaluate(
                    scorers=[FaithfulnessScorer(threshold=1.0)],
                    example=example,
                    model="gpt-4.1-mini"
                )
                return content.strip()
        
        return "Error: Maximum steps exceeded" 