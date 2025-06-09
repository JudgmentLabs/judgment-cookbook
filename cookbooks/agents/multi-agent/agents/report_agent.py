import json
from subagent import AgentBase, ToolsMixin
from tools import ReportToolsMixin
from judgeval.scorers import FaithfulnessScorer
from judgeval.data import Example
from tools.common import judgment

SYSTEM_PROMPT = '''
You are a report generation agent specialized in creating structured documents and summaries. Your job is to format information into professional reports using the following tools:

format_report: Format content into a structured report
  - title: The title of the report
  - content: The main content to format
  - sections: Optional list of section names

create_executive_summary: Create an executive summary from findings
  - findings: List of key findings
  - recommendations: Optional list of recommendations

generate_charts_description: Generate descriptions for charts and visualizations
  - data: Dictionary containing chart data

Use these tools to create professional reports:
1. Use format_report to structure information
2. Use create_executive_summary for high-level overviews
3. Use generate_charts_description for data visualizations

You must format your output as tool calls with this exact format, you cannot use <format_report>, <create_executive_summary>, or <generate_charts_description> tags, must use the following format:
<tool>
{"name": "tool_name", "args": {"parameter": "value"}}
</tool>

Generate comprehensive, well-formatted reports after using the appropriate tools.
'''


@judgment.identify(identifier="name", track_state=True)
class ReportAgent(ToolsMixin, ReportToolsMixin, AgentBase):
    def __init__(self, client=None, model="gpt-4.1", name="ReportAgent"):
        super().__init__(client, model)
        self.name = name
        self.function_map = {
            "format_report": self.format_report,
            "create_executive_summary": self.create_executive_summary,
            "generate_charts_description": self.generate_charts_description
        }

    def _execute_function(self, function_name: str, args: dict) -> str:
        """Execute a function with the given arguments and return string result."""
        if function_name not in self.function_map:
            return f"Error: Unknown function '{function_name}'"
        
        try:
            func = self.function_map[function_name]
            # Call function directly with arguments (no dict wrapper)
            if function_name == "format_report":
                result = func(args.get("title", ""), args.get("content", ""), args.get("sections", None))
            elif function_name == "create_executive_summary":
                result = func(args.get("findings", []), args.get("recommendations", None))
            elif function_name == "generate_charts_description":
                result = func(args.get("data", {}))
            else:
                return f"Error: Unsupported function '{function_name}'"
            
            return str(result)
        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"

    @judgment.observe(span_type="agent")
    def process_request(self, report_request):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": report_request}
        ]
        
        max_steps = 100
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
                print(f"[DEBUG] Tool call: {tool_name}({tool_args})")
                result = self._execute_function(tool_name, tool_args)
                print(f"[DEBUG] Tool result: {result}")
                
                # Add tool result as user message
                tool_response = f"<result>{result}</result>"
                messages.append({"role": "user", "content": tool_response})
                
                step += 1
            else:
                # No tool call found, this should be the final answer
                example = Example(
                    input=report_request,
                    actual_output=content.strip(),
                    retrieval_context=[str(s) for s in messages]
                )
                judgment.async_evaluate(
                    scorers=[FaithfulnessScorer(threshold=1.0)],
                    example=example,
                    model="gpt-4.1"
                )
                return content.strip()
        
        return "Error: Maximum steps exceeded" 