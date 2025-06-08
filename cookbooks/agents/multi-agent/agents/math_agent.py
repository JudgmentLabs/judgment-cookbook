import json
from subagent import AgentBase, ToolsMixin
from tools import MathToolsMixin
from judgeval.scorers import FaithfulnessScorer
from judgeval.data import Example
from tools.common import judgment, client

SYSTEM_PROMPT = '''
You are a mathematical problem-solving agent. Your job is to solve mathematical problems and word problems using the following tools:

formalize: Converts a word problem into a math equation
  - problem: The word problem text to convert

format: Converts a math equation into executable code  
  - equation: The math equation to convert to code

calculate: Evaluates code and returns the numerical answer
  - code: The code expression to evaluate

You MUST use these tools in sequence for every problem:
1. Use formalize to convert the word problem to an equation
2. Use format to convert the equation to executable code  
3. Use calculate to get the final numerical answer

Format your tool calls as:
<tool>
{"name": "tool_name", "args": {"parameter": "value"}}
</tool>

Only provide the final answer after completing all tool steps.
'''

@judgment.identify(identifier="name", track_state=True)
class MathAgent(ToolsMixin, MathToolsMixin, AgentBase):
    def __init__(self, client=None, model="gpt-4.1", name="MathAgent"):
        super().__init__(client, model)
        self.name = name
        self.function_map = {
            "formalize": self.formalize,
            "format": self.format,
            "calculate": self.calculate
        }

    def _execute_function(self, function_name: str, args: dict) -> str:
        """Execute a function with the given arguments and return string result."""
        if function_name not in self.function_map:
            return f"Error: Unknown function '{function_name}'"
        
        try:
            func = self.function_map[function_name]
            # Call function directly with arguments (no dict wrapper)
            if function_name == "formalize":
                result = func(args.get("problem", ""))
            elif function_name == "format":
                result = func(args.get("equation", ""))
            elif function_name == "calculate":
                result = func(args.get("code", ""))
            else:
                return f"Error: Unsupported function '{function_name}'"
            
            return str(result)
        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"
    
    @judgment.observe(span_type="agent")
    def process_request(self, math_problem):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": math_problem}
        ]
        
        max_steps = 100
        step = 0
        
        while step < max_steps:
            # Get response from model
            response = self._call_model(messages, [])
            
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
                print(f"[MATH] Tool call: {tool_name}({tool_args})")
                result = self._execute_function(tool_name, tool_args)
                print(f"[MATH] Tool result: {result}")
                
                # Add tool result as user message
                tool_response = f"<result>{result}</result>"
                messages.append({"role": "user", "content": tool_response})
                
                step += 1
            else:
                # No tool call found, this should be the final answer
                
                return content.strip()
        
        return "Error: Maximum steps exceeded" 
