from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
import tiktoken

class AgentBase(ABC):
    """Base class for all agents providing common LLM interaction functionality."""
    
    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        
    def _call_model(self, messages: List[Dict], tools: List[Dict] = None) -> Any:
        """Call the LLM with messages and optional tools."""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error calling model: {e}")
            return None

    def _parse_tool_call(self, content: str) -> tuple:
        """Parse tool call and plan from content. Returns (tool_name, args, plan) or (None, None, None)."""
        import re
        
        plan_match = re.search(r'<plan>(.*?)</plan>', content, re.DOTALL)
        plan = plan_match.group(1).strip() if plan_match else None
        
        tool_pattern = r'<tool>\s*(.*?)\s*</tool>'
        match = re.search(tool_pattern, content, re.DOTALL)
        
        if match:
            try:
                tool_json = json.loads(match.group(1))
                tool_name = tool_json.get("name")
                tool_args = tool_json.get("args", {})
                return tool_name, tool_args, plan
            except json.JSONDecodeError:
                return None, None, plan
        
        return None, None, plan

    def _execute_function(self, function_name: str, args: dict) -> str:
        """Execute a function with the given arguments and return string result."""
        if function_name not in self.function_map:
            return f"Error: Unknown function '{function_name}'"
        
        try:
            func = self.function_map[function_name]
            result = func(**args)
            return str(result)
        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"
        
    def _count_tokens(self, messages):
        """Count actual tokens using OpenAI's tokenizer."""
        
        encoding = tiktoken.encoding_for_model("gpt-4")
        total_tokens = 0
        for msg in messages:
            total_tokens += len(encoding.encode(msg.get('content')))
        return total_tokens

    @abstractmethod
    def process_request(self, request: str) -> str:
        """Process a request and return the response. Must be implemented by subclasses."""
        pass 