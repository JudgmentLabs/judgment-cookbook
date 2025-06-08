import json
from abc import ABC
from typing import Dict, Any
from judgeval.common.tracer import Tracer

judgment = Tracer(project_name="multi-agent-system", deep_tracing=False)

class ToolsMixin(ABC):
    """Mixin providing base tool functionality for agents."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function_map: Dict[str, callable] = {}
    
    def _parse_tool_call(self, content: str) -> tuple:
        """Parse tool call from content. Returns (tool_name, args) or (None, None)."""
        import re
        
        # Look for <tool> tags
        tool_pattern = r'<tool>\s*(.*?)\s*</tool>'
        match = re.search(tool_pattern, content, re.DOTALL)
        
        if match:
            try:
                tool_json = json.loads(match.group(1))
                tool_name = tool_json.get("name")
                tool_args = tool_json.get("args", {})
                return tool_name, tool_args
            except json.JSONDecodeError:
                return None, None
        
        return None, None 