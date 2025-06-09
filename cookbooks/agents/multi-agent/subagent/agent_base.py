from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json

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
            return response.choices[0].message
            
        except Exception as e:
            print(f"Error calling model: {e}")
            return None
    
    @abstractmethod
    def process_request(self, request: str) -> str:
        """Process a request and return the response. Must be implemented by subclasses."""
        pass 