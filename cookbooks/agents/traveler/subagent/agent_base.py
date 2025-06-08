import json
from openai import OpenAI


class AgentBase:
    """Base class providing common functionality for all agent classes."""

    def __init__(self, client: OpenAI, model: str):
        """Initialize the base agent.

        Args:
            client: The OpenAI client instance.
            model: The model name to use for LLM calls.
        """
        self.client = client
        self.model = model

    def _load_tools_from_json(self, filepath='./toolkit.json'):
        """Load tool definitions from a JSON file.

        Args:
            filepath: Path to the JSON file containing tool definitions.

        Returns:
            List of tool definition dictionaries or empty list on error.
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: {filepath} not found.")
            return []
        except json.JSONDecodeError:
            print(f"Error: {filepath} is not valid JSON.")
            return []

    def _filter_tools(self, all_tools, allowed_tool_names):
        """Filter tool definitions by name.

        Args:
            all_tools: List of all tool definition dictionaries.
            allowed_tool_names: List of tool names to filter by.

        Returns:
            List of filtered tool definitions.
        """
        return [
            tool for tool in all_tools
            if tool.get('name') in allowed_tool_names
        ]

    def _call_model(self, messages, tools):
        """Call the OpenAI API with the given messages and tools.

        Args:
            messages: List of message dictionaries.
            tools: List of tool definition dictionaries.

        Returns:
            The OpenAI API response.
        """
        try:
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                tools=tools
            )
            return response
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise

    def _parse_args(self, arguments):
        """Parse tool call arguments to ensure they're in dictionary format.

        Args:
            arguments: The arguments from the function call (string or dict).

        Returns:
            Dictionary of parsed arguments or None on error.
        """
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not parse arguments string: {arguments}")
                return None
        elif isinstance(arguments, dict):
            return arguments
        else:
            print(f"Warning: Unexpected argument type: {type(arguments)}")
            return None
