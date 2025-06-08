from openai import OpenAI
import json
from tools import get_flights, get_hotels, get_car_rentals, get_things_to_do
import time
import threading
import sys
from memory import Memory
from subagent import ResearchAgent
from subagent.agent_base import AgentBase
from judgeval.common.tracer import Tracer, wrap
from judgeval.scorers import FaithfulnessScorer
from judgeval.data import Example

# --- System Prompts ---

SYSTEM_PROMPT_MEMORY = '''
You are an information extraction assistant. Your sole task is to identify and extract key pieces of information from the user's message that would be useful for a travel agent.
Focus on details like:
- user's name
- user's age
- user's travel preferences (dietary, interests, budget hints)
- user's travel dates
- origin / destination
- number of travelers
- any other explicit constraints or desires.

You ONLY have access to memory tools:
- add_memory: Use this to store newly learned information. Provide information as key-value pairs.
- update_memory: Use this if the user corrects or updates previously known information.

The current memory state is provided below. Use it to avoid re-adding existing information unless it's an update.
<BEGIN MEMORY {memory_context} END MEMORY>

Analyze the user's message. If you find *new* or *updated* information relevant to the categories above, invoke the corresponding function (`add_memory` or `update_memory`) using the provided tool-calling mechanism. Pass all the relevant pieces of information extracted from the user's message as arguments to the function call.
If no new or updated relevant information is found in the user message, do not call any tools and output nothing.
'''

SYSTEM_PROMPT_TRAVEL = '''
You are a helpful travel agent. Your goal is to create a travel itinerary based on the user's request and the information stored in memory.
You have access to the following tools to find travel options:
- get_flights
- get_hotels
- get_car_rentals
- get_things_to_do

Since, there may be requests by the user that cannot be done by just using the travel tools, you also have access to the research tool to find more general purpose information. As a travel agent, your goal is to maximize your helpfulness so using the research tool is strongly encouraged when you do not know the answer.
This is called:
- research

You also have access to memory tools for reference or if critical new details emerge during planning, but prioritize travel tools:
- add_memory
- get_memory
- update_memory
- get_all_memory

The current memory state is provided below. Use it to personalize the itinerary.
<BEGIN MEMORY> {memory_context} <END MEMORY>

Based on the user's request and the memory, use the travel tools to find suitable options (flights, hotels, etc.). Aim to generate 1-3 well-structured itinerary options.
Ensure that for round trips, the flight total cost is the sum of the departure and return flight costs.
Format the response clearly for terminal display, including costs and details as shown in the example below.
DO NOT include coordinates. Ensure the total summary cost is accurate.
DO NOT include any assistant prompting messages in your response (e.g. "Please specify...").

Example Format:
======================================================================
🧳 [Origin] to [Destination] | [Trip Type]: [Dates]
======================================================================
[Introductory text]
--------------------------
    🛩️ FLIGHT OPTIONS
--------------------------
Option 1: [Description]
--------------------------
💵 Total Cost      : $[Cost]
... [details] ...
--------------------------
Option 2: [Description]
... etc ...
[Include Hotel, Car Rental, Things To Do sections similarly if requested/applicable]
---------------------------
    💵 TOTAL COST SUMMARY (Example)
---------------------------
Flight + Hotel (Option X): $[Flight Cost] + $[Hotel Cost] = $[Total]
======================================================================

Now, analyze the user's request and the memory context. Determine ALL the tools you need to call (e.g., get_flights, get_hotels, get_car_rentals, get_things_to_do) to gather the necessary information for the itinerary.
Call ALL required tools *in parallel* in your response. Once you have the results from these tools in the conversation history, you will then formulate the final itinerary text.
'''

judgment = Tracer(
    project_name="traveler-test",
    deep_tracing=False
)


class TravelAgent(AgentBase):
    def __init__(self, client=None, model="gpt-4.1"):
        # Use provided client or initialize a new one if none provided
        if client is None:
            client = wrap(OpenAI())
        # Call the parent class constructor
        super().__init__(client, model)

        self.memory = Memory({"year": "2025"})
        self.research_agent = ResearchAgent(
            client=self.client,
            model=self.model,
            memory=self.memory
        )
        self.function_map = {
            "get_flights": get_flights,
            "get_hotels": get_hotels,
            "get_car_rentals": get_car_rentals,
            "get_things_to_do": get_things_to_do,
            "add_memory": self.memory.add_memory,
            "get_memory": self.memory.get_memory,
            "update_memory": self.memory.update_memory,
            "get_all_memory": self.memory.get_all_memory,
            "research": self._handle_research
        }

        # Load all tools and filter based on function names
        self.all_tools_json = self._load_tools_from_json()

        # Define tool name lists
        self.memory_tool_names = ["add_memory", "update_memory"]
        self.travel_tool_names = [
            "get_flights", "get_hotels", "get_car_rentals", "get_things_to_do", "research"]
        # combined_tool_names reflects tools exposed by TravelAgent
        self.combined_tool_names = list(self.function_map.keys())

        # Filter the loaded JSON based on names
        self.memory_tools_json = self._filter_tools(
            self.all_tools_json, self.memory_tool_names)
        self.combined_tools_json = self._filter_tools(
            self.all_tools_json, self.combined_tool_names)

    def _process_tool_calls(self, response, messages, allowed_tool_names):
        """Processes tool calls in a response, executes them, and updates messages."""
        processed_calls = False
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'type') and item.type == "function_call":
                    if item.name in allowed_tool_names:
                        processed_calls = True
                        messages.append(item)
                        # Log the tool being called
                        print(f"--- Executing Tool: {item.name} ---")
                        result = self._call_function(item)
                        messages.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": str(result)
                        })
                    else:
                        print(
                            f"Warning: Model tried to call tool '{item.name}' when only {allowed_tool_names} were expected.")
        return processed_calls

    def _call_function(self, function_call):
        """Execute a function based on a tool call."""
        args_dict = self._parse_args(function_call.arguments)
        if not args_dict:
            return f"Error: Invalid arguments format for {function_call.name}"

        function_name = function_call.name
        if function_name in self.function_map:
            try:
                return self.function_map[function_name](args_dict)
            except Exception as e:
                print(
                    f"Error calling function {function_name} with args {args_dict}: {e}")
                return f"Error executing function {function_name}: {e}"
        else:
            return f"No function found: {function_name}"

    def _spinner(self, stop_event):
        """Display a spinner animation in the terminal while processing."""
        spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        i = 0
        while not stop_event.is_set():
            sys.stdout.write(
                '\r' + spinner_chars[i % len(spinner_chars)] + ' Creating your travel itinerary...')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear the spinner line
        sys.stdout.flush()

    @judgment.observe(span_type="function")
    def process_request(self, user_message):
        """Process a user request through memory extraction and travel planning."""
        stop_spinner = threading.Event()
        spinner_thread = threading.Thread(
            target=self._spinner, args=(stop_spinner,))
        spinner_thread.daemon = True
        spinner_thread.start()

        try:
            initial_memory_context = self.memory.get_all_memory()
            memory_messages = [
                {"role": "system", "content": SYSTEM_PROMPT_MEMORY.format(
                    memory_context=initial_memory_context)},
                {"role": "user", "content": user_message}
            ]
            memory_response = self._call_model(
                memory_messages, self.memory_tools_json)
            self._process_tool_calls(
                memory_response, memory_messages, self.memory_tool_names)

            current_memory_context = self.memory.get_all_memory()

            travel_messages = [
                {"role": "system", "content": SYSTEM_PROMPT_TRAVEL.format(
                    memory_context=current_memory_context)}
            ] + memory_messages[1:]

            tool_request_response = self._call_model(
                travel_messages, self.combined_tools_json)
            self._process_tool_calls(
                tool_request_response, travel_messages, self.combined_tool_names)

            travel_messages.append({
                "role": "user",
                "content": "Now, please generate the final travel itinerary in the required format, summarizing all the information and tool results gathered above."
            })
            final_response = self._call_model(travel_messages, tools=[])

            example = Example(
                input=user_message,
                actual_output=final_response.output_text,
                retrieval_context=[str(s) for s in travel_messages]
            )
            judgment.async_evaluate(
                scorers=[FaithfulnessScorer(threshold=1.0)],
                example=example,
                model="gpt-4.1"
            )

            output_text = getattr(final_response, 'output_text', None)
            if output_text:
                return output_text
            else:
                print("--- Travel Pass: Error - No final text generated ---")
                return "Sorry, I encountered an error generating the final itinerary."

        except Exception as e:
            print(f"An error occurred during request processing: {e}")
            # For now, re-raise to see traceback during development
            raise
        finally:
            stop_spinner.set()
            spinner_thread.join(timeout=1.0)

    def _handle_research(self, args_dict):
        """Handles a research request by delegating to the ResearchAgent."""
        if not args_dict or not isinstance(args_dict, dict):
            return "Error: Invalid arguments format for research tool."

        query = args_dict.get("query")
        if not query:
            return "Error: 'query' parameter is required for research."

        print(f"📚 TravelAgent: Starting research on: '{query}'")

        try:
            # Call the research agent's main method
            result = self.research_agent.perform_research(query)
            print(f"✅ TravelAgent: Research completed successfully")
            return result
        except Exception as e:
            error_msg = f"Error performing research: {str(e)}"
            print(f"❌ {error_msg}")
            # Return a graceful error message that will be shown to the user
            return f"I tried to research information about '{query}', but encountered a technical issue. Here's what I can tell you based on my existing knowledge: This topic may require specific up-to-date information that I don't have access to at the moment."
