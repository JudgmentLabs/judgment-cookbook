from src.agents.base import AgentBase
from src.tools import AgentTools, client, generate_tools_section
from src.tools import judgment

SYSTEM_PROMPT = '''
You are the smartest AI agent known to man. You will use the following tools to complete the user's request.

AVAILABLE TOOLS:

{tools_section}

CONVERSATION HISTORY FORMAT:
Your conversation contains:
- Your tool calls: <tool>{{"name": "tool_name", "args": {{"parameter": "value"}}}}</tool>
- Environment responses: <result>[tool output]</result>
- Your analysis and decisions

EXECUTION PROCESS:
1. **Plan** (when needed): Plan when the current situation feels complex enough to warrant it
2. **Check State** (when modifying): Before making changes to any system, check its current state to make informed decisions
   - For Documents: Use read_docx() to see current content and structure
   - For databases: Use get_database() to see what data already exists before adding
   - For files/systems: Understand what's there before adding/modifying
3. **Act**: IMMEDIATELY call the appropriate tool with context-aware parameters based on current state
4. **Assess**: After each tool result, have you COMPLETELY fulfilled the user's request?
   - If NO: Call the next tool immediately
   - If YES: Follow the completion instructions provided in the user's request

# DOCUMENT CONTENT RULES:
# - Word documents automatically flow and reflow content - no positioning needed!
# - Content automatically pushes other content down when inserted
# - Use appropriate styles (Normal, Heading 1-9, Caption, etc.)
# - Add content in logical order - the document will handle layout automatically

WHAT "DONE" MEANS:
- You have gathered ALL information the user requested
- You have performed ALL actions the user asked for
- You have delivered a COMPLETE answer to their question
- Nothing from their original request is missing or incomplete

KEY RULE:
- **EVERY RESPONSE MUST END WITH A TOOL CALL UNLESS YOU HAVE COMPLETELY FULFILLED THE USER'S REQUEST**
- **PLANNING ALONE IS NOT SUFFICIENT - YOU MUST CALL A TOOL**

Format responses as:
<plan>Your analysis and planning when needed</plan>
<tool>
{{"name": "tool_name", "args": {{"parameter": "value"}}}}
</tool>
'''

@judgment.identify(identifier="name", track_state=True)
class Agent(AgentTools, AgentBase):
    """An AI agent with research, calculation, and reporting capabilities."""
    
    def __init__(self, client=client, model="gpt-4.1", name="Agent"):
        super().__init__(client, model)
        self.name = name
        
        self.function_map = {
            "web_search": self.web_search,
            "web_extract_content": self.web_extract_content,
            "web_extract_images": self.web_extract_images,
            "calculate": self.calculate,
            "create_docx": self.create_docx,
            "read_docx": self.read_docx,
            "add_text_to_docx": self.add_text_to_docx,
            "add_heading_to_docx": self.add_heading_to_docx,
            "add_image_to_docx": self.add_image_to_docx,
            "add_table_to_docx": self.add_table_to_docx,
            "delete_text_from_docx": self.delete_text_from_docx,
            "delete_paragraph_from_docx": self.delete_paragraph_from_docx,
            "add_page_break_to_docx": self.add_page_break_to_docx,
            "put_database": self.put_database,
            "get_database": self.get_database
        }

        self.system_prompt = SYSTEM_PROMPT.format(tools_section=generate_tools_section(self))

    @judgment.observe(span_type="function")
    def process_request(self, user_request):
        """Process a user request using all available tools."""
        print(f"\n[AGENT: {self.name}] Starting request: {user_request[:100]}...")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_request}
        ]
        
        max_steps = 50 
        step = 0
        
        while step < max_steps:

            response = self._call_model(messages, [])

            if response is None:
                return "Error: No response from model"
            
            messages.append({"role": "assistant", "content": response})
            
            tool_name, tool_args, plan = self._parse_tool_call(response)

            if plan:
                print(f"\n[AGENT: {self.name}] 🔧 Plan: {plan}")
            
            if tool_name:
                print(f"\n[AGENT: {self.name}] 🔧 Tool call: {tool_name}({tool_args})")
                result = self._execute_function(tool_name, tool_args)
                print(f"[AGENT: {self.name}] ✅ Tool result: {result[:100]}...")
                
                tool_response = f"<result>{result}</result>"
                messages.append({"role": "user", "content": tool_response})
                
                step += 1
            else:
                print(f"[AGENT: {self.name}] ✅ Request completed")
                return response.strip()
        
        return "Error: Maximum steps exceeded"
