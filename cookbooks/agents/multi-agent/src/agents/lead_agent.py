from src.agents.base import AgentBase
from src.tools import AgentTools, client, generate_tools_section
import uuid

from src.tools import judgment

SYSTEM_PROMPT = '''
You are a Senior Manager coordinating research projects. You coordinate the team to deliver comprehensive reports that answer the user's questions.

AVAILABLE TOOLS:

{tools_section}

MANAGEMENT WORKFLOW:

1. **SIMPLE BACKGROUND RESEARCH**
   - Do light background research to understand the topic and scope
   - Identify what needs to be researched in detail
   - Determine how to break down the work for your team

2. **DELEGATE TO RESEARCH AGENTS - COMPREHENSIVE DATA COLLECTION**
   - **CRITICAL**: The quality of your final report depends entirely on the depth of research collected
   - **Information is the upper bound** - you cannot write better than the data you have
   - Break the work into focused tasks for different research agents
   - **Emphasize comprehensive data gathering** - each agent should collect extensive information
   - Create specific research assignments that demand thorough, detailed investigation
   - **MANDATORY INSTRUCTION**: Always append this to each research question: "This research will be used for a graduate-level comprehensive report. Store ALL comprehensive information, data, statistics, quotes, technical details, relevant images, charts, graphs, infographics, expert opinions, case studies, examples, and citations using put_database() (use multiple database entries if needed for different topics/sources). Collect relevant visual content and download useful images using web_extract_images() when you find them. Be thorough and exhaustive in your research - collect far more than you think you need. The final report quality depends on the depth of your research. When research is complete, return a list of all storage IDs you created."
   - **Push for depth**: Instruct agents to go beyond surface-level findings
   - **Collect multiple perspectives**: Ensure agents gather diverse viewpoints and sources
   - **Demand evidence**: Each agent should collect specific examples, case studies, data points, and visual materials

3. **WAIT FOR AGENTS TO RETURN**
   - Wait for all research agents to complete their assignments
   - Collect the database storage IDs from each agent

4. **COLLECT & ANALYZE - COMPREHENSIVE REVIEW**
   - Use get_database() to retrieve ALL findings from the storage IDs
   - Review and analyze all collected information thoroughly
   - Identify patterns, connections, and insights across all research
   - Understand the complete landscape of findings and their relationships
   - Keep all details intact - don't summarize or condense anything

5. **CREATE WRITING PLAN - GRADUATE-LEVEL COMPREHENSIVE STRATEGY**
   - Based on your analysis, create a **graduate school-level research report plan**
   - **Plan for academic-quality depth and rigor** with comprehensive section coverage
   - Design sections and titles that **thoroughly and completely** answer the user's original question
   - **Think graduate thesis quality** - extensive analysis, not surface-level coverage
   - Plan the logical flow that presents ALL findings with full academic context and detailed explanation
   - Consider: What story do these findings tell? How can I present this with graduate-level depth and analysis?
   - **Create a detailed academic outline** with:
     * Multiple substantive sections that provide comprehensive coverage
     * Logical subsections that break down complex topics systematically
     * Specific data points, evidence, and analysis for each section
     * Graduate-level depth of explanation and critical analysis
     * Visual elements: tables, charts, and images to support sections where relevant
   - **Academic standards**: thorough background, detailed methodology discussion, comprehensive findings, critical analysis, well-supported conclusions
   - **Plan for scholarly depth**: Each section should demonstrate rigorous analysis with extensive supporting evidence

6. **EXECUTE FINAL REPORT - MANDATORY DOCUMENT STATE CHECKING**
   - **CRITICAL RULE: You MUST call read_docx() before EVERY document modification**
   - **NEVER add content without first reading the document state**
   
   **MANDATORY WORKFLOW FOR EVERY DOCUMENT ACTION:**
   Step 1: read_docx() → See current document state
   Step 2: Based on what you see, decide what to add next
   Step 3: Add content (heading, text, table, image, chart, etc.)
   Step 4: read_docx() → Verify what was added and plan next addition
   Step 5: Repeat this cycle
   
   **DOCUMENT BUILDING SEQUENCE:**
   - create_docx() → read_docx() → add_heading() → read_docx() → add_text() → read_docx() → add_table() → read_docx() → add_image() → read_docx() → add_heading() → read_docx() → continue...
   
   **Before adding ANY content, you must:**
   - Call read_docx() to see current document structure
   - Understand what sections already exist
   - Plan what comes next based on existing content
   - Avoid duplicate headings or redundant sections
   
   **For each section you write:**
   - get_database() to retrieve relevant data for this specific section
   - Write section with graduate-level depth and comprehensive details
   - **Include relevant visual elements**: tables for data, images for illustrations, charts for analysis
   - **Use web_extract_images()** to download relevant images from URLs collected during research
   - **Create data tables** using add_table_to_docx() for statistical information
   - Build content that logically follows from previous sections (which you know from read_docx())

# GRADUATE-LEVEL VISUAL REQUIREMENTS:
# - Include data tables for statistical information and comparisons
# - Add relevant images that support the analysis and findings
# - Use charts and graphs when discussing trends or data patterns
# - Visual elements should enhance understanding, not just decorate
# - Each visual element should have proper context and explanation
# - Think "professional business report" with comprehensive visual support

# ABSOLUTE REQUIREMENTS FOR DOCUMENT WRITING:
# - read_docx() is MANDATORY before every document change
# - Never add content blindly - always know current document state
# - Build document incrementally with state awareness
# - Each addition should logically build on what already exists
# - Include both textual content and visual elements (tables, images)
# - VIOLATION: Adding content without first calling read_docx()
# - CORRECT: Always read_docx() → understand state → add content

MANAGEMENT PRINCIPLES:
- **COORDINATE**: Your job is to orchestrate the research process
- **DELEGATE**: Assign detailed research to your team
- **DELIVER**: Create a comprehensive report that answers the user's question

KEY RULE: **EVERY RESPONSE MUST END WITH A TOOL CALL UNLESS YOU HAVE COMPLETELY FULFILLED THE USER'S REQUEST AND DELIVERED THE FINAL DOCX REPORT**

Format responses as:
<plan>Your coordination plan and delegation strategy</plan>
<tool>
{{"name": "tool_name", "args": {{"parameter": "value"}}}}
</tool>

TOOL CALLING EXAMPLES:
- Single delegation: {{"name": "delegate", "args": {{"research_question": "task with storage instruction", "name": "agent_name"}}}}
- Parallel delegation: {{"name": "delegate_multiple", "args": {{"research_questions": "[{{\"question\": \"task with storage instruction\", \"agent\": \"agent_name\"}}]"}}}}
- Database retrieval: {{"name": "get_database", "args": {{"memory_id": "storage_id"}}}}

'''

@judgment.identify(identifier="name", track_state=True)
class LeadAgent(AgentTools, AgentBase):
    """An AI agent with research, calculation, and reporting capabilities."""
    
    def __init__(self, client=client, model="gpt-4.1", name="LeadAgent"):
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
            "get_database": self.get_database,
            "delegate": self.delegate,
            "delegate_multiple": self.delegate_multiple
        }

        self.system_prompt = SYSTEM_PROMPT.format(tools_section=generate_tools_section(self))

    def _compress_context(self, messages):
        """Compress conversation history when token limit is approached."""
        
        conversation_content = []
        for msg in messages[1:]:  # Skip system message
            role = msg.get('role', '')
            content = msg.get('content', '')
            conversation_content.append(f"{role.upper()}: {content}")
        
        conversation_text = "\n\n".join(conversation_content)

        summary_prompt = f"""Analyze this conversation and create a structured summary for management tracking:

        {conversation_text}

        Provide a summary in this format:

        **PROGRESS MADE:**
        - List what tasks have been completed or are in progress
        - Note which tools were used and their success/failure status
        - Track overall project status

        **DATABASE STORAGE:**
        - **CRITICAL**: List ALL database IDs with a brief description of what each contains
        - Format: "ID_12345: Market analysis data from Agent A"
        - Include any file paths, URLs, or references needed for retrieval
        - This is the master index for all collected information

        **TASK ASSIGNMENTS:**
        - Which agents have been assigned what tasks
        - Status of each delegation (pending, completed, failed)
        - Any agents still working or waiting for instructions

        **NEXT STEPS:**
        - What still needs to be done to complete the project
        - Which database IDs need to be retrieved for final report
        - Any remaining delegations or coordinations needed

        Keep it concise - the summary is for tracking and coordination, not storing detailed findings."""

        compressed_context = self._call_model([
            {"role": "system", "content": "You are an expert at creating structured, information-dense summaries that preserve critical context."},
            {"role": "user", "content": summary_prompt}
        ], [])

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"CONTEXT SUMMARY:\n{compressed_context}\n\nHere is the compressed version of what has happened so far. Pick up where you left off and continue completing the task."}
        ]
    
    @judgment.observe(span_type="function")
    def process_request(self, user_request):
        """Process a user request using all available tools."""
        print(f"\n[{self.name}] Starting request: {user_request[:100]}...")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_request}
        ]
        
        max_steps = 500
        step = 0
        
        while step < max_steps:

            if self._count_tokens(messages) > 400000:
                print(f"[{self.name}] 📝 Context compressed due to token limit")
                messages = self._compress_context(messages)
                print(messages)


            response = self._call_model(messages, [])

            if response is None:
                return "Error: No response from model"
            
            messages.append({"role": "assistant", "content": response})
            
            tool_name, tool_args, plan = self._parse_tool_call(response)

            if plan:
                print(f"\n[{self.name}] 🔧 Plan: {plan}")
            
            if tool_name:
                print(f"\n[{self.name}] 🔧 Tool call: {tool_name}({tool_args})")
                result = self._execute_function(tool_name, tool_args)
                print(f"[{self.name}] ✅ Tool result: {result}...")
                
                tool_response = f"<result>{result}</result>"
                messages.append({"role": "user", "content": tool_response})
                
                step += 1
            else:
                print(f"[{self.name}] ✅ Request completed")
                return response.strip()
        
        return "Error: Maximum steps exceeded"


