import json
import uuid
from subagent import AgentBase, ToolsMixin
from judgeval.scorers import FaithfulnessScorer
from judgeval.data import Example
from tools.common import judgment, client

SYSTEM_PROMPT = '''
You are an orchestrator agent that acts like a project manager coordinating multiple specialist agents. You can break down complex queries and delegate to specialized agents.

Available coordination tools:

break_down_query: Analyze a complex query and identify research components
  - query: The complex query to break down

delegate_research: Delegate a research task to a new research agent
  - task: The specific research task to perform
  - agent_id: Optional identifier for the research agent (will be generated if not provided)

delegate_math: Delegate mathematical problems and calculations
  - problem: The mathematical problem to solve

delegate_report: Delegate report generation and formatting tasks
  - request: The report generation request

get_research_results: Get results from a specific research agent
  - agent_id: The identifier of the research agent to check

synthesize_results: Combine results from multiple agents into cohesive response
  - results: Dictionary of results from different agents
  - original_query: The original user query

Your workflow:
1. BREAK DOWN complex queries into manageable research components
2. DELEGATE each component to a specialized research agent
3. COORDINATE the research agents and collect their results
4. DELEGATE to math/report agents as needed
5. SYNTHESIZE all results into a comprehensive response

You can manage multiple research agents simultaneously for parallel work.

You must format your output as tool calls with this exact format, you cannot use <break_down_query>, <delegate_research>, <get_research_results>, <delegate_math>, <delegate_report>, or <synthesize_results> tags, must use the following format:
<tool>
{"name": "tool_name", "args": {"parameter": "value"}}
</tool>

Think like a project manager: decompose, delegate, coordinate, and deliver.
'''

@judgment.identify(identifier="name", track_state=True)  
class OrchestratorAgent(ToolsMixin, AgentBase):
    def __init__(self, model="gpt-4.1", name="OrchestratorAgent"):
        super().__init__(client=client, model=model)
        self.name = name
        
        # Track active research agents
        self.active_research_agents = {}
        
        # Lazy initialization to avoid circular imports
        self._math_agent = None
        self._report_agent = None
        
        self.function_map = {
            "break_down_query": self.break_down_query,
            "delegate_research": self.delegate_research,
            "get_research_results": self.get_research_results,
            "delegate_math": self.delegate_math,
            "delegate_report": self.delegate_report,
            "synthesize_results": self.synthesize_results
        }

    def _get_math_agent(self):
        """Get or create math agent instance."""
        if self._math_agent is None:
            from .math_agent import MathAgent
            self._math_agent = MathAgent(self.client, self.model, name="MathAgent-Coordinated")
        return self._math_agent
    
    def _get_report_agent(self):
        """Get or create report agent instance."""
        if self._report_agent is None:
            from .report_agent import ReportAgent
            self._report_agent = ReportAgent(self.client, self.model, name="ReportAgent-Coordinated")
        return self._report_agent
    
    def _get_research_agent(self, task: str = None, agent_id: str = None):
        """Get or create research agent instance and optionally assign a task."""
        from .research_agent import ResearchAgent
        
        # Generate unique agent ID if not provided or if collision exists
        if not agent_id or agent_id in self.active_research_agents:
            agent_id = str(uuid.uuid4())[:8]
        
        # Create new research agent instance
        research_agent = ResearchAgent(self.client, self.model, name=f"ResearchAgent-{agent_id}")
        
        # Store the agent
        self.active_research_agents[agent_id] = {
            "agent": research_agent,
            "task": task,
            "status": "idle" if task is None else "working",
            "result": None
        }
        
        # If task provided, start the work
        if task:
            print(f"[ORCHESTRATOR] 🔍 Delegating research task to agent '{agent_id}': {task[:50]}...")
            try:
                result = research_agent.process_request(task)
                self.active_research_agents[agent_id]["result"] = result
                self.active_research_agents[agent_id]["status"] = "completed"
                print(f"[ORCHESTRATOR] ✅ Research agent '{agent_id}' completed task")
            except Exception as e:
                self.active_research_agents[agent_id]["status"] = "failed"
                print(f"[ORCHESTRATOR] ❌ Research agent '{agent_id}' failed: {str(e)}")
                raise e
        
        return research_agent, agent_id

    @judgment.observe(span_type="tool")
    def break_down_query(self, query: str) -> str:
        """Break down a complex query into research components."""
        breakdown_prompt = f"""
Analyze this query to determine if it has MULTIPLE DISTINCT RESEARCH COMPONENTS that require separate agents.

QUERY: {query}

DECISION CRITERIA:
- SINGLE COMPONENT: Query asks about one topic/domain (e.g., "What is machine learning?")
- MULTIPLE COMPONENTS: Query asks about multiple topics, domains, or types of research (e.g., "Compare solar vs wind energy costs AND efficiency")

WHEN TO CREATE MULTIPLE COMPONENTS:
✓ Multiple topics: "solar energy AND wind energy"  
✓ Different research types: "market trends AND technical specifications"
✓ Multiple domains: "healthcare regulations AND AI technology"
✓ Different expertise areas: "financial analysis AND environmental impact"

EXAMPLES:
Single: "What are coffee shop startup costs?" → SINGLE_COMPONENT: Coffee shop startup costs research
Multiple: "Compare solar vs wind energy costs and efficiency" → COMPONENT_1: Solar energy costs and efficiency data | COMPONENT_2: Wind energy costs and efficiency data
Multiple: "Find compound interest formula and calculate ROI for $10K" → COMPONENT_1: Compound interest formulas and definitions | COMPONENT_2: ROI calculation methods for $10K investment

FORMAT YOUR RESPONSE:
If SINGLE component needed:
SINGLE_COMPONENT: [Specific research task]

If MULTIPLE components needed:
COMPONENT_1: [Specific research task 1]
COMPONENT_2: [Specific research task 2]
COMPONENT_3: [Specific research task 3] (if needed)
COMPONENT_4: [Specific research task 4] (if needed)

Each component should be focused, independent, and together cover the full query scope.
"""
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a project manager expert at breaking down complex tasks."},
                {"role": "user", "content": breakdown_prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    @judgment.observe(span_type="tool")
    def delegate_research(self, task: str, agent_id: str = None) -> str:
        """Delegate a research task to a new research agent."""
        try:
            _, agent_id = self._get_research_agent(task=task, agent_id=agent_id)
            return f"Research task delegated to agent '{agent_id}'. Status: completed"
        except Exception as e:
            return f"Research task delegation failed for agent '{agent_id}': {str(e)}"

    @judgment.observe(span_type="tool")
    def get_research_results(self, agent_id: str) -> str:
        """Retrieve results from a research agent."""
        if agent_id not in self.active_research_agents:
            return f"Error: Research agent '{agent_id}' not found."
        
        agent_info = self.active_research_agents[agent_id]
        
        if agent_info["status"] == "completed":
            return f"Results from {agent_id}:\n{agent_info['result']}"
        elif agent_info["status"] == "working":
            return f"Research agent '{agent_id}' is still working on: {agent_info['task']}"
        else:
            return f"Research agent '{agent_id}' failed to complete task."

    @judgment.observe(span_type="tool")
    def synthesize_results(self, results: dict, original_query: str) -> str:
        """Combine results from multiple agents into cohesive response."""
        # Convert results dict to formatted text
        all_results = ""
        if isinstance(results, str):
            # Handle case where results might be passed as string
            all_results = results
        else:
            for agent_id, result in results.items():
                all_results += f"\n=== {agent_id} Results ===\n{result}\n"
        
        synthesis_prompt = f"""
Synthesize the following research results into a comprehensive, cohesive response to the original query.

ORIGINAL QUERY: {original_query}

RESEARCH RESULTS:
{all_results}

TASK: Create a unified, well-structured response that:
1. Addresses all aspects of the original query
2. Integrates findings from all research components
3. Identifies connections and patterns across results
4. Provides clear, actionable insights

Synthesized Response:
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are an expert analyst who synthesizes complex research into clear insights."},
                {"role": "user", "content": synthesis_prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    def list_active_agents(self) -> str:
        """List all currently active research agents for debugging."""
        if not self.active_research_agents:
            return "No active research agents."
        
        agent_list = []
        for agent_id, info in self.active_research_agents.items():
            status = info["status"]
            task = info["task"][:50] + "..." if len(info["task"]) > 50 else info["task"]
            agent_list.append(f"- {agent_id}: {status} | Task: {task}")
        
        return f"Active Research Agents ({len(self.active_research_agents)}):\n" + "\n".join(agent_list)

    @judgment.observe(span_type="tool")
    def delegate_math(self, problem: str) -> str:
        """Delegate mathematical problems to the math agent."""
        print(f"[ORCHESTRATOR] Delegating to Math Agent: {problem[:50]}...")
        agent = self._get_math_agent()
        result = agent.process_request(problem)
        print(f"[ORCHESTRATOR] Math Agent completed task")
        return result
    
    @judgment.observe(span_type="tool")
    def delegate_report(self, request: str) -> str:
        """Delegate report generation to the report agent."""
        print(f"[ORCHESTRATOR] Delegating to Report Agent: {request[:50]}...")
        agent = self._get_report_agent()
        result = agent.process_request(request)
        print(f"[ORCHESTRATOR] Report Agent completed task")
        return result

    def _execute_function(self, function_name: str, args: dict) -> str:
        """Execute a function with the given arguments and return string result."""
        if function_name not in self.function_map:
            return f"Error: Unknown function '{function_name}'"
        
        try:
            func = self.function_map[function_name]
            if function_name == "break_down_query":
                result = func(args.get("query", ""))
            elif function_name == "delegate_research":
                result = func(args.get("task", ""), args.get("agent_id", None))
            elif function_name == "get_research_results":
                result = func(args.get("agent_id", ""))
            elif function_name == "delegate_math":
                result = func(args.get("problem", ""))
            elif function_name == "delegate_report":
                result = func(args.get("request", ""))
            elif function_name == "synthesize_results":
                result = func(args.get("results", {}), args.get("original_query", ""))
            else:
                return f"Error: Unsupported function '{function_name}'"
            
            return str(result)
        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"

    @judgment.observe(span_type="function")
    def process_request(self, user_request):
        print(f"[ORCHESTRATOR] Planning workflow for: {user_request[:100]}...")
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_request}
        ]
        
        max_steps = 200  # Allow coordination of multiple agents
        step = 0
        
        while step < max_steps:
            # Get response from model
            response = self._call_model(messages, [])
            
            # Extract content from response
            if hasattr(response, 'content'):
                content = response.content
            elif hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
            else:
                content = str(response)
            
            # Add assistant response to messages
            messages.append({"role": "assistant", "content": content})
            
            # Check for tool call (delegation)
            tool_name, tool_args = self._parse_tool_call(content)
            
            if tool_name:
                print(f"[ORCHESTRATOR] Step {step + 1}: Executing {tool_name}")
                result = self._execute_function(tool_name, tool_args)
                
                # Add tool result as user message
                tool_response = f"<result>{result}</result>"
                messages.append({"role": "user", "content": tool_response})
                
                step += 1
            else:
                # No tool call found, orchestrator is providing final synthesis
                print(f"[ORCHESTRATOR] Workflow complete - synthesizing final response")
                example = Example(
                    input=user_request,
                    actual_output=content.strip(),
                    retrieval_context=[str(s) for s in messages]
                )
                judgment.async_evaluate(
                    scorers=[FaithfulnessScorer(threshold=1.0)],
                    example=example,
                    model="gpt-4.1"
                )
                return content.strip()
        
        return "Error: Maximum orchestration steps exceeded" 