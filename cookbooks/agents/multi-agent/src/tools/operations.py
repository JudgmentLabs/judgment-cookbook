from src.tools.common import client
from typing import List, Dict
import uuid
import contextvars
from src.tools.window_logger import window_logger

from src.tools.common import judgment
@judgment.observe_tools()
class OperationsTools:
    """LLM-powered operations and calculations."""

    def calculate(self, problem: str) -> str:
        """Input: mathematical problem in natural language (str) | Action: solve mathematical calculations and expressions | Output: numerical result (str)"""
        print(f"[AGENT] 🧮 Calculating: {problem[:50]}...")
        
        math_prompt = f"""
You are a calculator. Calculate this and respond with ONLY the numerical answer. No explanations, no steps, no words - just the number.

Problem: {problem}

Answer:
"""
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are an expert mathematician who solves problems clearly and accurately."},
                {"role": "user", "content": math_prompt}
            ]
        )
        
        result = response.choices[0].message.content.strip()
        print(f"[AGENT] ✅ Math calculation completed")
        return result 

    def delegate(self, research_question: str, name: str) -> str:
        """Input: research question in natural language (str) and agent name (str) | Action: create a research agent to investigate the specific question | Output: unique findings ID for retrieval (str)"""

        from src.agents.agent import Agent

        agent_id = name
        
        window_logger.set_expected_window_count(1)
        
        window_logger.create_window(agent_id)
        window_logger.log_to_window(agent_id, f"🚀 Agent {agent_id} initialized")
        
        agent = WindowLoggedAgent(name=agent_id)
        
        print(f"[{self.name}]🤖 Delegating to {agent_id}: {research_question[:50]}...")
        
        result = agent.process_request(research_question)
        
        window_logger.mark_agent_completed(agent_id)

        return result
    
    def delegate_multiple(self, research_questions: List[Dict[str, str]]) -> str:
        """Input: list of research questions (List[Dict[question: str, agent: str]])| Action: create multiple research agents simultaneously in parallel | Output: combined findings and storage IDs from all agents (str)"""
    
        import threading
        from src.agents.agent import Agent
        
        results = []
        threads = []
        
        # Get the current context to copy for each thread
        current_ctx = contextvars.copy_context()
        
        agent_names = set()
        for task in research_questions:
            if isinstance(task, dict) and "agent" in task:
                agent_names.add(task["agent"])
        
        expected_count = len(agent_names)
        print(f"[{self.name}] 📊 Setting up {expected_count} agent windows with smart positioning")
        print(f"[{self.name}] 📝 Agent names: {sorted(list(agent_names))}")
        window_logger.set_expected_window_count(expected_count)
        
        for agent_name in sorted(agent_names):
            print(f"[{self.name}] 🔧 Creating window for: {agent_name}")
            window_logger.create_window(agent_name)
            window_logger.log_to_window(agent_name, f"🚀 Agent {agent_name} initialized")
        
        def run_agent(task):
            try:
                if not isinstance(task, dict) or "question" not in task or "agent" not in task:
                    error_msg = f"Invalid task format: {task}"
                    results.append(f"Error: {error_msg}")
                    return
                
                agent_name = task["agent"]
                
                window_logger.log_to_window(agent_name, f"📋 Task: {task['question']}")
                
                agent = WindowLoggedAgent(name=agent_name)
                result = agent.process_request(task["question"])
                
                results.append(f"[{agent_name}]: {result}")
                
                window_logger.mark_agent_completed(agent_name)
                
            except Exception as e:
                agent_name = task.get("agent", "Unknown") if isinstance(task, dict) else "Unknown"
                error_msg = str(e)
                results.append(f"[{agent_name}] Error: {error_msg}")
                window_logger.mark_agent_error(agent_name, error_msg)
        
        for task in research_questions:
            # Create a fresh context copy for each thread to avoid collision
            thread_ctx = current_ctx.copy()
            
            # Use a proper function wrapper to avoid lambda closure issues
            def create_thread_runner(task_to_run, context_to_use):
                def thread_runner():
                    context_to_use.run(run_agent, task_to_run)
                return thread_runner
            
            thread = threading.Thread(target=create_thread_runner(task, thread_ctx))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        return "\n\n".join(results)
    

class WindowLoggedAgent:
    """Agent wrapper that logs to separate windows"""

    def __init__(self, name: str):
        from src.agents.agent import Agent
        self.name = name
        self.agent = Agent(name=name)

    def process_request(self, user_request):
        """Process request with window logging"""
        window_logger.log_to_window(self.name, f"🔄 Processing: {user_request[:100]}...")
        
        original_print = print
        
        def window_print(*args, **kwargs):
            message = " ".join(str(arg) for arg in args)

            if f"[AGENT: {self.name}]" in message or "[AGENT:" not in message:
                window_logger.log_to_window(self.name, message)
            original_print(*args, **kwargs)
        
        import builtins
        builtins.print = window_print
        
        try:
            result = self.agent.process_request(user_request)
            window_logger.log_to_window(self.name, f"📝 Final result: {result[:200]}...")
            return result
        finally:
            builtins.print = original_print