#!/usr/bin/env python3
"""
Multi-Agent System - Independent Agents with Collaborative Workflows

This demonstrates a futuristic multi-agent system where agents are completely 
independent but can collaborate when coordinated externally, OR be coordinated
by an orchestrator agent acting like a product manager.
"""

from agents import OrchestratorAgent, MathAgent, ResearchAgent, ReportAgent

def demo_individual_agents():
    """Demonstrate individual agent capabilities."""
    print("=== Individual Agent Demonstrations ===")
    
    # Math Agent
    # print("\n--- Math Agent ---")
    # math_agent = MathAgent()
    # math_result = math_agent.process_request("Sarah has 15 apples. She gives 7 to her friend. How many does she have left?")
    # print(f"Math Result: {math_result}")
    
    # Research Agent  
    print("\n--- Research Agent ---")
    research_agent = ResearchAgent()
    research_result = research_agent.process_request("Latest trends in artificial intelligence")
    print(f"Research Result: {research_result[:200]}...")
    
    # Report Agent
    # print("\n--- Report Agent ---")
    # report_agent = ReportAgent()
    # report_result = report_agent.process_request("Create an executive summary about AI market growth")
    # print(f"Report Result: {report_result[:200]}...")

def demo_orchestrated_workflows():
    """Demonstrate orchestrator agent coordinating multiple specialist agents."""
    print("\n=== Orchestrated Multi-Agent Workflows ===")
    
    orchestrator = OrchestratorAgent()
    
    complex_requests = [
        "Calculate the compound interest on $10,000 at 5% for 3 years, then research current investment trends, and finally create a comprehensive financial analysis report with charts",
        # "Research renewable energy market growth, calculate projected market size if it grows 20% annually for 5 years starting from $500B, then generate an executive summary with visualizations",
        # "Solve this business problem: A company's profit increased by 15% to $230,000. What was the original profit? Then research industry benchmarks and create a professional business report."
    ]
    
    for i, request in enumerate(complex_requests, 1):
        print(f"\n--- Orchestrated Workflow {i} ---")
        print(f"Request: {request}")
        print(f"[ORCHESTRATOR] Processing complex multi-agent workflow...")
        result = orchestrator.process_request(request)
        print(f"Final Result: {result[:300]}...")

def demo_collaborative_workflow_1():
    """Demonstrate financial analysis workflow with agent collaboration."""
    print("\n=== Manual Collaborative Workflow 1: Financial Analysis ===")
    
    # Step 1: Math Agent calculates compound interest
    math_agent = MathAgent()
    math_request = "Calculate compound interest on $10,000 at 5% annual rate for 3 years"
    print(f"🧮 Math Agent: {math_request}")
    financial_calculation = math_agent.process_request(math_request)
    print(f"📊 Calculation Result: {financial_calculation}")
    
    # Step 2: Research Agent gathers market context
    research_agent = ResearchAgent()
    research_request = "Research current investment market trends and interest rates"
    print(f"\n🔍 Research Agent: {research_request}")
    market_research = research_agent.process_request(research_request)
    print(f"📈 Research Result: {market_research[:150]}...")
    
    # Step 3: Report Agent creates comprehensive analysis
    report_agent = ReportAgent()
    report_content = f"""
    Financial Calculation Results:
    {financial_calculation}
    
    Market Research Findings:
    {market_research}
    """
    report_request = f"Create a professional investment analysis report with charts based on this data: {report_content}"
    print(f"\n📋 Report Agent: Creating comprehensive investment analysis...")
    final_report = report_agent.process_request(report_request)
    print(f"📑 Final Report: {final_report[:200]}...")

def demo_collaborative_workflow_2():
    """Demonstrate business research workflow."""
    print("\n=== Manual Collaborative Workflow 2: Business Analysis ===")
    
    # Step 1: Research Agent gathers business data
    research_agent = ResearchAgent()
    research_request = "Research electric vehicle market size, growth rates, and key players"
    print(f"🔍 Research Agent: {research_request}")
    business_research = research_agent.process_request(research_request)
    
    # Step 2: Math Agent calculates market projections
    math_agent = MathAgent()
    math_request = "If the EV market is $300 billion and grows 25% annually, what will it be in 5 years?"
    print(f"\n🧮 Math Agent: {math_request}")
    market_projection = math_agent.process_request(math_request)
    
    # Step 3: Report Agent synthesizes findings
    report_agent = ReportAgent()
    combined_data = f"""
    Market Research:
    {business_research}
    
    Growth Projections:
    {market_projection}
    """
    print(f"\n📋 Report Agent: Creating business intelligence report...")
    business_report = report_agent.process_request(f"Create an executive summary with charts for: {combined_data}")
    print(f"📑 Business Intelligence Report Generated")

def main():
    """Demonstrate futuristic multi-agent collaboration."""
    print("🚀 MULTI-AGENT SYSTEM")
    print("Independent Agents • Orchestrated Coordination • Manual Collaboration")
    print("=" * 80)

    try:
        # Show individual capabilities
        #demo_individual_agents()
        
        # Show orchestrated workflows (product manager style)
        demo_orchestrated_workflows()
        
        # # Show manual collaborative workflows 
        # demo_collaborative_workflow_1()
        # demo_collaborative_workflow_2()
        
        print("\n" + "=" * 80)
        print("✅ Multi-Agent System Demonstration Complete")
        print("🤖 Independent agents can work alone OR be orchestrated")
        print("🔮 Future: Flexible agent coordination patterns")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("Make sure you have set up your OpenAI API key and installed dependencies.")

if __name__ == "__main__":
    main() 