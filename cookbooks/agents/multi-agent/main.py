from src.agents.agent import Agent
from src.agents.lead_agent import LeadAgent

def demo():
    """Test complex queries to AI Agent."""
    print("=" * 60)
    print("🚀 AI AGENT DEMO")
    print("=" * 60)
    
    agent = LeadAgent()
    
    test_cases = [
        "Research the origins and impact of these 12 conspiracy theories: Flat Earth Theory, Moon Landing Hoax, Paul McCartney Died ('Paul is Dead'), Simulation Theory, Hollow Earth Theory, Area 51 & UFO Cover-ups, Bermuda Triangle, Atlantis/Lost Civilizations, Bigfoot/Sasquatch Government Cover-up, Chemtrails (Weather Modification), HAARP Weather Control, and Denver International Airport Underground. For each conspiracy, document the original source/creator, how it spread, peak popularity periods, real-world consequences (violence, policy changes, economic impact), and current believer statistics.",

        "Research the impact of 2024 US tariffs on Chinese steel and aluminum imports, including specific effects on US manufacturing costs in automotive and construction sectors, with price data from major suppliers",
        
        "Analyze how the 2024 EU deforestation regulation affects palm oil supply chains from Indonesia and Malaysia, including compliance costs for major food companies like Unilever and Nestlé",
        
        "What are the available health insurance options before Medicare eligibility? What are the pros and cons of different insurance types for various demographics (employed, self-employed, unemployed, families, young adults, etc.)?"    
    ]

    for i in range(len(test_cases)):
        test_cases[i] += ". Find any neccesary images and export all the information to a docx file."
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Query: {query}")
        print("Response:")
        try:
            result = agent.process_request(query)
            print(result[:300] + "..." if len(result) > 300 else result)
        except Exception as e:
            print(f"Error: {e}")
        print()

def main():
    """Run all demonstrations."""
    print("🤖 RUNNING AGENT DEMO")
    try:
        demo()
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯  AGENT DEMO COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main() 