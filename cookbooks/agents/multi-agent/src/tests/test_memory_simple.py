from src.agents.agent import Agent
from src.tools import judgment

@judgment.observe(span_type="function")
def test_agent_database():
    print("🤖 Testing Agent Database Integration...")
    print("=" * 60)
    
    agent = Agent(name="TestAgent")

    print("🤖 Testing Agent Database Integration...")
    print("=" * 60)

    print("\n📝 TEST 1: STORING INFORMATION")
    print("-" * 40)
    
    content = "OpenAI released GPT-4 in March 2023, featuring improved reasoning and multimodal capabilities"
    tags = "openai,gpt-4,ai,language-model"
    importance = "high"
    
    print(f"Storing: {content[:50]}...")
    result = agent.store_database(content, tags, importance)
    print(f"Result: {result}")
    
    print("\n🔍 TEST 2: SEARCHING DATABASE")
    print("-" * 40)
    
    query = "GPT-4 language model"
    print(f"Searching for: '{query}'")
    search_result = agent.search_database(query, 3)
    print(f"Found: {search_result}")
    
    print("\n🔄 TEST 3: STORE-THEN-SEARCH WORKFLOW")
    print("-" * 40)
    
    unique_content = "ChromaDB version 0.4.15 introduced performance improvements for large collections"
    unique_tags = "chromadb,version,performance"
    
    print(f"1. Storing: {unique_content}")
    store_result = agent.store_database(unique_content, unique_tags, "high")
    print(f"   Store result: {store_result}")
    
    print(f"2. Searching for: 'ChromaDB performance'")
    search_result = agent.search_database("ChromaDB performance", 2)
    print(f"   Search result: {search_result}")
    
    if unique_content in str(search_result):
        print("✅ SUCCESS: Stored content found in search!")
    else:
        print("❌ ISSUE: Stored content not found")
    
    print("\n" + "=" * 60)
    print("🎉 AGENT DATABASE TEST COMPLETED!")

if __name__ == "__main__":
    test_agent_database()