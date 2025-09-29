import chromadb
import re
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI()

def init_chromadb():
    client = chromadb.Client()
    collection = client.create_collection("knowledge_base")
    
    sample_docs = [
        # Relevant documents about pizza (5 docs)
        "Pizza originated in Naples, Italy in the 18th century with simple ingredients like tomatoes and mozzarella.",
        "Traditional Neapolitan pizza uses San Marzano tomatoes, mozzarella di bufala, fresh basil, and olive oil.",
        "Pizza became popular worldwide after Italian immigrants brought it to America in the early 1900s.",
        "The first pizza restaurant in the United States was Lombardi's, opened in New York City in 1905.",
        "Pizza Margherita was named after Queen Margherita of Italy and features the colors of the Italian flag.",
        
        # Irrelevant documents (5 docs)
        "Python is a programming language known for its simple syntax and readability.",
        "Basketball was invented by Dr. James Naismith in 1891 in Springfield, Massachusetts.",
        "Coffee cultivation requires specific climate conditions and two main species: Arabica and Robusta.",
        "The Great Wall of China stretches over 13,000 miles and was built over 2,000 years ago.",
        "Photography was invented in the early 19th century by Joseph Nicéphore Niépce in 1826."
    ]
    
    for i, doc in enumerate(sample_docs):
        collection.add(documents=[doc], ids=[f"doc_{i}"])
    
    return collection

def search_documents(collection, query: str, n_results: int = 3):
    """Search for relevant documents"""
    results = collection.query(query_texts=[query], n_results=n_results)
    if results['ids'][0]:
        return results['ids'][0]
    return []

def get_document(collection, doc_id: str):
    """Get specific document by ID"""
    result = collection.get(ids=[doc_id])
    if result['ids']:
        return result['documents'][0]
    return None

def add_document(collection, text: str):
    """Add new document to knowledge base"""
    doc_id = f"doc_{len(collection.get()['ids'])}"
    collection.add(documents=[text], ids=[doc_id])
    return doc_id

def synthesize_report(query: str, documents: list) -> str:
    """Create a structured report from documents"""
    if not documents:
        return f"No relevant information found for '{query}'."
    
    report = f"Research Report: {query}\n\n"
    for i, doc in enumerate(documents, 1):
        report += f"Source {i}: {doc}\n\n"
    
    return report

class ResearchAgent:
    def __init__(self):
        self.collection = init_chromadb()
        self.system_prompt = """You are the smartest AI agent known to man. You will use the following tools to complete the user's request.

AVAILABLE TOOLS:
1. search_documents(query, n_results) - Search for relevant documents, returns list of document IDs
2. get_document(doc_id) - Get specific document content by ID
3. add_document(text) - Add new document to knowledge base
4. synthesize_report(query, documents) - Create structured report from documents

EXECUTION PROCESS:
1. **Plan** (when needed): Plan when the current situation feels complex enough to warrant it
2. **Check State** (when modifying): Before making changes, check current state to make informed decisions
3. **Act**: IMMEDIATELY call the appropriate tool with context-aware parameters
4. **Assess**: After each tool result, have you COMPLETELY fulfilled the user's request?
   - If NO: Call the next tool immediately
   - If YES: Follow the completion instructions

WHAT "DONE" MEANS:
- You have gathered ALL information the user requested
- You have performed ALL actions the user asked for
- You have delivered a COMPLETE answer to their question
- Nothing from their original request is missing or incomplete

KEY RULE:
- **EVERY RESPONSE MUST END WITH A TOOL CALL UNLESS YOU HAVE COMPLETELY FULFILLED THE USER'S REQUEST**

Format responses as:
<plan>Your analysis and planning when needed</plan>
<tool>
{"name": "tool_name", "args": {"parameter": "value"}}
</tool>"""
    
    def call_tool(self, tool_name: str, params: dict):
        """Call a tool and return result"""
        tool_handlers = {
            "search_documents": lambda: search_documents(self.collection, **params),
            "get_document": lambda: get_document(self.collection, **params),
            "add_document": lambda: add_document(self.collection, **params),
            "synthesize_report": lambda: synthesize_report(**params)
        }
        return tool_handlers.get(tool_name, lambda: None)()
    
    def run(self, user_query: str, max_iterations: int = 10):
        """Run the agent with LLM-driven tool calling"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Query: {user_query}"}
        ]
        
        for iteration in range(max_iterations):
            # Get LLM response
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.1
            )
            
            llm_output = response.choices[0].message.content
            messages.append({"role": "assistant", "content": llm_output})
            
            try:
                # Parse tool calls
                if "<tool>" in llm_output:
                    # Extract JSON from <tool> tags
                    tool_match = re.search(r'<tool>\s*(\{.*?\})\s*</tool>', llm_output, re.DOTALL)
                    if tool_match:
                        import json
                        tool_data = json.loads(tool_match.group(1))
                        tool_name = tool_data["name"]
                        params = tool_data["args"]
                        
                        # Call the tool
                        tool_result = self.call_tool(tool_name, params)
                        
                        # Add result to conversation
                        messages.append({
                            "role": "user", 
                            "content": f"<result>{tool_result}</result>"
                        })
                
                # Check if agent is done (no tool call and has content)
                elif "<tool>" not in llm_output and llm_output.strip():
                    return llm_output.strip()
                
            except Exception:
                continue
        
        return "Agent reached maximum iterations"

def research_agent_complete(user_query: str) -> str:
    """Main function to run the research agent"""
    agent = ResearchAgent()
    return agent.run(user_query)

if __name__ == "__main__":
    query = "Add information about Chicago-style deep dish pizza and create a comprehensive report about pizza history and styles"
    
    print(f"Query: {query}")
    result = research_agent_complete(query)
    print(f"Result: {result}")