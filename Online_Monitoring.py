import chromadb
import re
import os
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset
from judgeval.tracer import Tracer, wrap

# Load environment variables from .env file
load_dotenv()

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

judgment = Tracer(project_name="agentic-rag-monitoring")
client = wrap(OpenAI())

def init_chromadb():
    client = chromadb.Client()
    collection = client.create_collection("knowledge_base")
    
    print("Loading Wikipedia dataset...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")


    batch_size = 100
    print(f"Adding {min(batch_size, len(dataset))} articles to ChromaDB...")
 
    documents = []
    metadatas = []
    ids = []

    for i in range(min(batch_size, len(dataset))):
        article = dataset[i]
        documents.append(article["text"])
        metadatas.append({"url": article["url"]})
        ids.append(article["title"])

    # Single batch add
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"ChromaDB initialized with {min(batch_size, len(dataset))} Wikipedia articles")
    return collection

@judgment.observe(span_type="tool")
def search_documents(collection, query: str, n_results: int = 10):
    """Search for relevant documents"""
    results = collection.query(query_texts=[query], n_results=n_results)
    if results['ids'][0]:
        return results['ids'][0]
    return []

@judgment.observe(span_type="tool")
def get_document(collection, doc_id: str):
    """Get specific document by ID"""
    result = collection.get(ids=[doc_id])

    if result['ids']:
        url = result['metadatas'][0]['url']
        content = result['documents'][0]
        print(url)
        return f"Title: {doc_id}\nURL: {url}\nContent: {content}"
    return None

@judgment.observe(span_type="tool")
def synthesize_report(query: str, documents: list) -> str:
    """Create a structured report with proper citations using LLM"""
    
    citation_prompt = f"""
    Create a comprehensive research report for: "{query}"
    
    Use the following sources and provide proper citations. Each source contains the title, URL, and content:
    
    {chr(10).join([f"Source {i+1}: {doc}..." for i, doc in enumerate(documents, 1)])}
    
    Requirements:
    1. Synthesize information from all sources
    2. Create a structured report with introduction, main points, and conclusion
    3. Ensure all information is properly attributed to its source
    4. Write in a professional, academic style
    5. Use numbered in-text citations throughout the report (e.g., "This concept was first introduced [1] and later expanded [2]")
    
    Format the report as:
    # Research Report: [Query]
    
    ## Introduction
    [Brief introduction to the topic]
    
    ## Key Findings
    [Main points with numbered in-text citations like [1], [2], [3], etc.]
    
    ## Conclusion
    [Summary and conclusions]
    
    ## Works Cited
    [Numbered list of all sources used and their corresponding URL, formatted as: "1. Title of Article - Wikipedia (URL)"]
    """
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": citation_prompt}],
        temperature=0.1
    )
    
    return response.choices[0].message.content

class ResearchAgent:
    def __init__(self):
        self.collection = init_chromadb()
        self.system_prompt = """You are a research agent. You will use the following tools to complete a report based on the user's request.

AVAILABLE TOOLS:
1. search_documents(query, n_results) - Search for relevant documents, returns list of document IDs
2. get_document(doc_id) - Get specific document content by ID (returns "Title: [title]\nContent: [content]")
3. add_document(text) - Add new document to knowledge base
4. synthesize_report(query, documents) - Create structured report from documents (documents should be actual text content, not IDs)

EXECUTION PROCESS:
1. **Search**: Find relevant documents with search_documents()
2. **Retrieve**: Get actual content of each document with get_document()
3. **Synthesize**: Create report with synthesize_report() using actual document content
4. **Assess**: Have you COMPLETELY fulfilled the user's request?
   - If NO: Continue searching until you have enough information to answer the question
   - If YES: Complete the task

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
            "synthesize_report": lambda: synthesize_report(**params)
        }
        return tool_handlers.get(tool_name, lambda: None)()
    
    @judgment.observe(span_type="function")
    def run(self, user_query: str, max_iterations: int = 50):
        """Run the agent with LLM-driven tool calling"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Query: {user_query}"}
        ]
        
        for iteration in range(max_iterations):
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
            )
            
            llm_output = response.choices[0].message.content
            messages.append({"role": "assistant", "content": llm_output})
            
            # Parse tool calls
            if "<tool>" in llm_output:
                # Extract JSON from <tool> tags
                tool_match = re.search(r'<tool>\s*(\{.*?\})\s*</tool>', llm_output, re.DOTALL)
                if tool_match:
                    import json
                    tool_data = json.loads(tool_match.group(1))
                    tool_name = tool_data["name"]
                    params = tool_data["args"]
                    
                    tool_result = self.call_tool(tool_name, params)
                    
                    # If synthesize_report, return the report directly
                    if tool_name == "synthesize_report":
                        return tool_result
                    
                    # Add result to conversation
                    messages.append({
                        "role": "user", 
                        "content": f"<result>{tool_result}</result>"
                    })
            
            # Check if agent is done (no tool call and has content)
            elif "<tool>" not in llm_output and llm_output.strip():
                return llm_output.strip()
        
        return "Agent reached maximum iterations"

def research_agent_complete(user_query: str) -> str:
    """Main function to run the research agent"""
    agent = ResearchAgent()
    return agent.run(user_query)

if __name__ == "__main__":
    
    # Test with first query
    query =  "What is the history of computers?"
    print(f"Query: {query}")
    result = research_agent_complete(query)
    print(f"Result: {result}")