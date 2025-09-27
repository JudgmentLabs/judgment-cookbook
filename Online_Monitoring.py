# ChromaDB Knowledge Base Agent with 3 Tools
# Simple roll-for-loop implementation

import chromadb
from typing import Dict, List

def init_chromadb():
    """Initialize ChromaDB with sample data"""
    client = chromadb.Client()
    collection = client.create_collection("knowledge_base")
    
    # Add sample documents
    sample_docs = [
        "Python is a programming language known for its simplicity and readability.",
        "Machine learning is a subset of AI that learns from data.",
        "ChromaDB is a vector database for storing embeddings.",
        "Online monitoring tracks AI agent performance in real-time.",
        "Dual-scorer evaluation combines output and trajectory assessment."
    ]
    
    for i, doc in enumerate(sample_docs):
        collection.add(
            documents=[doc],
            metadatas=[{"topic": f"topic_{i}"}],
            ids=[f"doc_{i}"]
        )
    
    return collection

def search_documents(collection, query: str, n_results: int = 3) -> List[Dict]:
    """Tool 1: Search for relevant documents"""
    results = collection.query(query_texts=[query], n_results=n_results)
    
    formatted_results = []
    for i in range(len(results['ids'][0])):
        formatted_results.append({
            'id': results['ids'][0][i],
            'text': results['documents'][0][i],
            'metadata': results['metadatas'][0][i]
        })
    
    return formatted_results

def get_document(collection, doc_id: str) -> Dict:
    """Tool 2: Get a specific document by ID"""
    result = collection.get(ids=[doc_id])
    if result['ids']:
        return {
            'id': result['ids'][0],
            'text': result['documents'][0],
            'metadata': result['metadatas'][0]
        }
    return {}

def add_document(collection, text: str, metadata: Dict = None) -> str:
    """Tool 3: Add a new document to the knowledge base"""
    if metadata is None:
        metadata = {}
    
    doc_id = f"doc_{len(collection.get()['ids'])}"
    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[doc_id]
    )
    return doc_id

def roll_for_loop_agent(query: str) -> str:
    """Agent that uses roll-for-loop approach with 3 tools"""
    collection = init_chromadb()
    
    try:
        # Step 1: Search for relevant documents
        search_results = search_documents(collection, query, n_results=3)
        
        if not search_results:
            return "No relevant documents found."
        
        # Step 2: Get detailed information from top result
        top_doc_id = search_results[0]['id']
        doc_details = get_document(collection, top_doc_id)
        
        # Step 3: Synthesize answer
        if doc_details:
            answer = f"Based on the knowledge base: {doc_details['text']}"
        else:
            answer = "Unable to retrieve detailed information."
        
        return answer
        
    except Exception as e:
        return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    # Test queries
    test_queries = [
        "What is Python?",
        "How does machine learning work?",
        "What is ChromaDB?"
    ]
    
    print("🚀 ChromaDB Agent with Roll-For-Loop")
    print("=" * 40)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        answer = roll_for_loop_agent(query)
        print(f"📝 Answer: {answer}")
    
    print("\n🎉 Demo Complete!")