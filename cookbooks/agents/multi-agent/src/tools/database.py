import uuid
from datetime import datetime
from src.tools.common import get_memory_collection

from src.tools.common import judgment
@judgment.observe_tools()
class DatabaseTools:
    """Knowledge storage and retrieval tools."""

    def put_database(self, content: str, tags: str = "", importance: str = None) -> str:
        """Input: information + tags/importance (str) | Action: save information to database with unique ID | Output: storage confirmation with ID (str)"""
        
        collection = get_memory_collection()
        
        memory_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
                    
        metadata = {
            'timestamp': timestamp,
            'importance': importance,
            'tags': tags,
            'memory_id': memory_id
        }
        
        collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )
        
        print(f"[AGENT: {self.name}] 💾 Data stored: {memory_id} ({importance}) - {len(content)} chars")
        return f"Agent {self.name} stored data into database with ID: {memory_id}"

    def get_database(self, memory_id: str) -> str:
        """Input: memory ID | Action: retrieve specific data by ID | Output: retrieved data (str)"""
        
        collection = get_memory_collection()
            
        result = collection.get(ids=[memory_id])
        if result['documents']:
            return result['documents'][0]
        return f"No data found for in database with ID: {memory_id}"
            
    def search_database(self, query: str, limit: int = 10) -> str:
        """Input: search query + limit (str, int) | Action: search database for relevant information | Output: matching entries with relevance scores (str)"""
        
        collection = get_memory_collection()
        
        results = collection.query(
            query_texts=[query],
            n_results=min(limit, 10)
        )
        
        if not results['documents'] or not results['documents'][0]:
            return f"No relevant knowledge found for query: '{query}'"
        
        print(f"[AGENT: {self.name}] 🔍 Retrieved {len(results['documents'][0])} knowledge entries for: {query}")
        return f"The query results for the database: {results['documents'][0]}"