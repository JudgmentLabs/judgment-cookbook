from openai import OpenAI
from judgeval.common.tracer import Tracer, wrap
import inspect
import chromadb

client = OpenAI()

judgment = Tracer(project_name="multi-agent-system", deep_tracing=False)
memory_client = chromadb.PersistentClient(path="./memory_db")

def get_memory_collection(collection_name="documents"):
    """Get or create a memory collection."""
    return memory_client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"}
    )

def generate_tools_section(agent_instance):
    """Auto-generate the AVAILABLE TOOLS section organized by tool categories, filtered by function_map."""
    
    if not hasattr(agent_instance, 'function_map') or not agent_instance.function_map:
        return "No tools available."
    
    sections = []
    
    for parent_class in agent_instance.__class__.__mro__:
        class_name = parent_class.__name__
        
        if class_name == 'AgentTools' or not class_name.endswith('Tools'):
            continue
            
        display_name = class_name.replace('Tools', '').upper() + ' TOOLS'
        tools = []
        
        for name, method in inspect.getmembers(parent_class, predicate=inspect.isfunction):
            if name.startswith('_') or name not in agent_instance.function_map:
                continue
            
            sig = inspect.signature(method)
            params = []
            for param_name, param in sig.parameters.items():
                if param_name != 'self':
                    param_type = param.annotation.__name__ if param.annotation != inspect.Parameter.empty else 'any'
                    params.append(f"{param_name}: {param_type}")
            
            docstring = inspect.getdoc(method) or "No description available"
            
            param_str = ", ".join(params)
            tool_info = f"  • {name}({param_str}) - {docstring}"
            tools.append(tool_info)
        
        if tools:
            sections.append(f"{display_name}:\n" + "\n".join(tools))
    
    return "\n\n".join(sections)