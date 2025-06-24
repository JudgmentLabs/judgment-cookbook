from src.tools.web import WebTools
from src.tools.docx_tools import DocxTools
from src.tools.operations import OperationsTools
from src.tools.database import DatabaseTools

class AgentTools(WebTools, DocxTools, OperationsTools, DatabaseTools):
    """Combined tools for AI agents."""
    pass