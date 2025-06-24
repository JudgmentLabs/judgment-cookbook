import os
import requests
from tavily import TavilyClient

tavily_client = TavilyClient()

from src.tools.common import judgment
@judgment.observe_tools()
class WebTools:
    """Web search and content extraction tools."""

    def web_search(self, query: str) -> str:
        """Input: search query or topic (str) | Action: comprehensive web search for information with ranked results | Output: formatted search results with titles, snippets, and URLs (str)"""

        response = tavily_client.search(
            query=query, 
            include_images=True,
            include_image_description=True,
            )
        
        return f"Web search results for query: {query} are: {response}"

    def web_extract_content(self, url: str) -> str:
        """Input: complete webpage URL (str) | Action: extract and clean main content, removing ads/navigation | Output: cleaned text content (str)"""
       
        response = tavily_client.extract(
            urls=[url],
            include_images=True,
        )
        return f"Web extract content results for url: {url} are: {response}"
    
    def web_extract_images(self, url: str) -> str:
        """Input: direct image URL ending in .jpg/.png (str) | Action: download image and save to reports/media/ directory | Output: local image file path (str)"""
        try:
            os.makedirs("reports/media", exist_ok=True)
            
            response = requests.get(url)
            response.raise_for_status()
            
            filename = os.path.basename(url.split('?')[0])
            filepath = os.path.join("reports/media", filename)
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return f"Successfully downloaded image to: {filepath}"
            
        except Exception as e:
            return f"Error downloading image: {str(e)}" 