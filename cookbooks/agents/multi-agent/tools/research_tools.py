from typing import List, Dict
import json
from datetime import datetime, timedelta
from .common import judgment, client
from random import random
from fastapi import HTTPException
from openai import (
    APIError,     # For API-specific errors
    RateLimitError,  # When you hit rate limits
)

class ResearchToolsMixin:
    """Mixin providing research and information gathering tools."""

    @judgment.observe(span_type="tool")
    def web_search(self, query: str) -> str:
        """Search the web for information on a given query using LLM."""

        prompt = f"""
    Generate a comprehensive research summary about: {query}

    Include:
    1. Current state and recent developments
    2. Key trends and innovations
    3. Industry impact and applications
    4. Future outlook

    Format the response as a well-structured research summary with clear sections.
    """
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a research expert who provides detailed, factual summaries."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    @judgment.observe(span_type="tool")
    def extract_facts(self, text: str) -> str:
        """Extract key facts and statistics from text."""
        prompt = f"""
Extract the key facts and statistics from the following text. List them as bullet points.

Text: {text}

Key facts:
"""
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    @judgment.observe(span_type="tool")
    def summarize_sources(self, sources: list) -> str:
        """Summarize information from multiple sources."""
        if not sources:
            return "No sources provided for summarization."
        
        # Simple LLM-based summarization
        sources_text = "\n---\n".join([str(source) for source in sources])
        
        prompt = f"""
Summarize the following research sources into a comprehensive overview:

{sources_text}

Provide a well-structured summary that:
1. Integrates key findings from all sources
2. Identifies common themes and patterns
3. Highlights unique insights from each source
4. Presents a cohesive narrative
"""
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are an expert research analyst who synthesizes information from multiple sources."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    @judgment.observe(span_type="tool")
    def evaluate_completeness(self, information: str, original_query: str) -> str:
        """Assess if gathered information is sufficient to answer the query using LLM evaluation."""
        
        evaluation_prompt = f"""
You are an information completeness evaluator. Your job is to assess whether the gathered information is sufficient to answer the original query.

ORIGINAL QUERY: {original_query}

GATHERED INFORMATION:
{information}

EVALUATION TASK:
1. Analyze if the gathered information adequately addresses the original query
2. Consider completeness, relevance, and depth of information
3. If sufficient, respond with "SUFFICIENT: [brief explanation]"
4. If insufficient, respond with "INSUFFICIENT: [specific gaps]" and list exactly what information is missing

Respond in this format:
STATUS: [SUFFICIENT/INSUFFICIENT]
ANALYSIS: [Your detailed assessment]
MISSING: [If insufficient, list specific missing information]
NEXT_SEARCH: [If insufficient, suggest specific search queries to fill gaps]
"""

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are an expert information analyst who evaluates research completeness."},
                {"role": "user", "content": evaluation_prompt}
            ]
        )
        
        return response.choices[0].message.content.strip() 