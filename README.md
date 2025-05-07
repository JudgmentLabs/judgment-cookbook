### judgment-cookbook

This repo contains cookbooks demonstrating Judgment evaluations of AI Agents

Before running these examples, make sure you have:

1. Installed the Judgeval package:
   ```bash
   pip install judgeval
   ```

2a. Set up your Judgeval API key and organization ID as environment variables:
   ```bash
   export JUDGMENT_API_KEY="your_api_key"
   export JUDGMENT_ORG_ID="your_org_id"
   ```

2b. Alternatively, set up a .env file with your API key and organization ID included
   ```bash
   JUDGMENT_API_KEY="your_api_key"
   JUDGMENT_ORG_ID="your_org_id"
   ```
   Finally, run source .env to update the environment file. 

The cookbooks contained in this repo assume that you have a Judgment Labs account. If you don't have an account, sign up for one [here](https://app.judgmentlabs.ai/login).

## Cookbooks Overview

This repository provides a collection of cookbooks to demonstrate various evaluation techniques and agent implementations using Judgeval.

### LangGraph Agent Examples

These cookbooks showcase agents built using the LangGraph framework, demonstrating complex state management and chained operations.

*   **`langgraph_agent/`**: Core examples of LangGraph agent patterns:
    *   `basic/`: A foundational LangGraph agent setup illustrating tool use and conditional edges.
    *   `human_in_the_loop/`: Demonstrates incorporating human feedback and intervention within a LangGraph agent flow.
*   **`financial_agent/`**: A LangGraph-based agent for financial queries, featuring RAG capabilities with a vector database for contextual data retrieval and evaluation of its reasoning and data accuracy.

### Direct API Agent Examples

These cookbooks feature agents that interact directly with LLM APIs (e.g., OpenAI, Anthropic), often implementing custom logic for tool use, function calling, and RAG.

*   **`openai_travel_agent/`**: A travel planning agent using direct OpenAI API calls, custom tool functions, and RAG with a vector database for up-to-date and contextual travel information. Evaluated for itinerary quality and information relevance.
*   **`food_suggestion/`**: (Located in `cookbooks/food_recommendation/`) A simple agent providing food recommendations via direct OpenAI API calls and custom search/recommendation logic. Evaluated for relevancy and correctness.
*   **`movie_recommendation_agent/`**: An agent for recommending movies. (Assumed Direct API / RAG - *details WIP*). Focuses on evaluating suggestion quality and relevance.
*   **`music_suggestion/`**: An agent that provides music recommendations. (Assumed Direct API / RAG - *details WIP*). Evaluated for suggestion accuracy.
### Scoring & Evaluation Examples

These cookbooks focus on specific scoring mechanisms and evaluation patterns:

*   **`classifier_scorer/`**: Illustrates evaluating various classification and text processing tasks, such as:
    *   `branding.py`: Scoring for brand guideline adherence.
    *   `competitor_mentions.py`: Identifying and scoring mentions of competitors.
    *   `pii_checker.py`: Detecting and evaluating Personal Identifiable Information (PII) handling.
    *   `text2sql.py`: Evaluating the correctness of text-to-SQL generation.
*   **`custom_scorers/`**: Provides examples of how to implement and use custom scorers to tailor evaluations to specific needs beyond built-in scorers.
