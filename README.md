# judgment-cookbook

This repo contains cookbooks demonstrating evaluations of AI Agents using the `judgeval` package implemented by [Judgment Labs](https://judgmentlabs.ai/).


## Prerequisites
Before running these examples, make sure you have:

1. Installed the latest version of the Judgeval package:
   ```bash
   pip install judgeval
   ```

2. Set up your Judgeval API key and organization ID as environment variables:
   ```bash
   export JUDGMENT_API_KEY="your_api_key"
   export JUDGMENT_ORG_ID="your_org_id"
   ```

To get your API key and Organization ID, make an account on the [Judgment Labs platform](https://app.judgmentlabs.ai/login).


## Cookbooks Overview
This repository provides a collection of cookbooks to demonstrate various evaluation techniques and agent implementations using Judgeval.

### Handrolled API Agent Examples

These cookbooks feature agents that interact directly with LLM APIs (e.g., OpenAI, Anthropic), often implementing custom logic for tool use, function calling, and RAG.

*   **`traveler/`**: A travel planning agent using OpenAI API calls, custom tool functions, and the Amadeus API for up-to-date and contextual travel information. Evaluated on factual adherence to retrieval context.
*   **`multi-agent/`**: A flexible multi-agent framework for orchestrating and evaluating the collaboration of multiple agents and tools on complex tasks like financial analysis. Evaluated on factual adherence to retrieval context.

### LangGraph Agent Examples

These cookbooks showcase agents built using the LangGraph framework, demonstrating complex state management and chained operations.

*   **`langgraph_music_recommender/`**: An agent that generates song recommendations based on user music taste.
   
### Writing Custom Scorers

These cookbooks focus on how to implement and use custom scorers:

*   **`custom_scorers/`**: Provides examples of how to implement and use custom scorers to tailor evaluations to specific needs beyond built-in scorers.
