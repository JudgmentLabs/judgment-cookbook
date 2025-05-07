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

### Agent Examples

These cookbooks showcase how to build and evaluate different types of AI agents:

*   **`financial_agent/`**: An agent designed for financial queries, demonstrating evaluation of its responses and data retrieval accuracy.
*   **`food_suggestion/`**: A simple agent that provides food recommendations, evaluated for relevancy and correctness.
*   **`langgraph_agent/`**: Contains examples of agents built with LangGraph, including:
    *   `basic/`: A foundational LangGraph agent setup.
    *   `human_in_the_loop/`: Demonstrates incorporating human feedback and intervention within a LangGraph agent.
*   **`movie_recommendation_agent/`**: An agent for recommending movies, focusing on evaluating the quality and relevance of suggestions.
*   **`music_suggestion/`**: An agent that provides music recommendations, with evaluations for suggestion accuracy.
*   **`openai_travel_agent/`**: A travel planning agent built using OpenAI models, evaluated for its ability to generate useful travel itineraries and information.

### Scoring & Evaluation Examples

These cookbooks focus on specific scoring mechanisms and evaluation patterns:

*   **`classifier_scorer/`**: Illustrates evaluating various classification and text processing tasks, such as:
    *   `branding.py`: Scoring for brand guideline adherence.
    *   `competitor_mentions.py`: Identifying and scoring mentions of competitors.
    *   `pii_checker.py`: Detecting and evaluating Personal Identifiable Information (PII) handling.
    *   `text2sql.py`: Evaluating the correctness of text-to-SQL generation.
*   **`custom_scorers/`**: Provides examples of how to implement and use custom scorers to tailor evaluations to specific needs beyond built-in scorers.
