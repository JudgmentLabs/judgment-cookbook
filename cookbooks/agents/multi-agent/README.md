# Multi-Agent System

A flexible multi-agent framework for orchestrating and evaluating the collaboration of multiple agents and tools on complex tasks.

## Running the Multi-Agent System
1. Set up your environment variables as needed (see .env.example).
2. Run `main.py`.

## judgeval integration points

`agents/`: Each of the agents in the multi-agent system use the `@judgment.identify` on each agent class, and `@judgment.observe` on each agent tool/function. The orchestrator, report, and research agents all have [online evals](https://docs.judgmentlabs.ai/performance/online_evals) attached via the `async_evaluate()` method.
`tools/`: Each tool is decorated with `@judgment.observe`.

## Directory structure

- main.py         - Main orchestration logic for running the multi-agent system
- agents/         - Definitions and logic for individual agents
- tools/          - Implementation of tools accessible to agents
- subagent/       - Sub-agent logic or reusable agent components
- reports/        - Generated reports and evaluation outputs
- tests/          - Test suite for the multi-agent system
