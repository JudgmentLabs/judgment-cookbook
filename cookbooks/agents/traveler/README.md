# Blookit - A Simple Travel Agent

A basic travel agent that can process natural language requests and call various travel-related services/tools.

A good example of this is emulating what a travel company would do to build their own agent (e.g. Expedia, Airbnb, Booking.com, etc.).

Here, we are simulating this by using the [Amadeus API](https://developers.amadeus.com/) to act as some of our tools. Once you have your Amadeus access set up, run the script `utils.py` to load your keys.

## Running the Agent

1. Copy the `.env.example` file to `.env` and add your own API keys

2. Run the CLI interface:
   ```
   python cli.py
   ```

## judgeval integration points

1. `agent.py`: The central `process_request()` function is decorated using the tracer and has an `async_evaluate` to trigger an [online evaluation](https://docs.judgmentlabs.ai/performance/online_evals). The `wrap()` decorator is also used to capture the LLM calls.

2. `tools.py`: Each tool is decorated with `@judgment.observe`.

## Current features and tools built into the agent

- Flight search tool
- Hotel search tool

## Project Structure

- `agent.py` - Main agent logic that processes user requests
- `tools.py` - Implementation of travel-related tools
- `tools.json` - Tool schema for the agent
- `cli.py` - Command line interface for interacting with the agent
