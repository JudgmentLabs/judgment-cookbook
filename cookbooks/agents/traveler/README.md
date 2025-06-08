# Blookit - A Simple Travel Agent

A basic travel agent that can process natural language requests and call various travel-related services/tools.

A good example of this is emulating what a travel company would do to build their own agent (e.g. Expedia, Airbnb, Booking.com, etc.).

Here, we are simulating this by using the Amadeus API to act as some of our tools.

## Running the Agent

1. Copy the `.env.example` file to `.env` and add your own API keys

2. Run the CLI interface:
   ```
   python cli.py
   ```

## Current features and tools built into the agent

- Flight search tool
- Hotel search tool

## Project Structure

- `agent.py` - Main agent logic that processes user requests
- `tools.py` - Implementation of travel-related tools
- `tools.json` - Tool schema for the agent
- `cli.py` - Command line interface for interacting with the agent
