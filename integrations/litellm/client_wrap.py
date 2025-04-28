import asyncio
import os
from dotenv import load_dotenv

import litellm # <-- Import litellm

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
# from openai import OpenAI # Keep commented if not used

# from openai import AzureOpenAI, AsyncAzureOpenAI # Keep commented for now

from judgeval.common.tracer import Tracer, JudgmentTraceCallbackHandler # Import the new handler

# Load environment variables from .env file
load_dotenv()
# Initialize the Tracer (make sure JUDGMENT_API_KEY and JUDGMENT_ORG_ID are set in your environment)
# Ensure JUDGMENT_ORG_ID is set if not passed explicitly
tracer = Tracer(project_name="client_wrap_test", api_key=os.getenv("JUDGMENT_API_KEY"), organization_id=os.getenv("JUDGMENT_ORG_ID"))

# Instantiate the callback handler
judgment_callback = JudgmentTraceCallbackHandler()

# Wrap the clients - REMOVED wrap() as callbacks handle tracing now
# Ensure API keys are set in your environment variables
chatOpenAIclient = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    callbacks=[judgment_callback] # Add the handler
)
chatAnthropicclient = ChatAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-3-haiku-20240307",
    callbacks=[judgment_callback] # Add the handler
)

# Define a more complex message
complex_prompt = """
Write a detailed tutorial (around 2000 words) explaining the concept of 
asynchronous programming in Python. Cover the following topics clearly:
- The need for async programming (I/O-bound vs CPU-bound).
- The `asyncio` library.
- `async` and `await` keywords.
- Coroutines and how they differ from regular functions.
- Event loops.
- Running multiple coroutines concurrently (e.g., using `asyncio.gather`).
- Tasks in asyncio.
- Common pitfalls and best practices.

Provide clear, concise code examples for each concept.
"""
hello_message = [HumanMessage(content=complex_prompt)]
# Add OpenAI format message list for LiteLLM/direct OpenAI calls
hello_message_openai_fmt = [{"role": "user", "content": complex_prompt}]

@tracer.observe()
def main():
    print("--- Making LLM calls (this might take a few minutes) ---")
    try:
        sync_openai_response = chatOpenAIclient.invoke(hello_message)
        print(f"OpenAI Response: {sync_openai_response.content}")
    except Exception as e:
        print(f"Error calling OpenAI: {e}")

    try:
        sync_anthropic_response = chatAnthropicclient.invoke(hello_message)
        print(f"Anthropic Response: {sync_anthropic_response.content}")
    except Exception as e:
        print(f"Error calling Anthropic: {e}")

    # --- Add LiteLLM Call --- 
    try:
        # Example: Using LiteLLM to call OpenAI gpt-3.5-turbo
        # Ensure OPENAI_API_KEY is set in env for LiteLLM to pick up
        print("--- Calling LiteLLM (gpt-3.5-turbo) ---")
        litellm_response = litellm.completion(
             model="gpt-3.5-turbo",
             messages=hello_message_openai_fmt # Use OpenAI format
        )
        # Access content based on LiteLLM response structure
        litellm_content = litellm_response.choices[0].message.content
        print(f"LiteLLM Response Content Length: {len(litellm_content)}")
    except Exception as e:
        print(f"Error calling LiteLLM: {e}")
    # --------------------------

if __name__ == "__main__":
    main()

    print("\nTrace completed. Check Judgment for trace details.")
