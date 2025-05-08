import asyncio
import os
from dotenv import load_dotenv

import litellm
from judgeval.common.tracer import Tracer
from judgeval.integrations.langgraph import JudgevalCallbackHandler
from langchain_core.messages import HumanMessage

load_dotenv()
tracer = Tracer(project_name="client_wrap_test")

judgment_callback = JudgevalCallbackHandler(tracer)

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
    try:
        print("--- Calling LiteLLM (gpt-4.1) ---")
        litellm_response = litellm.completion(
             model="gpt-4.1",
             messages=hello_message_openai_fmt
        )
        litellm_content = litellm_response.choices[0].message.content
        print(f"LiteLLM Response Content Length: {len(litellm_content)}")
    except Exception as e:
        print(f"Error calling LiteLLM: {e}")
    # --------------------------

if __name__ == "__main__":
    main()

    print("\nTrace completed. Check Judgment for trace details.")
