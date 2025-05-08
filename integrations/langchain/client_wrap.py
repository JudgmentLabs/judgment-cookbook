import asyncio
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from judgeval.common.tracer import Tracer
from judgeval.integrations.langgraph import JudgevalCallbackHandler

load_dotenv()
tracer = Tracer(project_name="client_wrap_test")

judgment_callback = JudgevalCallbackHandler(tracer)
chatOpenAIclient = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1",
    callbacks=[judgment_callback]
)
chatAnthropicclient = ChatAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-3-haiku-20240307",
    callbacks=[judgment_callback]
)

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

if __name__ == "__main__":
    main()

    print("\nTrace completed. Check Judgment for trace details.")
