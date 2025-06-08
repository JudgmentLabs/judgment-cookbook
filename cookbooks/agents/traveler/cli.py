"""
Simple CLI for Travel Agent

This script provides a command-line interface for interacting with the travel agent.
"""

from openai import OpenAI
from agent import TravelAgent
from utils import get_valid_token


def main():
    print("\n")
    print("👋 Welcome to Blookit, your personal travel agent!")
    print("\n")
    print("I can help you with flights ✈️, hotels 🏨, car rentals 🚗, and things to do 🎢.")
    print("\n")
    print("Some examples of how you can use me:")
    print("👉 Could you help me find round trip flights from San Francisco to Hawaii for May 12-20?")
    print("👉 Book me a trip to New York City from May 17st to May 25th. I'll need a rountrip flight from San Francisco and a hotel with a gym and wifi for my stay.")
    print("\n")
    print("Type 'exit' to end the conversation.\n")

    get_valid_token()

    client = OpenAI()
    agent = TravelAgent(client=client)

    while True:
        user_input = input("\nHow can I help you? ")

        if user_input.lower() in ['exit', 'quit', 'bye']:
            break

        response = agent.process_request(user_input)
        print("\n")
        print(response)


if __name__ == "__main__":
    main()
