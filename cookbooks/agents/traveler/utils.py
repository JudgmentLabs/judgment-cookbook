import os
from datetime import datetime
import requests
from dotenv import load_dotenv, set_key

load_dotenv()

AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")

def get_valid_token():
    """
    Get a valid Amadeus API token.
    If the token is older than 30 minutes, refresh it.
    """
    
    token = os.getenv("AMADEUS_ACCESS_TOKEN")
    
    if not token:
        refresh_token()

    token_created_at = os.getenv("AMADEUS_TOKEN_CREATED_AT")

    created_time = datetime.fromisoformat(token_created_at)
    current_time = datetime.now()
    time_difference = (current_time - created_time).total_seconds() / 60
    
    if time_difference >= 30:
        refresh_token()
    
    return

def refresh_token():
    """
    Refresh the Amadeus API token and set it in the environment.
    """

    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_CLIENT_ID,
        "client_secret": AMADEUS_CLIENT_SECRET
    }

    response = requests.post(url, headers=headers, data=data)
   
    if response.status_code == 200:
        token_data = response.json()
        set_key('.env', "AMADEUS_ACCESS_TOKEN", token_data["access_token"])
        
        token_creation_time = datetime.now()
        set_key('.env', "AMADEUS_TOKEN_CREATED_AT", token_creation_time.isoformat())

        print("🔑 Amadeus Token Refreshed")
        load_dotenv()
    
    return
if __name__ == "__main__":
    get_valid_token()