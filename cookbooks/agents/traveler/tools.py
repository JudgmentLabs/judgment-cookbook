"""
Travel Tools Module

This module contains the implementation of various travel-related tools that can be used by the agent.
"""

import os
import requests
from openai import OpenAI
from judgeval.common.tracer import Tracer, wrap

judgment = Tracer(
    project_name="traveler-test",
    deep_tracing=False
)
AMADEUS_TOKEN = os.getenv("AMADEUS_ACCESS_TOKEN")
print("AMADEUS_TOKEN: " + AMADEUS_TOKEN)
client = wrap(OpenAI())

@judgment.observe(span_type="tool")
def get_flights(parameters):
    """
    Tool to search for flights based on the provided parameters.
    
    Args:
        params (dict): Dictionary containing search parameters such as:
            - origin: Departure city or airport
            - destination: Arrival city or airport
            - departure_date: When the user wants to leave
            - return_date: when the user wants to return (optional)
            - passengers: Number of travelers (optional)
            - budget: Maximum price the user is willing to pay
            - non_stop: Whether the user wants non-stop flights only
    Returns:
        str: Response with flight information (placeholder for now)
    """
    
    # if parameters is a string, convert it to a dictionary
    if isinstance(parameters, str):
        import json
        try:
            parameters = json.loads(parameters)
        except json.JSONDecodeError:
            return {"error": "Invalid parameters format. Expected JSON or dictionary."}
    
    # Set defaults for optional parameters
    if "passengers" not in parameters or parameters["passengers"] is None:
        parameters["passengers"] = 1
    
    if "budget" not in parameters or parameters["budget"] is None:
        parameters["budget"] = 1500
    
    url = 'https://test.api.amadeus.com/v2/shopping/flight-offers'
    headers = {
        'Authorization': f'Bearer {AMADEUS_TOKEN}',
        "Content-Type": "application/json"  
    }
    params = {
        # ✈️  Trip endpoints & dates  ────────────────────────────────────────────
        "originLocationCode":    parameters["origin"],          # e.g. "BOS"
        "destinationLocationCode": parameters["destination"],        # e.g. "PAR"
        "departureDate":         parameters["departure_date"],          # "YYYY-MM-DD", mandatory

        # If you want a round-trip, uncomment and set:
        # "returnDate":          None,          # "YYYY-MM-DD"

        # 👥  Passenger counts  ─────────────────────────────────────────────────
        "adults":                parameters["passengers"],          # int ≥1  (required)
        # "children":            None,          # int ≥0  (optional)
        # "infants":             None,          # int ≥0  (≤ adults)

        # 🛋️  Cabin & airline filters  ─────────────────────────────────────────
        # "travelClass":         None,          # "ECONOMY" | "PREMIUM_ECONOMY" | "BUSINESS" | "FIRST"
        # "includedAirlineCodes": None,         # "AA,DL"  (comma-separated IATA codes)
        # "excludedAirlineCodes": None,         # "WN,AS"  (cannot combine with included)

        # 🔀  Routing & price filters  ─────────────────────────────────────────
        "nonStop":               str(parameters["non_stop"]).lower(),          # Must be "true" or "false" as string
        "currencyCode":          "USD",          # "USD", "EUR", ...
        "maxPrice":              parameters["budget"],          # int >0   (per traveler)
        "max":                   10,          # int ≥1   (number of offers to return)
    }
    
    # optional return date for round trips
    if "return_date" in parameters:
        params["returnDate"] = parameters["return_date"]
    
    response = requests.get(url, headers=headers, params=params)
    print("🛩️ Flights Found")
    return response.json()

@judgment.observe(span_type="tool")
def get_hotels(parameters):
    """
    Tool to search for hotels based on the provided parameters.
    
    Args:
        params (dict): Dictionary containing search parameters such as:
            - location: City or area (required)
            - radius: distance from the center of the city in miles (optional)
            - amenities: amenities the hotel should have (optional)
            - ratings: minimum rating of the hotel (optional)
            - checkin_date: checkin date (optional)
            - checkout_date: checkout date (optional)
            
    Returns:
        str: Response with hotel information (placeholder for now)
    """
    
     # if parameters is a string, convert it to a dictionary
    if isinstance(parameters, str):
        import json
        try:
            parameters = json.loads(parameters)
        except json.JSONDecodeError:
            return {"error": "Invalid parameters format. Expected JSON or dictionary."}

    url = 'https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city'
    headers = {
        'Authorization': f'Bearer {AMADEUS_TOKEN}'
    }
    params = {
        "cityCode": parameters["location"],
        "radiusUnit": "MILE",
        "hotelSource": "DIRECTCHAIN"
    }
    
    if "radius" in parameters:
        params["radius"] = parameters["radius"]
    if "amenities" in parameters:
        params["amenities"] = parameters["amenities"]
    if "ratings" in parameters:
        params["ratings"] = parameters["ratings"]
    
    response = requests.get(url, headers=headers, params=params)
    
    # if response is a string
    response_json = json.loads(response.text)
    
    # limit response to just 30 hotels
    hotel_list = response_json["data"][:30]
    hotel_ids = [hotel["hotelId"] for hotel in hotel_list]
    
    # Build URL directly with comma-separated hotel IDs
    hotel_ids_str = ','.join(hotel_ids)
    url = f'https://test.api.amadeus.com/v3/shopping/hotel-offers?hotelIds={hotel_ids_str}'
    
    # Add optional parameters if provided
    if "checkin_date" in parameters:
        url += f'&checkInDate={parameters["checkin_date"]}'
    if "checkout_date" in parameters:
        url += f'&checkOutDate={parameters["checkout_date"]}'
    
    # Make the direct request
    prices_response = requests.get(url, headers=headers)
    
    prices_data = prices_response.json()
    
    # for each hotel, extract the price from the hotel offers response
    for hotel in hotel_list:
        hotel["price"] = "Not available" 
        if "data" in prices_data:
            for hotel_offer in prices_data.get("data", []):
                if hotel_offer.get("hotel", {}).get("hotelId") == hotel["hotelId"] and hotel_offer.get("offers"):
                    hotel["price"] = hotel_offer["offers"][0]["price"]["total"]
                    
    print("🏨 Hotels Found")
    return hotel_list

@judgment.observe(span_type="tool")
def get_car_rentals(params):
    """
    Tool to search for car rentals based on the provided parameters.
    
    Args:
        params (dict): Dictionary containing search parameters such as:
            - location: Pickup location
            - pickup_date: When to pick up the car
            - return_date: When to return the car
            - car_type: Type of car (optional)
            
    Returns:
        str: Response with car rental information (placeholder for now)
    """
    # TODO: This is just a placeholder - would call a real API in a production environment
    return "Car rental search tool called with parameters: " + str(params)

@judgment.observe(span_type="tool")
def get_things_to_do(params):
    """
    Tool to search for activities and attractions based on the provided parameters.
    
    Args:
        params (dict): Dictionary containing search parameters such as:
            - location: City or area
            - date: Date when the activity should be available
            - category: Type of activity (optional)
            - price_range: Min and max price (optional)
            
    Returns:
        str: Response with activities information (placeholder for now)
    """
    # TODO: This is just a placeholder - need to parse params and fill out body as so. use randomo values for now

    url = 'https://test.api.amadeus.com/v1/shopping/activities'
    headers = {
        'Authorization': f'Bearer {AMADEUS_TOKEN}'
    }
    body = {
        "latitude": params.get('latitude', 41.397158),
        "longitude": params.get('longitude', 2.160873),
        "radius": params.get('radius', 20)
    }

    response = requests.get(url, headers=headers, params=body)
    return response.json()