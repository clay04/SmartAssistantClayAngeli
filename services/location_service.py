import os
import requests
from config import Config

GOOGLE_MAPS_API_KEY = Config.GOOGLE_MAPS_API_KEY

BASE_URL_GEOCODE = "https://maps.googleapis.com/maps/api/geocode/json"
BASE_URL_PLACES = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"


def get_place_info(latitude: float, longitude: float) -> dict:
    """
    Ambil informasi lokasi dari koordinat (alamat lengkap + tempat sekitar).
    """

    # --- 1. Reverse Geocoding (dari lat/lon ke alamat)
    geocode_params = {
        "latlng": f"{latitude},{longitude}",
        "key": GOOGLE_MAPS_API_KEY,
        "language": "id"  # biar hasilnya bahasa Indonesia
    }

    geocode_res = requests.get(BASE_URL_GEOCODE, params=geocode_params)
    geocode_data = geocode_res.json()

    if geocode_data.get("status") != "OK":
        raise Exception(f"Geocoding error: {geocode_data.get('status')}")

    address = geocode_data["results"][0]["formatted_address"]

    # --- 2. Nearby Places (misalnya ambil tempat dalam radius 500 meter)
    places_params = {
        "location": f"{latitude},{longitude}",
        "radius": 500,
        "key": GOOGLE_MAPS_API_KEY,
        "language": "id"
    }

    places_res = requests.get(BASE_URL_PLACES, params=places_params)
    places_data = places_res.json()

    if places_data.get("status") != "OK":
        raise Exception(f"Places error: {places_data.get('status')}")

    nearby_places = []
    for place in places_data.get("results", [])[:5]:  # ambil maksimal 5 tempat
        nearby_places.append({
            "name": place.get("name"),
            "address": place.get("vicinity"),
            "rating": place.get("rating")
        })

    return {
        "address": address,
        "nearby_places": nearby_places
    }
