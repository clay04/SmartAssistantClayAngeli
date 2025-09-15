import requests
from config import Config

OPENCAGE_API_KEY = Config.OPENCAGE_API_KEY

BASE_URL_GEOCODE = "https://api.opencagedata.com/geocode/v1/json"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def get_place_info(latitude: float, longitude: float) -> dict:
    """
    Ambil informasi lokasi dari koordinat (alamat lengkap + tempat sekitar).
    """

    # --- 1. Reverse Geocoding dengan OpenCage
    geocode_params = {
        "q": f"{latitude},{longitude}",
        "key": OPENCAGE_API_KEY,
        "language": "id",
        "pretty": 1
    }

    geocode_res = requests.get(BASE_URL_GEOCODE, params=geocode_params)
    if geocode_res.status_code != 200:
        raise Exception(f"Reverse geocoding failed: {geocode_res.text}")

    geocode_data = geocode_res.json()
    if not geocode_data.get("results"):
        raise Exception("Tidak ditemukan hasil geocoding")

    best_match = geocode_data["results"][0]
    address = best_match.get("formatted", "Alamat tidak diketahui")
    components = best_match.get("components", {})

    address_info = {
        "road": components.get("road"),
        "neighbourhood": components.get("neighbourhood"),
        "suburb": components.get("suburb"),
        "village": components.get("village"),
        "city": components.get("city"),
        "state": components.get("state"),
        "postcode": components.get("postcode"),
        "country": components.get("country"),
    }

    # --- 2. Nearby Places dengan Overpass API (radius 500m)
    # Cari amenity = shop, supermarket, convenience, restaurant, hospital, school
    query = f"""
    [out:json];
    (
      node["shop"](around:500,{latitude},{longitude});
      node["amenity"="hospital"](around:500,{latitude},{longitude});
      node["amenity"="school"](around:500,{latitude},{longitude});
      node["amenity"="restaurant"](around:500,{latitude},{longitude});
    );
    out;
    """
    overpass_res = requests.post(OVERPASS_URL, data={"data": query})
    if overpass_res.status_code != 200:
        raise Exception(f"Overpass query failed: {overpass_res.text}")

    overpass_data = overpass_res.json()
    nearby_places = []
    for element in overpass_data.get("elements", [])[:5]:
        nearby_places.append({
            "name": element.get("tags", {}).get("name"),
            "type": element.get("tags", {}).get("shop") or element.get("tags", {}).get("amenity"),
            "lat": element.get("lat"),
            "lon": element.get("lon"),
        })

    return {
        "address": {
            "display_name": address,
            "details": address_info
        },
        "nearby_places": nearby_places
    }
