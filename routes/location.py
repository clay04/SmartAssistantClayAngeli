from flask import Blueprint, request, jsonify
from services.location_service import get_place_info
import json
from extensions import socketio
from services.location_service import get_place_info
from services.token_service import validate_token
from db import get_db

location_bp = Blueprint("location", __name__)

@socketio.on("update_locaiton")
def handle_location_update(data):
    try:
        token = data.get("access_token")
        user_id = validate_token(token)
        if not user_id:
            emit("error": "Invalid Token")
            return
        
        latitude = data.get("latitude")
        longitude = data.get("longitude")
        
        if not latitude or not longitude:
            emit("error": "Latitude/Longitude di perlukan")
            return
        
        location_info = get_place_info(latitude, longitude)
        address = location_info["address"]["display_name"]
        nearby = ", ".join(
            [f"{p['name']} ({p['type']})" for p in location_info["nearby_places"] if p.get("name")])
        location_context = f"Lokasi saat ini: {address}. Tempat terdekat: {nearby}."
        
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
                        UPDATE user_input
                        SET latitude=%s, longitude=%s, location_text=%s
                        WHERE id_user=%s
                    """, (latitude, longitude, location_context, user_id))
        conn.commit()
        
        emit("ask_location", {
            "message" : "Lokasi berhasil diperbarui",
            "address" : location_context,
            "latitude" : latitude,
            "longitude" : longitude
        })
        
        print(f"✅ Location updated for user {user_id}: {location_context}")
        
    except Exception as e:
        emit("error", {"error": str(e)})
        print("Errror saat memperbarui Lokasi", e)