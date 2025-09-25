from flask import Blueprint, request, jsonify
from services.location_service import get_place_info
from flask_sock import Sock
import json

location_bp = Blueprint("location", __name__)

sock = Sock()

@location_bp.route("/status", methods=["POST"])
def location_status():
    data = request.json
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    
    print("Received location:", latitude, longitude)
    
    if latitude is None or longitude is None:
        return jsonify({"error": "Latitude and Longitude are required"}), 400
    
    try:
        location_info = get_place_info(latitude, longitude)
        print("Location Info:", location_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    # Here you can add logic to process the location data
    # For example, you might want to log it, store it in a database, etc.
    
    return jsonify({
        "message": "Location received",
        "latitude": latitude,
        "longitude": longitude,
        "location_info": location_info
    }), 200
    
    
@sock.route('/location/ws')
def location_ws(ws):
    while True:
        data = ws.receive()
        if data is None:
            break
        try:
            data_json = json.loads(data)
            latitude = data_json.get("latitude")
            longitude = data_json.get("longitude")
            
            print("WebSocket Received location:", latitude, longitude)
            
            if latitude is None or longitude is None:
                ws.send(json.dumps({"error": "Latitude and Longitude are required"}))
                continue
            
            try:
                location_info = get_place_info(latitude, longitude)
                print("Location Info:", location_info)
            except Exception as e:
                ws.send(json.dumps({"error": str(e)}))
                continue
            
            ws.send(json.dumps({
                "message": "Location received",
                "latitude": latitude,
                "longitude": longitude,
                "location_info": location_info
            }))
        except json.JSONDecodeError:
            ws.send(json.dumps({"error": "Invalid JSON"}))