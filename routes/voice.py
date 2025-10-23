import os, json, base64, tempfile, threading, datetime
from flask import Blueprint, request
from services.gemini_service import analyze_image
from services.location_service import get_place_info
from flask_sock import Sock
from services.database_service import save_interaction_async
from services.token_service import validate_token
from urllib.parse import parse_qs

voice_bp = Blueprint("voice", __name__)

sock = Sock()   

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@sock.route("/voice/ws")
def assistant_ws(ws):
    """
    WebSocket untuk real-time voice+image + streaming response dari Gemini
    """
    current_image_path = None
    
    while True:
        data = ws.receive()
        if not data:
            break
        
        # Jika data berupa bytes (gambar)
        if isinstance(data, (bytes, bytearray)):
            if not user_id:
                ws.send(json.dumps({"error": "User belum tervalidasi"}))
                continue

            try:
                user_folder = os.path.join(UPLOAD_FOLDER, str(user_id))
                os.makedirs(user_folder, exist_ok=True)
                filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                file_path = os.path.join(user_folder, filename)

                def save_file():
                    try:
                        # Kalau data masih string, ubah ke bytes
                        content = data
                        if isinstance(content, str):
                            # Jika data base64
                            if content.startswith("data:image"):
                                import base64
                                content = content.split(",")[1]  # buang header
                                content = base64.b64decode(content)
                            else:
                                # Kalau bukan base64, ubah langsung ke bytes
                                content = content.encode("utf-8")

                        with open(file_path, "wb") as f:
                            f.write(content)
                        print("✅ Image saved:", file_path)

                        ws.send(json.dumps({
                            "event": "image_uploaded",
                            "path": file_path.replace("\\", "/")
                        }))
                    except Exception as e:
                        print("❌ Error saving image:", e)
                        ws.send(json.dumps({"error": str(e)}))


                threading.Thread(target=save_file).start()
                current_image_path = file_path

            except Exception as e:
                ws.send(json.dumps({"error": str(e)}))
            continue


        try:
            msg = json.loads(data)
            
            environ = ws.environ
            query = environ.get("QUERY_STRING", "")
            params = parse_qs(query)
            token = params.get("token", [None])[0]
            
            user_id = validate_token(token)
            print("Validated user_id:", user_id)
            print("Received token:", token)
            if not user_id:
                ws.send(json.dumps({"error": "Invalid or expired token"}))
                continue
            
            print("📨 Received raw message:", msg)

            user_text = msg.get("text")
            latitude = msg.get("latitude")
            longitude = msg.get("longitude")

            print("Received WS message:", msg)
            print("Reachived ws message:", len(msg))
            print("Perintah User", user_text)
            print("Image present:", {current_image_path})
            
            print("Location:", latitude, longitude)
            
            if latitude is None or longitude is None:
                ws.send(json.dumps({"error": "Latitude and Longitude are required"}))
                continue
            else:
                try:
                    location_info = get_place_info(latitude, longitude)
                    #ws.send(json.dumps({"token": f"Lokasi: {location_info}"}))
                except Exception as e:
                    ws.send(json.dumps({"error": f"Location error: {str(e)}"}))

            final_result = ""
                    
            for token in analyze_image(current_image_path, user_text, latitude, longitude):
                ws.send(json.dumps({"token": token}))
                final_result += token

            print("✅ Image Analysis Result:", final_result)
            
            threading.Thread(
                target=save_interaction_async,
                args=(user_id, user_text, final_result, current_image_path if current_image_path else None, latitude, longitude, location_info)
            ).start()    

            ws.send(json.dumps({"event": "end"}))
            continue

        except Exception as e:
            ws.send(json.dumps({"error": str(e)}))
            continue