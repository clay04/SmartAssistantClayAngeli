import os, base64, threading
from flask import Blueprint
from flask_socketio import emit
from werkzeug.utils import secure_filename
from services.gemini_service import analyze_image
from services.location_service import get_place_info
from services.database_service import save_interaction_async
from services.token_service import validate_token
from extensions import socketio

voice_bp = Blueprint("voice", __name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@socketio.on("connect")
def handle_connect():
    print("✅ Client connected")

@socketio.on("disconnect")
def handle_disconnect():
    print("❌ Client disconnected")

@socketio.on("voice_message")
def handle_voice_message(data):
    print("📩 Received voice_message:", list(data.keys()))

    try:
        token = data.get("access_token")
        user_id = validate_token(token)
        if not user_id:
            emit("error", {"error": "Invalid or expired token"})
            return

        user_text = data.get("text", "")
        image_b64 = data.get("image")
        latitude = data.get("latitude")
        longitude = data.get("longitude")

        print(f"👤 User {user_id} | Pesan: {user_text}")

        if latitude is None or longitude is None:
            emit("error", {"error": "Latitude and Longitude are required"})
            return

        location_info = get_place_info(latitude, longitude)
        final_result = ""

        # simpan gambar sementara
        save_path = None
        if image_b64:
            image_data = base64.b64decode(image_b64)
            filename = secure_filename(f"user_{user_id}_{threading.get_ident()}.jpg")
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(save_path, "wb") as f:
                f.write(image_data)

        # jalankan proses Gemini di thread terpisah biar streaming aman
        def process_gemini():
            nonlocal final_result
            try:
                for token in analyze_image(save_path, user_text, latitude, longitude):
                    print("🟢 Emitting token ke client:", token)
                    socketio.emit("response_token", {"token": token})
                    #socketio.sleep(0)
                    final_result += token

                socketio.emit("end", {"event": "end"})

                threading.Thread(
                    target=save_interaction_async,
                    args=(user_id, user_text, final_result, save_path, latitude, longitude, location_info)
                ).start()

            except Exception as e:
                print("⚠️ Error di Gemini thread:", e)
                socketio.emit("error", {"error": str(e)})

        threading.Thread(target=process_gemini).start()

    except Exception as e:
        emit("error", {"error": str(e)})
