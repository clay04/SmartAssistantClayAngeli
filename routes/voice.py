import os, base64, threading
from flask import Blueprint
from flask_socketio import emit
from werkzeug.utils import secure_filename
from flask import copy_current_request_context

from services.gemini_service import analyze_image
from services.database_service import save_interaction_async
from services.token_service import validate_token
from extensions import socketio
from db import get_db  # untuk ambil location_text dari user_input

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
        # --- 1. Validasi token ---
        token = data.get("access_token")
        user_id = validate_token(token)
        if not user_id:
            emit("error", {"error": "Invalid or expired token"})
            return

        # --- 2. Ambil data dasar ---
        user_text = data.get("text", "")
        image_b64 = data.get("image")

        print(f"👤 User {user_id} | Pesan: {user_text}")

        # --- 3. Ambil lokasi terakhir dari database ---
        db = get_db()
        cur = db.cursor()
        cur.execute("""
            SELECT latitude, longitude, location_text 
            FROM user_inputs WHERE id_user=%s
        """, (user_id,))
        loc_row = cur.fetchone()
        cur.close()

        if not loc_row or not loc_row["latitude"] or not loc_row["longitude"]:
            emit("error", {"error": "Lokasi belum tersedia, kirimkan lokasi dulu."})
            return

        latitude = loc_row["latitude"]
        longitude = loc_row["longitude"]
        location_context = loc_row["location_text"]
        print(f"📍 Menggunakan lokasi dari DB: {location_context}")

        # --- 4. Simpan gambar sementara ---
        save_path = None
        if image_b64:
            image_data = base64.b64decode(image_b64)
            filename = secure_filename(f"user_{user_id}_{threading.get_ident()}.jpg")
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(save_path, "wb") as f:
                f.write(image_data)

        # --- 5. Jalankan Gemini di thread terpisah biar streaming lancar ---
        @copy_current_request_context
        def process_gemini():
            final_result = ""
            try:
                for token in analyze_image(
                    image_path=save_path,
                    user_text=user_text,
                    user_id=user_id  # Gemini akan ambil location_text langsung dari DB
                ):
                    print("🟢 Streaming token:", token)
                    socketio.emit("response_token", {"token": token})
                    final_result += token

                socketio.emit("end", {"event": "end"})

                threading.Thread(
                    target=save_interaction_async,
                    args=(user_id, user_text, final_result, save_path, latitude, longitude, location_context)
                ).start()

                print(f"✅ Gemini selesai untuk user {user_id}")

            except Exception as e:
                print("💥 Error Gemini thread:", e)
                socketio.emit("error", {"error": str(e)})

        threading.Thread(target=process_gemini).start()

    except Exception as e:
        print("💥 Error handle_voice_message:", e)
        emit("error", {"error": str(e)})
