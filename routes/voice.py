import os, json, base64, tempfile, threading
from flask import Blueprint, request, jsonify
from services.gemini_service import speech_to_text, analyze_image
from services.location_service import get_place_info
from werkzeug.utils import secure_filename
from flask_sock import Sock
from pydub import AudioSegment
from services.database_service import save_interaction_async
from services.token_service import validate_token

voice_bp = Blueprint("voice", __name__)

sock = Sock()   

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@sock.route("/voice/ws")
def assistant_ws(ws):
    """
    WebSocket untuk real-time voice+image + streaming response dari Gemini
    """
    while True:
        data = ws.receive()
        if not data:
            break

        try:
            msg = json.loads(data)
            token = msg.get("access_token")
            user_id = validate_token(token)
            print("Validated user_id:", user_id)
            print("Received token:", token)
            if not user_id:
                ws.send(json.dumps({"error": "Invalid or expired token"}))
                continue
            
            audio_b64 = msg.get("audio")
            image_b64 = msg.get("image")
            #Location
            latitude = msg.get("latitude")
            longitude = msg.get("longitude")
            
            print("Received WS message:", msg)
            print("Reachived ws message:", len(msg))
            print("Audio present:", bool(audio_b64))
            print("Image present:", bool(image_b64))
            
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

            prompt_text = ""
            if audio_b64:
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    f.write(base64.b64decode(audio_b64))
                    f.flush()
                    try:
                        # biarkan ffmpeg autodetect format
                        audio = AudioSegment.from_file(f.name)
                        wav_path = f"{f.name}.wav"
                        audio.export(wav_path, format="wav")
                        prompt_text = speech_to_text(wav_path)
                    except Exception as e:
                        ws.send(json.dumps({"error": f"FFmpeg decode error: {str(e)}"}))
                        continue
                    
                    prompt_text = speech_to_text(f.name)
                    print("✅ STT Result:", prompt_text)

            final_result = ""
            
            if image_b64:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                    f.write(base64.b64decode(image_b64))
                    f.flush()
                    image_b64 = f.name
                    
            for token in analyze_image(f.name, prompt_text, latitude, longitude):
                ws.send(json.dumps({"token": token}))
                final_result += token

            print("✅ Image Analysis Result:", final_result)
            
            threading.Thread(
                target=save_interaction_async,
                args=(user_id, prompt_text, final_result, image_b64 if image_b64 else None, latitude, longitude, location_info)
            ).start()    

            ws.send(json.dumps({"event": "end"}))

        except Exception as e:
            ws.send(json.dumps({"error": str(e)}))