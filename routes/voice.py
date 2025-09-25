import os, json, base64, tempfile
from flask import Blueprint, request, jsonify
from services.gemini_service import speech_to_text, analyze_image
from services.location_service import get_place_info
from werkzeug.utils import secure_filename
from flask_sock import Sock
from pydub import AudioSegment

voice_bp = Blueprint("voice", __name__)

sock = Sock()   

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@voice_bp.route("/assistant", methods=["POST"])
def voice_assistant():
    print("🔍 request.files:", request.files)
    print("🔍 request.form:", request.form)
    
    audio_file = request.files.get("audio")
    image_file = request.files.get("image")
    
    print(image_file)
    print(audio_file)

    if not audio_file and not image_file:
        print("❌ No audio or image file provided")
        return jsonify({"error": "Audio & Image file not provided"}), 400

    try:
        audio_path, image_path = None, None

        if audio_file:
            audio_filename = secure_filename(audio_file.filename)
            audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
            audio_file.save(audio_path)
            print("✅ Audio saved:", audio_path)

        if image_file:
            image_filename = secure_filename(image_file.filename)
            image_path = os.path.join(UPLOAD_FOLDER, image_filename)
            image_file.save(image_path)
            print("✅ Image saved:", image_path)

        # Jalankan STT kalau ada audio
        prompt_text = ""
        if audio_path:
            prompt_text = speech_to_text(audio_path)

        # Analisa gambar kalau ada
        result = ""
        if image_path:
            result = analyze_image(image_path, prompt_text)

        return jsonify({
            "recognized_text": prompt_text,
            "response": result
        }), 200

    except Exception as e:
        return jsonify({"error": f"Gagal memproses suara/gambar: {str(e)}"}), 500

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
            audio_b64 = msg.get("audio")
            image_b64 = msg.get("image")
            
            print("Received WS message:", msg)
            print("Reachived ws message:", len(msg))
            print("Audio present:", bool(audio_b64))
            print("Image present:", bool(image_b64))
            
            #Location
            latitude = msg.get("latitude")
            longitude = msg.get("longitude")
            
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
            for token in analyze_image(f.name, prompt_text, latitude, longitude):
                ws.send(json.dumps({"token": token}))
                final_result += token

            print("✅ Image Analysis Result:", final_result)



            ws.send(json.dumps({"event": "end"}))

        except Exception as e:
            ws.send(json.dumps({"error": str(e)}))