import os, json, base64, tempfile
from flask import Blueprint, request, jsonify
from services.gemini_service import speech_to_text, analyze_image
from werkzeug.utils import secure_filename
from flask_sock import Sock
from pydub import AudioSegment
from services.token_service import validate_token
from services.database_service import save_interaction_async

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
            audio_b64 = msg.get("audio")
            image_b64 = msg.get("image")

            print("🔍 Received WS message:", msg)
            print("Reachived ws message:", len(msg))
            #print("🔍 Audio present:", bool(audio_b64))
            #print("🔍 Image present:", bool(image_b64))

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
                    #print("✅ STT Result:", prompt_text)

            result_text = ""
            if image_b64:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                    f.write(base64.b64decode(image_b64))
                    f.flush()
                    #result_text = analyze_image(f.name, prompt_text)
                    for result_text in analyze_image(f.name, prompt_text):
                        ws.send(json.dumps({"image_token": result_text}))
                    
                    if hasattr(result_text, "__iter__") and not isinstance(result_text, str):
                        result_text = "".join(result_text)
                    else:
                        result_text = result_text
                        
                    print("✅ Image Analysis Result:", result_text)


            ws.send(json.dumps({"event": "end"}))

        except Exception as e:
            ws.send(json.dumps({"error": str(e)}))