import os
from flask import Blueprint, request, jsonify
from services.gemini_service import speech_to_text, analyze_image
from werkzeug.utils import secure_filename

voice_bp = Blueprint("voice", __name__)

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
