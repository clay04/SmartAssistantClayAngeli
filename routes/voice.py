from flask import Blueprint, request, jsonify, send_file
from services.gemini_service import speech_to_text, text_to_speech
import tempfile
from google import generativeai as genai

voice_bp = Blueprint('voice', __name__)

@voice_bp.route('/assistant', methods=['POST'])
def voice_assistant():
    audio = request.files.get('audio')

    if not audio:
        return jsonify({'error': 'Audio file not provided'}), 400

    try:
        # 1. STT
        recognized_text = speech_to_text(audio)

        # 2. Langsung ke Gemini (tanpa lewat ask_text)
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = (
            "Jawablah hanya dalam satu kalimat yang sangat jelas, ringkas, dan langsung ke intinya. "
            "Hindari simbol, format markdown, atau penjelasan teknis.\n"
            f"{recognized_text}"
        )
        response = model.generate_content(prompt)
        response_text = response.text.strip() if response.text else "Tidak ada jawaban."

        # 3. TTS
        audio_path = text_to_speech(response_text)

        return send_file(audio_path, mimetype='audio/mpeg', as_attachment=False,
                         download_name="response.mp3",
                         headers={
                             "X-Recognized-Text": recognized_text,
                             "X-Response-Text": response_text
                         })

    except Exception as e:
        return jsonify({"error": f"Gagal memproses suara: {e}"}), 500
