from flask import Blueprint, request, jsonify, send_file
from services.gemini_service import speech_to_text, analyze_image

voice_bp = Blueprint('voice', __name__)

@voice_bp.route('/assistant', methods=['POST'])
def voice_assistant():
    audio = request.files.get('audio')

    if not audio:
        return jsonify({'error': 'Audio file not provided'}), 400

    try:
        prompt_text = speech_to_text(audio)
        
        result = analyze_image(prompt_text)
        
        return jsonify({
            'recognized_text': prompt_text,
            'response': result
        }), 200

    except Exception as e:
        return jsonify({"error": f"Gagal memproses suara: {e}"}), 500
