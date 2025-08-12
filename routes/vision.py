from flask import Blueprint, request, jsonify
from services.gemini_service import analyze_image

vision_bp = Blueprint('vision', __name__)

@vision_bp.route('/analyze', methods=['POST'])
def image_analyze():
    image = request.files.get('image')
    
    if not image:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        result = analyze_image(image)
        return jsonify({'response': result}), 200
    except Exception as e:
        return jsonify({'error': f'Gagal menganalisis gambar: {e}'}), 500