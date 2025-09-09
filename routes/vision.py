import os
from flask import Blueprint, request, jsonify
from services.gemini_service import analyze_image
from werkzeug.utils import secure_filename

vision_bp = Blueprint('vision', __name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@vision_bp.route('/analyze', methods=['POST'])
def image_analyze():
    
    image_file = request.files.get('image')
    
    if not image_file:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        image_filename = secure_filename(image_file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        image_file.save(image_path)
        
        result = analyze_image(image_path)
        return jsonify({'response': result}), 200
    
    except Exception as e:
        return jsonify({'error': f'Gagal menganalisis gambar: {e}'}), 500