import io, hashlib, base64, tempfile
from google import generativeai as genai
from PIL import Image
from config import Config
import base64
import tempfile

from services.location_service import get_place_info

genai.configure(api_key=Config.GEMINI_API_KEY)

cache_result = {}

def get_file_hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
    
#Analyze Image
def analyze_image(image_path, user_text="", latitude=None, longitude=None):
    try:
        file_hash = get_file_hash(image_path)
        if file_hash in cache_result:
            print("Using cached image analysis result")
            return cache_result[file_hash]
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((512, 512))
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        
        # Get location context if coordinates provided
        location_context = ""
        if latitude and longitude:
            try:
                location_info = get_place_info(latitude, longitude)
                address = location_info["address"]["display_name"]
                nearby = ", ".join(
                    [f"{p['name']} ({p['type']})" for p in location_info["nearby_places"] if p.get("name")])
                location_context = f"Lokasi saat ini: {address}. Tempat terdekat: {nearby}."
            except Exception as e:
                location_context = f"Gagal mendapatkan info lokasi: {str(e)}"
        
        image_bs64 = base64.b64encode(image_bytes).decode('utf-8')
        
        guidance = (
            "Kamu adalah asisten visual untuk tunanetra.\n"
            "Jawablah pertanyaan pengguna dengan singkat, jelas, dan hanya berdasarkan isi gambar.\n"
            "Ikuti aturan berikut:\n\n"
            "1. Jika pertanyaan menanyakan lokasi/posisi benda, sebutkan dengan arah relatif "
            "(kanan, kiri, depan, tengah, belakang). Contoh: "
            "\"Tisu ada di sebelah kanan meja\" atau \"Tidak terlihat pada gambar\".\n"
            "2. Jika pertanyaan menanyakan teks/tulisan, bacakan teks yang terlihat. Jika tidak terbaca, "
            "jawab \"Tulisan tidak terbaca pada gambar\".\n"
            "3. Jika pertanyaan menanyakan kondisi/lingkungan, jelaskan ringkas objek penting "
            "dengan arah relatif juga.\n"
            "4. Jika informasi yang diminta tidak ada pada gambar, jawab singkat: "
            "\"Tidak terlihat pada gambar\".\n"
            "5. Jangan memberi deskripsi panjang kecuali diminta detail oleh pengguna.\n"
            "6. Gunakan bahasa sederhana agar mudah dipahami lewat pembacaan suara.\n\n"
            "Format jawaban: SATU kalimat, Bahasa Indonesia, langsung ke inti."
        )
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{guidance}. Pertanyaannya adalah: \n{user_text}"},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_bs64}},
                        {"text": f"ketika user bertanya mengenai lokasi atau lokasi sekitar baru ini di jawab. dan lokasinya adalah : {location_context}"},
                    ],
                }
            ], stream=True
        )
        
        final_text = ""
        for chunk in response:
            if hasattr(chunk, "text") and chunk.text:
                token = chunk.text.strip()
                final_text += token
                yield token  # streaming ke WS

        if not final_text:
            final_text = "Tidak ada respons dari Gemini."

        cache_result[file_hash] = final_text
        return final_text
        
    except Exception as e:
        return f"Error call Gemini: {str(e)}", 500
    
    
