import io
from google import generativeai as genai
from PIL import Image
from config import Config
import speech_recognition as sr
from pydub import AudioSegment
import base64
import tempfile


genai.configure(api_key=Config.GEMINI_API_KEY)

# Speech to Text
def speech_to_text(audio_path):
    try:         
        sound = AudioSegment.from_file(audio_path)
        sound = sound.set_channels(1).set_frame_rate(16000)
        
        converted_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sound.export(converted_path.name, format='wav')
        
        r = sr.Recognizer()
        with sr.AudioFile(converted_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data, language='id-ID')
            print("Recognized Text:", text)
            return text
        
    except sr.UnknownValueError:
        return "Tidak dapat mengenali ucapan"
    except sr.RequestError as e:
        return f"Kesalahan dalam permintaan: {e}"
    except Exception as e:
        return f"STT Error: {e}"
    
#Analyze Image
def analyze_image(image_path, prompt_text=""):
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        
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
                        {"text": f"{guidance}. Pertanyaannya adalah: \n{prompt_text}"},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_bs64}},
                    ],
                }
            ]
        )
        
        return response.text.strip() if response.text else "No response text available"
    except Exception as e:
        return f"Error call Gemini: {str(e)}", 500
    