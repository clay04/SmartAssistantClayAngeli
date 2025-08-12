import io
from google import generativeai as genai
from PIL import Image
from config import Config
import speech_recognition as sr
from pydub import AudioSegment
import tempfile


genai.configure(api_key=Config.GEMINI_API_KEY)

def speech_to_text(audio_file):
    try: 
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_file.save(temp_input.name)
        
        sound = AudioSegment.from_file(temp_input.name)
        sound = sound.set_channels(1).set_frame_rate(16000)
        converted_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sound.export(converted_path.name, format='wav')
        
        r = sr.Recognizer()
        with sr.AudioFile(converted_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data, language='id-ID')
            return text
        
    except sr.UnknownValueError:
        return "Tidak dapat mengenali ucapan"
    except sr.RequestError as e:
        return f"Kesalahan dalam permintaan: {e}"
    except Exception as e:
        return f"STT Error: {e}"
    
#Analyze Image

def analyze_image(image_file, prompt_text):
    try:
        image = Image.open(image_file.stream).convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([
            f"Jawablah hanya dalam 1 kalimat yang sangat singkat, jelas, dan langsung ke inti. Hindari penjelasan panjang. Pertanyaannya:\n{prompt_text}",
            {
                "mime_type": "image/jpeg",
                "data": image_bytes,
            }
        ])
        
        return response.text.strip() if response.text else "No response text available"
    except Exception as e:
        return f"Error call Gemini: {str(e)}", 500
    