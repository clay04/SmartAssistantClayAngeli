import io
from google import generativeai as genai
from PIL import Image
from config import Config
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
import tempfile

genai.configure(api_key=Config.GEMINI_API_KEY)

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
    
# TEXT-TO-SPEECH (Google TTS)
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='id')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        return f"Error generating audio: {str(e)}", 500
    

# SPEECH-TO-TEXT (Google SpeechRecognition)
def speech_to_text(audio_file):
    try:
        # Simpan audio sementara
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_file.save(temp_input.name)

        # Konversi ke format standar
        sound = AudioSegment.from_file(temp_input.name)
        sound = sound.set_channels(1).set_frame_rate(16000)
        converted_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sound.export(converted_path, format="wav")

        # Proses STT
        r = sr.Recognizer()
        with sr.AudioFile(converted_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data, language="id-ID")
            return text
    except sr.UnknownValueError:
        raise Exception("Suara tidak dikenali.")
    except sr.RequestError:
        raise Exception("Koneksi ke layanan Google Speech Recognition gagal.")
    except Exception as e:
        raise Exception(f"STT error: {e}")