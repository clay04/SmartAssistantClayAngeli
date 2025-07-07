import os
import time
import json
import pygame
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai

# Konfigurasi API Gemini
GEMINI_API_KEY = "AIzaSyDH9Q4m7C_u2dcPCybg9-rkfc5V76t10pY"  # Ganti dengan API key Anda
genai.configure(api_key=GEMINI_API_KEY)

# Fungsi untuk mendapatkan respons dari Gemini API
def get_gemini_response(user_input):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(user_input)
        return response.text.strip() if response.text else "Maaf, saya tidak bisa menjawab saat ini."
    except Exception as e:
        return f"Terjadi kesalahan saat menghubungi Gemini: {e}"

# Fungsi untuk mengubah teks menjadi suara
def speak(text):
    tts = gTTS(text=text, lang="id")
    filename = "response.mp3"
    tts.save(filename)
    
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        time.sleep(1)
    
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    os.remove(filename)

# Fungsi untuk mengubah suara menjadi teks
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Silakan bicara...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio, language="id-ID")
            print(f"Anda (suara): {text}")
            return text.lower()
        except sr.UnknownValueError:
            print("Maaf, saya tidak mengerti. Bisa ulangi?")
            return None
        except sr.RequestError:
            print("Maaf, ada masalah dengan layanan pengenalan suara.")
            return None

# Loop percakapan
print("Chatbot Menu Makanan aktif! Ketik 'exit' atau ucapkan 'keluar' untuk berhenti.")
while True:
    try:
        mode = input("Gunakan suara atau ketik? (suara/ketik): ").strip().lower()
        if mode == "suara":
            user_input = listen()
        else:
            user_input = input("Anda: ").strip().lower()
        
        if not user_input:
            continue
        
        if user_input in ["exit", "keluar"]:
            print("Chatbot: Terima kasih, sampai jumpa!")
            speak("Terima kasih, sampai jumpa!")
            break
        
        response = get_gemini_response(user_input)
        print(f"Chatbot: {response}")
        speak(response)
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
