import os
import time
import json
import pygame
import speech_recognition as sr
import cv2
import numpy as np
import tensorflow as tf
from google import generativeai as genai
from google.generativeai.types import content_types as types
import tempfile


# Konfigurasi API Gemini
GEMINI_API_KEY = "AIzaSyDH9Q4m7C_u2dcPCybg9-rkfc5V76t10pY"
genai.configure(api_key=GEMINI_API_KEY)

# Load model deteksi objek (COCO SSD MobileNet)
model_path = os.path.abspath("ssd_mobilenet_v2_fpnlite_320x320/saved_model")
model = tf.saved_model.load(model_path)
category_index = {1: "orang", 2: "sepeda", 3: "mobil", 4: "motor", 5: "pesawat", 
                  6: "bus", 7: "kereta", 8: "truk", 9: "kapal", 44: "botol", 47: "sendok"}

# Fungsi untuk mendapatkan respons dari Gemini API
def get_gemini_response(user_input):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = (
            "Jawablah dengan bahasa Indonesia yang sangat sederhana, jelas, dan langsung ke inti. "
            "Jangan gunakan format markdown, simbol seperti tanda bintang (*), atau penjelasan yang terlalu teknis. "
            "dan jangan membuat jawaban di bold atau di italic dan di garis bawahi"
            "Jawaban maksimal 2 kalimat. Jika tidak tahu, katakan tidak tahu. Pertanyaannya:\n"
            f"{user_input}"
        )
        
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "Maaf, saya tidak bisa menjawab saat ini."
    except Exception as e:
        return f"Terjadi kesalahan saat menghubungi Gemini: {e}"

# Fungsi untuk mengubah teks menjadi suara
def speak(text):
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=f"Katakan dengan suara ramah: {text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name='Kore',  # Ganti ke voice Indonesia jika ada
                    )
                )
            ),
        )
    )

    data = response.candidates[0].content.parts[0].inline_data.data

    # Simpan file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(data)
        tmp_filename = tmp.name

    pygame.mixer.init()
    pygame.mixer.music.load(tmp_filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(1)

    pygame.mixer.music.stop()
    pygame.mixer.quit()

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

# Fungsi untuk deteksi objek menggunakan kamera
def detect_objects(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detection_classes = detections['detection_classes'][0][:num_detections].numpy().astype(int)
    detection_scores = detections['detection_scores'][0][:num_detections].numpy()
    
    detected_objects = []
    for i in range(num_detections):
        class_id = detection_classes[i]
        score = detection_scores[i]

        if score > 0.5 and class_id in category_index:
            detected_objects.append(category_index[class_id])
        
    return detected_objects

# Fungsi utama untuk kamera real-time
def run_camera():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        detected_objects = detect_objects(frame)
        
        # Tampilkan objek yang terdeteksi di layar
        for obj in detected_objects:
            cv2.putText(frame, obj, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Deteksi Objek", frame)

        if detected_objects:
            detected_text = ", ".join(detected_objects)
            print(f"Objek terdeteksi: {detected_text}")
            chatbot_response = get_gemini_response(f"Apa itu {detected_text}?")
            print(f"Chatbot: {chatbot_response}")
            speak(chatbot_response)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Tekan 'q' untuk keluar
            break

    cap.release()
    cv2.destroyAllWindows()

# Loop percakapan chatbot
print("Chatbot dengan Deteksi Objek aktif! Ketik 'exit' atau ucapkan 'keluar' untuk berhenti.")
while True:
    try:
        mode = input("Gunakan suara, ketik, atau kamera? (suara/ketik/kamera): ").strip().lower()
        if mode == "suara":
            user_input = listen()
        elif mode == "kamera":
            run_camera()
            continue
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
