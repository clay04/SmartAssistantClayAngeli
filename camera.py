import os
import time
import pygame
import cv2
import numpy as np
import tensorflow as tf
import google.generativeai as genai
from gtts import gTTS
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk

# Konfigurasi API Gemini
GEMINI_API_KEY = "AIzaSyDH9Q4m7C_u2dcPCybg9-rkfc5V76t10pY"
genai.configure(api_key=GEMINI_API_KEY)

# Load model deteksi objek (COCO SSD MobileNet)
model_path = os.path.abspath("ssd_mobilenet_v2_fpnlite_320x320/saved_model")
model = tf.saved_model.load(model_path)
category_index = {1: "orang", 2: "sepeda", 3: "mobil", 4: "motor", 5: "pesawat", 
                  6: "bus", 7: "kereta", 8: "truk", 9: "kapal", 44: "botol", 47: "sendok"}

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deteksi Objek dengan Chatbot")
        self.root.geometry("800x600")

        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        self.text_area = scrolledtext.ScrolledText(root, height=10)
        self.text_area.pack(pady=10)

        self.start_button = tk.Button(root, text="Start Kamera", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(root, text="Stop Kamera", command=self.stop_camera)
        self.stop_button.pack(side=tk.LEFT, padx=10)

        self.cap = None

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image='')

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                detected_objects = self.detect_objects(frame)

                for obj in detected_objects:
                    cv2.putText(frame, obj, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

                if detected_objects:
                    detected_text = ", ".join(detected_objects)
                    chatbot_response = self.get_gemini_response(f"Apa itu {detected_text}?")
                    self.text_area.insert(tk.END, f"Objek terdeteksi: {detected_text}\nChatbot: {chatbot_response}\n")

            self.root.after(10, self.update_frame)

    def detect_objects(self, frame):
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

    def get_gemini_response(self, user_input):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(user_input)
            return response.text.strip() if response.text else "Maaf, saya tidak bisa menjawab saat ini."
        except Exception as e:
            return f"Terjadi kesalahan saat menghubungi Gemini: {e}"

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
