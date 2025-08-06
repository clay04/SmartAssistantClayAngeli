import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', "AIzaSyDH9Q4m7C_u2dcPCybg9-rkfc5V76t10pY")