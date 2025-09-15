import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', "AIzaSyDH9Q4m7C_u2dcPCybg9-rkfc5V76t10pY")
    GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', "AIzaSyAY9iCrbwGpcw3r40EiNSRT1XL0-O8qxGo")
    LOCATIONIQ_API_KEY = os.getenv('LOCATIONIQ_API_KEY', "pk.d5979551cf32542dd30a8a88acabd113")
    OPENCAGE_API_KEY = os.getenv('OPENCAGE_API_KEY', "19b89e38e01444e493ace9f541f2d3d7")