import os
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', "AIzaSyAJXtFXt0Yacbvst6OS4JcWQ1Gv4uUXN3E")
    GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', "AIzaSyAY9iCrbwGpcw3r40EiNSRT1XL0-O8qxGo")
    LOCATIONIQ_API_KEY = os.getenv('LOCATIONIQ_API_KEY', "pk.d5979551cf32542dd30a8a88acabd113")
    OPENCAGE_API_KEY = os.getenv('OPENCAGE_API_KEY', "19b89e38e01444e493ace9f541f2d3d7")
    
    # Mysql
    MYSQL_HOST = "localhost"
    MYSQL_PORT = 3306
    MYSQL_USER = "root"
    MYSQL_PASSWORD = ""
    MYSQL_DB = "db_smart_assistant"
    
    # JWT
    JWT_SECRET_KEY = "a3f2c12b9d8e1c4e8f43b2a9fdd5a8c6d72be9f1c7a2b915ec3f8a0df13b2e5c"
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=7)
    
