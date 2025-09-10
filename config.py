import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', "AIzaSyAJXtFXt0Yacbvst6OS4JcWQ1Gv4uUXN3E")