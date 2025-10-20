from db import create_user_connection
from datetime import datetime
import threading, json

def save_interaction_async(user_id, text_input, response_text, image_b64, latitude, longitude, location_text, conn=None):
    own_connection = False
    if conn is None:
        conn = create_user_connection()
        own_connection = True
        
    if isinstance(location_text, dict):
        location_text = json.dumps(location_text)
                    
    if isinstance(response_text, dict):
        response_text = json.dumps(response_text)
        
    print(type(location_text), location_text)
        
    def task():
        try:
            with conn.cursor() as cur:
                cur.execute("""
                            INSERT INTO user_inputs (id_user, text_input, image_base64, latitude, longitude, location_text, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (user_id, text_input, image_b64, latitude, longitude, location_text, datetime.utcnow()))
                
                input_id = cur.lastrowid
                
                cur.execute("""
                            INSERT INTO gemini_responses (id_user, response_text, created_at, id_input)
                            VALUES (%s, %s, %s, %s)
                            """, (user_id, response_text, datetime.utcnow(), input_id))
                
            conn.commit()
            print(f"[async-save] saved input_id={input_id} for user_id={user_id}")
            print(f"[DB DEBUG] user_id={user_id}, prompt={text_input[:50]}, resp={response_text[:50]}")
                
        except Exception as e:
            print(f"Error saving interaction: {e}")
        finally:
            if own_connection:
                conn.close()
            
    threading.Thread(target=task, daemon=True).start()