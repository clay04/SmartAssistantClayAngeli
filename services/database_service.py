from db import get_db
from datetime import datetime
import threading

def save_interaction_async(user_id, input_id, text_input, image_b64, latitude, longitude, location_text, response_text):
    def task():
        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                            INSERT INTO user_inputs (id_user, text_input, image_b64, latitude, longitude, location_text, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (user_id, text_input, image_b64, latitude, longitude, location_text, datetime.utcnow()))
                
                cur.execute("""
                            INSERT INTO gemini_responses (id_user, id_input, response_text, created_at)
                            VALUES (%s, %s, %s, %s)
                            """, (user_id, input_id, response_text, datetime.utcnow()))
                
                conn.commit()
                print(f"[async-save] saved input_id={input_id} for user_id={user_id}")
                
        except Exception as e:
            print(f"Error saving interaction: {e}")
        finally:
            conn.close()
            
    threading.Thread(target=task, daemon=True).start()