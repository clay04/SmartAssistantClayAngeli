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
    
def get_prompt_system(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM prompt ORDER BY id_prompt DESC LIMIT 1")
    return cur.fetchall()

def update_prompt_system(conn, prompt_text, update_by):
    with conn.cursor() as cur:
        cur.execute("""UPDATE prompt 
        SET prompt_text=%s, update_by=%s, update_at=NOW()
        ORDER BY id_prompt DESC LIMIT 1
        """, (prompt_text, update_by))
    conn.commit()
    
def get_user_last_location(conn):
    with conn.cursor() as cur:
        cur.execute("""
                    SELECT 
                        u.id_user, 
                        u.username, 
                        i.latitude, 
                        i.longitude, 
                        i.location_text, 
                        i.created_at
                    FROM user_inputs i
                    JOIN users u ON i.id_user = u.id_user
                    INNER JOIN (
                        SELECT id_user, MAX(created_at) AS latest
                        FROM user_inputs
                        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                        GROUP BY id_user
                    ) AS last_input 
                    ON i.id_user = last_input.id_user AND i.created_at = last_input.latest
                    ORDER BY i.created_at DESC
                    """)
    return cur.fetchall()
