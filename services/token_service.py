from db import get_db

def validate_token(token):
    conn = get_db()
    try :
        with conn.cursor() as cursor:
            cursor.execute("SELECT user_id FROM user_tokens WHERE acces_token = %s AND expired_at > NOW()", (token,))
            row = cursor.fetchone()
            if row:
                return row['user_id']
            
    except Exception as e:
        print(f"Token validation error: {e}")
    finally:
        conn.close()
    return None