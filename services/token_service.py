from db import create_user_connection

def validate_token(token, conn=None):
    own_connection = False
    if conn is None:
        conn = create_user_connection()
        own_connection = True
        
    try :
        with conn.cursor() as cursor:
            cursor.execute("SELECT user_id FROM user_tokens WHERE access_token = %s AND expired_at > NOW()", (token,))
            row = cursor.fetchone()
            if row:
                return row['user_id']
            
    except Exception as e:
        print(f"Token validation error: {e, token}")
    finally:
        if own_connection:
            conn.close()
    return None