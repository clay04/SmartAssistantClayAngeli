from datetime import datetime, timedelta
from db import get_db

def get_admin_by_username(conn, username):
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM admin_users WHERE username = %s", (username,))
        return cur.fetchone()

def create_admin(conn, full_name, username, password_hash):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO admin_users (full_name, username, password_hash, created_at) VALUES (%s, %s, %s, NOW())",
            (full_name, username, password_hash)
        )
        conn.commit()
        return cur.lastrowid

def save_admin_tokens(conn, admin_id, access_token, refresh_token=None, expires_in_hours=7):
    expired_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO admin_tokens (admin_id, access_token, refresh_token, expired_at)
            VALUES (%s, %s, %s, %s)
            """,
            (admin_id, access_token, refresh_token, expired_at)
        )
    conn.commit()
    
def validate_admin_token(token):
    conn = get_db()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT admin_id FROM admin_tokens WHERE access_token = %s AND expired_at > NOW()", (token,))
            row = cursor.fetchone()
            if row:
                return row['admin_id']
    except Exception as e:
        print(f"Admin token validation error: {e, token}")
    finally:
        conn.close()
    return None

def delete_admin_tokens(admin_id):
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM admin_tokens WHERE admin_id = %s", (admin_id,))
    conn.commit()
    conn.close()