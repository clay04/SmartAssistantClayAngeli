from datetime import datetime, timedelta

def create_user(conn, first_name, last_name, username, hashed_password):
    """
    Simpan user baru ke tabel users
    """
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (first_name, last_name, username, password) VALUES (%s, %s, %s, %s)",
        (first_name, last_name, username, hashed_password)
    )
    user_id = cur.lastrowid
    conn.commit()
    cur.close()
    return user_id

def update_user_data(conn, id_user, first_name, last_name):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE users
            SET first_name=%s, last_name=%s, updated_at=NOW()
            WHERE id_user=%s
        """, (first_name, last_name, id_user))
    conn.commit()

def get_user_by_username(conn, username):
    """
    Ambil user berdasarkan username
    """
    cur = conn.cursor()
    cur.execute("SELECT id_user, first_name, last_name, username, password FROM users WHERE username = %s", (username,))
    row = cur.fetchone()
    cur.close()
    return row

def save_tokens(conn, user_id, access_token, refresh_token, expires_days=7):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO user_tokens (user_id, access_token, refresh_token, expired_at)
            VALUES (%s, %s, %s, %s)
            """,
            (user_id, access_token, refresh_token, datetime.utcnow() + timedelta(days=expires_days))
        )
    conn.commit()


def get_refresh_token(conn, refresh_token):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM user_tokens WHERE refresh_token = %s AND expired_at > NOW()",
            (refresh_token,)
        )
        return cur.fetchone()


def delete_tokens(conn, user_id):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM user_tokens WHERE user_id = %s", (user_id,))
    conn.commit()

def get_list_users(conn, search_query):
    with conn.cursor() as cur:
        if search_query:
            cur.execute("""
                SELECT id_user, first_name, last_name, username, created_at
                FROM users
                WHERE first_name LIKE %s OR last_name LIKE %s OR username LIKE %s
                ORDER BY created_at DESC
            """, (f"%{search_query}%", f"%{search_query}%", f"%{search_query}%"))
        else:
            cur.execute("""
                SELECT id_user, first_name, last_name, username, created_at
                FROM users
                ORDER BY created_at DESC
            """)
        
        return cur.fetchall()
    
def get_users_details(conn, id_user):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT *
            FROM users
            WHERE id_user = %s
        """, (id_user,))
        return cur.fetchone()
        
def get_user_by_id_user(conn, id_user):
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE id_user = %s", (id_user,))
        return cur.fetchone()
        
def delete_user(conn, id_user):
    with conn.cursor() as cur:
        # Hapus Relasi
        cur.execute("DELETE FROM gemini_responses WHERE id_user = %s", (id_user,))
        cur.execute("DELETE FROM user_inputs WHERE id_user = %s", (id_user,))
        cur.execute("DELETE FROM user_tokens WHERE user_id = %s", (id_user,))
        
        # Hapus user
        cur.execute("DELETE FROM users WHERE id_user = %s", (id_user,))
    conn.commit()
    return True