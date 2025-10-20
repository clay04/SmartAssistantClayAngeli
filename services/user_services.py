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

