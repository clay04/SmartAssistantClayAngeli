def create_user(conn, username, hashed_password):
    """
    Simpan user baru ke tabel users
    """
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, password) VALUES (%s, %s)",
        (username, hashed_password)
    )
    conn.commit()
    cur.close()

def get_user_by_username(conn, username):
    """
    Ambil user berdasarkan username
    """
    cur = conn.cursor()
    cur.execute("SELECT id, username, password FROM users WHERE username = %s", (username,))
    row = cur.fetchone()
    cur.close()
    return row
