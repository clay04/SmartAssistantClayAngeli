def get_user_session(conn, token):
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM user_tokens WHERE expired_at > NOW()")
        return cur.fetchall()
    