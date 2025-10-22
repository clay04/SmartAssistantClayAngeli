def get_user_session(conn, token):
    with conn.cursor() as cur:
        cur.execute("SELECT user_id FROM user_token WHERE access_token = %s AND expired_at > NOW()", (token))
        return cur.fetchall()
    