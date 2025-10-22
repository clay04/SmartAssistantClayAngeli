def get_all_history(conn, search):
    with conn.cursor() as cur:
        if search:
            cur.execute("""
                SELECT 
                    ui.id_input,
                    u.id_user,
                    CONCAT(u.first_name, ' ', u.last_name) AS full_name,
                    u.username,
                    ui.text_input,
                    ui.location_text,
                    gr.response_text,
                    ui.created_at AS input_time,
                    gr.created_at AS response_time
                FROM user_inputs ui
                LEFT JOIN gemini_responses gr ON ui.id_input = gr.id_input
                LEFT JOIN users u ON ui.id_user = u.id_user
                WHERE u.username LIKE %s OR u.first_name LIKE %s OR u.last_name LIKE %s
                ORDER BY ui.created_at DESC
            """, (f"%{search}%", f"%{search}%", f"%{search}%"))
        else:
            cur.execute("""
                SELECT 
                    ui.id_input,
                    u.id_user,
                    CONCAT(u.first_name, ' ', u.last_name) AS full_name,
                    u.username,
                    ui.text_input,
                    ui.location_text,
                    gr.response_text,
                    ui.created_at AS input_time,
                    gr.created_at AS response_time
                FROM user_inputs ui
                LEFT JOIN gemini_responses gr ON ui.id_input = gr.id_input
                LEFT JOIN users u ON ui.id_user = u.id_user
                ORDER BY ui.created_at DESC
            """)
            
        return cur.fetchall()