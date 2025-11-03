from flask import Blueprint, request, jsonify
from db import get_db
from services.token_service import validate_token
from datetime import timedelta
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity
from services.admin_service import save_admin_tokens, delete_admin_tokens, validate_admin_token, get_admin_by_username, create_admin
from services.auth_service import hash_password, check_password
from services.database_service import get_prompt_system, update_prompt_system, get_user_last_location

admin_bp = Blueprint("admin", __name__)

@admin_bp.route("/login", methods=["POST"])
def admin_login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    
    conn = get_db()
    admin = get_admin_by_username(conn, username)

    if not admin or not check_password(admin["password_hash"], password):
        return jsonify({"error": "Invalid username or password"}), 401
    
    access_token = create_access_token(
        identity=f"admin:{admin['id_admin']}",
        expires_delta=timedelta(hours=2)
    )
    
    refresh_token = create_refresh_token(
        identity=f"admin:{admin['id_admin']}",
        expires_delta=timedelta(days=7)
    )
    
    save_admin_tokens(conn, admin["id_admin"], access_token, refresh_token)
    
    return jsonify({
        "message": "Admin login successful",
        "access_token": access_token,
        "refresh_token": refresh_token,
        "admin": {
            "id_admin": admin["id_admin"],
            "username": admin["username"],
        }
    })
    
@admin_bp.route("/logout", methods=["POST"])
@jwt_required()
def admin_logout():
    identity = get_jwt_identity()
    if not str(identity).startswith("admin:"):
        return jsonify({"error": "Unauthorized"}), 403
    
    admin_id = identity.split(":")[1]
    conn = get_db()
    delete_admin_tokens(conn, admin_id)
    
    return jsonify({"message": "Admin logged out successfully"}), 200

@admin_bp.route("/verify", methods=["POST"])
def verify_admin_token():
    data = request.get_json()
    token = data.get("access_token")
    
    admin_id = validate_admin_token(token)
    if not admin_id:
        return jsonify({"error": "Invalid or expired token"}), 401
    
    return jsonify({"message": "Token valid", "admin_id": admin_id}), 200

@admin_bp.route("/register", methods=["POST"])
def admin_register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    full_name = data.get("full_name")

    if not all([username, password, full_name]):
        return jsonify({"error": "All fields are required"}), 400

    conn = get_db()

    # cek apakah username sudah ada
    existing_admin = get_admin_by_username(conn, username)
    if existing_admin:
        return jsonify({"error": "Username already exists"}), 409

    # hash password
    hashed_password = hash_password(password)

    # simpan ke database
    admin_id = create_admin(conn, full_name, username, hashed_password)

    # generate token JWT
    access_token = create_access_token(
        identity=f"admin:{admin_id}",
        expires_delta=timedelta(hours=4)
    )
    refresh_token = create_refresh_token(
        identity=f"admin:{admin_id}",
        expires_delta=timedelta(days=7)
    )

    # simpan token di tabel admin_tokens
    save_admin_tokens(conn, admin_id, access_token, refresh_token)

    return jsonify({
        "message": "Admin registration successful",
        "access_token": access_token,
        "refresh_token": refresh_token,
        "admin": {
            "id_admin": admin_id,
            "username": username,
            "full_name": full_name
        }
    }), 201
    
    
@admin_bp.route("/prompt", methods=['GET'])
@jwt_required()
def get_prompt():
    identity = get_jwt_identity()
    if not str(identity).startswith("admin:"):
        return jsonify({"error": "Unauthorized"}), 403
    
    conn = get_db()
    prompt = get_prompt_system(conn)
    if not prompt:
        return jsonify({"Prompt not found"})
    
    return jsonify({
        "message": "Prompt di dapatkan",
        "prompt": prompt
    })
    
@admin_bp.route("/prompt/update", methods=['PUT'])
@jwt_required()
def update_prompt():
    identity = get_jwt_identity()
    if not str(identity).startswith("admin:"):
        return jsonify({"error": "Unauthorized"}), 403
    
    data = request.json
    prompt_text = data.get('prompt_text')
    update_by = data.get('update_by', 'admin')
    conn = get_db()
    
    update_prompt_system(conn, prompt_text, update_by)
    
    return jsonify({
        "message": "Prompt updat successufuly",
    })


@admin_bp.route("/user-locations", methods=["GET"])
@jwt_required()
def get_user_locations():
    identity = get_jwt_identity()
    if not str(identity).startswith("admin:"):
        return jsonify({"error": "Unauthorized"}), 403

    conn = get_db()
    
    lst_location = get_user_last_location(conn)
    
    for r in lst_location:
        if isinstance(r.get("location_text"), str):
            try:
                r["location_text"] = json.loads(r["location_text"])
            except:
                r["location_text"] = None
    
    return jsonify({"locations": lst_location})
    