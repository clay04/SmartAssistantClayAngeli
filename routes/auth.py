from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity
from services.auth_service import hash_password, check_password
from services.user_services import create_user, get_user_by_username, save_tokens, get_refresh_token, delete_tokens
from db import get_db
from datetime import timedelta

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    first_name = data.get("first_name")
    last_name = data.get("last_name")
    username = data.get("username")
    password = data.get("password")

    if not all([first_name, last_name, username, password]):
        return jsonify({"error": "All fields are required"}), 400
    
    conn = get_db()
    
    if get_user_by_username(get_db(), username):
        return jsonify({"error": "Username already exists"}), 409
    
    hashed_password = hash_password(password)
    create_user(conn,first_name, last_name, username, hashed_password)
    
    return jsonify({"message": "User registered successfully"}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    conn = get_db()
    user = get_user_by_username(conn, username)
    
    if not user or not check_password(user["password"], password):
        return jsonify({"error": "Invalid username or password"}), 401
    
    access_token = create_access_token(
        identity=str(user["id_user"]),
        expires_delta=timedelta(hours=1)
    )
    
    refresh_token = create_refresh_token(
        identity=str(user["id_user"]),
        expires_delta=timedelta(days=7)
    )
    
    save_tokens(conn, user["id_user"], access_token, refresh_token)
    
    return jsonify({
        "message": "Login successful",
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": {
            "id_user": user["id_user"],
            "first_name": user["first_name"],
            "last_name": user["last_name"],
            "username": user["username"]
        }    
    }), 200

# REFRESH TOKEN
@auth_bp.route("/refresh", methods=["POST"])
@jwt_required(refresh=True)  # hanya refresh token yang valid
def refresh():
    data = request.get_json()
    refresh_token = data.get("refresh_token")
    
    if not refresh_token:
        return jsonify({"error": "Refresh token is required"}), 400
    
    conn = get_db()
    token_data = get_refresh_token(conn, refresh_token)
    
    if not token_data:
        return jsonify({"error": "Invalid or expired refresh token"}), 401
    
    new_access_token = create_access_token(
        identity=str(token_data["user_id"]),
        expires_delta=timedelta(hours=1)
    )
    
    return jsonify({
        "access_token": new_access_token
    }), 200

@auth_bp.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    user_id = get_jwt_identity()
    conn = get_db()
    delete_tokens(conn, user_id)
    return jsonify({"message": "Logged out successfully"}), 200


# PROTECTED ROUTE
@auth_bp.route("/profile", methods=["GET"])
@jwt_required()
def profile():
    user_id = get_jwt_identity()
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("SELECT id_user, first_name, last_name, username FROM users WHERE id_user = %s", (user_id,))
        user = cur.fetchone()
    return jsonify(user)

@auth_bp.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user_id = get_jwt_identity()
    return jsonify(logged_in_as=current_user_id), 200
