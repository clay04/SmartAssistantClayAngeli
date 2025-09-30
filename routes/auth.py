from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from services.auth_service import hash_password, check_password
from services.user_services import create_user, get_user_by_username
from db import get_db

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    
    hashed_password = hash_password(password)
    conn = get_db()
    
    create_user(conn, username, hashed_password)
    
    return jsonify({"message": "User registered successfully"}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    conn = get_db()
    user = get_user_by_username(conn, username)
    
    if not user or not check_password(user["password"], password):
        return jsonify({"error": "Invalid username or password"}), 401
    
    token = create_access_token(identity=username)
    return jsonify({"token": token}), 200

# REFRESH TOKEN
@auth_bp.route("/refresh", methods=["POST"])
@jwt_required(refresh=True)  # hanya refresh token yang valid
def refresh():
    current_user = get_jwt_identity()
    new_access_token = create_access_token(identity=current_user)
    return jsonify({"access_token": new_access_token}), 200

@auth_bp.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200
