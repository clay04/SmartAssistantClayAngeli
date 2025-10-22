from flask import Blueprint, request, jsonify
from db import get_db
from services.token_service import validate_token
from flask_jwt_extended import jwt_required, get_jwt_identity
from services.user_services import get_list_users, get_users_details, get_user_by_id_user
from services.session_service import get_user_session

users_bp = Blueprint("users", __name__)

@users_bp.route("/list", methods=["GET"])
@jwt_required()
def get_all_users():
    identity = get_jwt_identity()
    
    # Pastikan hanya admin yang bisa akses
    if not str(identity).startswith("admin:"):
        return jsonify({"error": "Unauthorized access"}), 403

    conn = get_db()
    search_query = request.args.get("search", "").strip()

    users = get_list_users(conn, search_query)
    
    return jsonify({
        "count": len(users),
        "users": users
    }), 200
    

@users_bp.route("/detail/<int:id_user>", methods=["GET"])
@jwt_required()
def get_user_details(id_user):
    identity = get_jwt_identity()
    
    # Pastikan hanya admin yang bisa akses
    if not str(identity).startswith("admin:"):
        return jsonify({"error": "Unauthorized access"}), 403

    conn = get_db()

    user = get_users_details(conn, id_user)
    
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "meassage": "User details fetched successfully",
        "user": user
    }), 200
    
@users_bp.route("/delete/<int:id_user>", methods=["DELETE"])
@jwt_required()
def delete_user(id_user):
    identity = get_jwt_identity()
    
    # Pastikan hanya admin yang bisa akses
    if not str(identity).startswith("admin:"):
        return jsonify({"error": "Unauthorized access"}), 403

    conn = get_db()
    user = get_user_by_id_user(conn, id_user)
    if not user:
        return jsonify({"error": "User not found"}), 404
    delete_user(conn, id_user)
    
    return jsonify({"message": "User deleted successfully"}), 200

@users_bp.route("/session", methods=["GET"])
@jwt_required()
def user_session():
    identity = get_jwt_identity()
    if not str(identity).startswith("admin:"):
        return jsonify({"error": "Unautorized access"}), 403
    
    conn = get_db()
    token = request.args.get("search", "").strip()
    
    session = get_user_session(conn, token)
    
    return jsonify({
        "count": len(session),
        "session" : session
    }), 200
    