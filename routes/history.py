from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from services.history_service import get_all_history
from db import get_db

history_bp = Blueprint("history", __name__)

@history_bp.route("/all", methods=["GET"])
@jwt_required()
def get_history():
    identity = get_jwt_identity()
    if not str(identity).startswith("admin:"):
        return jsonify({"error": "Unauthorized access"}), 403
    
    conn = get_db()
    search = request.args.get("search", "").strip()
    
    history = get_all_history(conn, search)
    
    return jsonify({
        "count": len(history),
        "history": history
    }), 200
    
