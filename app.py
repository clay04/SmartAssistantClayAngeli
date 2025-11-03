import eventlet
import eventlet.wsgi
eventlet.monkey_patch()

from flask import Flask
from flask import send_from_directory
from flask_cors import CORS

from config import Config
from db import get_db, close_db
from extensions import mysql, bcrypt, jwt
import os

from routes.auth import auth_bp
from routes.voice import voice_bp, socketio
from routes.vision import vision_bp
from routes.location import location_bp
from routes.admin import admin_bp
from routes.users import users_bp
from routes.history import history_bp

app = Flask(__name__)
app.config.from_object(Config)
app.config['UPLOAD_FOLDER'] = 'uploads'

bcrypt.init_app(app)
jwt.init_app(app)

socketio.init_app(app, cors_allowed_origins="*", async_mode="eventlet")

CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}}, supports_credentials=True)


app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(voice_bp, url_prefix='/voice')
app.register_blueprint(vision_bp, url_prefix='/vision')
app.register_blueprint(location_bp, url_prefix='/location')
app.register_blueprint(admin_bp, url_prefix='/admin')
app.register_blueprint(users_bp, url_prefix='/users')
app.register_blueprint(history_bp, url_prefix='/history')

app.teardown_appcontext(close_db)

@app.route("/testdb")
def testdb():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT DATABASE();")
    result = cur.fetchone()
    cur.close()
    return {"connected_to": result}

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")

@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    try:
        with app.app_context():
            conn = get_db()
            cur = conn.cursor()
            cur.execute("SELECT DATABASE();")
            print("DB connection successful:", cur.fetchone())
            cur.close()
    except Exception as e:
        print("DB connection failed:", e)

    print("🚀 Starting Flask-SocketIO server on 0.0.0.0:5000 ...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)

