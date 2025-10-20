from flask import Flask
import eventlet
import eventlet.wsgi
from config import Config
from db import get_db, close_db
from extensions import mysql, bcrypt, jwt

from routes.auth import auth_bp
from routes.voice import voice_bp, sock
from routes.vision import vision_bp
from routes.location import location_bp
#from routes.admin import admin_bp

app = Flask(__name__)
app.config.from_object(Config)
app.config['UPLOAD_FOLDER'] = 'uploads'

bcrypt.init_app(app)
jwt.init_app(app)
sock.init_app(app)

app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(voice_bp, url_prefix='/voice')
app.register_blueprint(vision_bp, url_prefix='/vision')
app.register_blueprint(location_bp, url_prefix='/location')
#app.register_blueprint(admin_bp, url_prefix='/admin')

app.teardown_appcontext(close_db)

@app.route("/testdb")
def testdb():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT DATABASE();")
    result = cur.fetchone()
    cur.close()
    return {"connected_to": result}

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

    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 5000)), app)