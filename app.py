from flask import Flask
import eventlet
import eventlet.wsgi
#from routes.auth import auth_bp
from routes.voice import voice_bp, sock
from routes.vision import vision_bp
from routes.location import location_bp
#from routes.admin import admin_bp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

#app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(voice_bp, url_prefix='/voice')
app.register_blueprint(vision_bp, url_prefix='/vision')
app.register_blueprint(location_bp, url_prefix='/location')
#app.register_blueprint(admin_bp, url_prefix='/admin')

sock.init_app(app)

if __name__ == "__main__":
    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 5000)), app)
