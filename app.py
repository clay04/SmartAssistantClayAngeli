from flask import Flask
#from routes.auth import auth_bp
from routes.voice import voice_bp
from routes.vision import vision_bp
#from routes.admin import admin_bp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

#app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(voice_bp, url_prefix='/voice')
app.register_blueprint(vision_bp, url_prefix='/vision')
#app.register_blueprint(admin_bp, url_prefix='/admin')

if __name__ == '__main__':
    app.run(debug=True)