# extensions.py
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_socketio import SocketIO

mysql = MySQL()
bcrypt = Bcrypt()
jwt = JWTManager()

socketio = SocketIO(
    cors_allowed_origins="*",
    async_mode="eventlet",
    ping_timeout=70,
    ping_interval=25,
)
