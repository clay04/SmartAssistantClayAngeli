# extensions.py
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager

mysql = MySQL()
bcrypt = Bcrypt()
jwt = JWTManager()
