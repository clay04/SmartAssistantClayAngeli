from flask_bcrypt import Bcrypt

bycript = Bcrypt()

def hash_password(password):
    return bycript.generate_password_hash(password).decode('utf-8')

def check_password(hashed_password : str, plain_password : str) -> bool:
    return bycript.check_password_hash(hashed_password, plain_password)