import jwt
from datetime import datetime, timedelta
from app.core.config import settings

def create_test_token():
    payload = {
        "sub": "test_user_id",
        "email" : "test_user@example.com",
        "exp": datetime.utcnow() + timedelta(hours=1),  # Token expires in 1 hour
        "prefs": {"language":"java"}
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return token

if __name__ == "__main__":
    print(create_test_token())