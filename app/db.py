# app/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .models import Base, User

# ----- KẾT NỐI DB -----
# Nếu dự án cũ đã có URL rồi thì thay chuỗi này = URL cũ
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/app.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},  # chỉ cần cho SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Tạo bảng nếu chưa có
Base.metadata.create_all(bind=engine)


# ----- HELPER HÀM USER -----


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def get_user_for_login(db: Session, username: str, password: str) -> User | None:
    """
    Dùng cho login: kiểm tra username + password plain text trong DB.
    """
    return (
        db.query(User)
        .filter(User.username == username, User.password == password)
        .first()
    )


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()
