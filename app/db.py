# app/db.py
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from .models import Base, User

SQLALCHEMY_DATABASE_URL = "sqlite:///./data/app.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},  # chỉ cần cho SQLite
)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def get_user_for_login(db: Session, username: str, password: str) -> User | None:
    return (
        db.query(User)
        .filter(User.username == username, User.password == password)
        .first()
    )

def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()
