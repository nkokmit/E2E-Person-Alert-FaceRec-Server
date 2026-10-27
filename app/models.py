# app/models.py
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, Text, Boolean

Base = declarative_base()


class User(Base):
    __tablename__ = "users"  

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    password = Column(String(128), nullable=False)
    is_admin = Column(Boolean, default=False)

    def check_password(self, password: str) -> bool:
        return self.password == password


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True)
    evt_id = Column(String(64), index=True, nullable=False, unique=True)
    ts = Column(Float, index=True, nullable=False)

    camera_id = Column(String(64), index=True, nullable=False)
    source = Column(String(32), default="esp32")
    bbox = Column(Text)
    snapshot_url = Column(Text)
    person_id = Column(String(64), nullable=True)

    duration_sec = Column(Float, nullable=True)
    alert_type = Column(String(16), nullable=True)
    confidence = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)

