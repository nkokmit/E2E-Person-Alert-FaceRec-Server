# app/models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.db import Base

class Person(Base):
    __tablename__ = "persons"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    faces = relationship("FaceEmbedding", back_populates="person")

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id"))
    # lưu text base64/npz path tạm thời cho đơn giản
    embedding: Mapped[str] = mapped_column(Text)
    person = relationship("Person", back_populates="faces")

class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    evt_id: Mapped[str] = mapped_column(String(32), index=True, unique=True)
    ts: Mapped[float] = mapped_column(Float, index=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(16))  # rtsp | ios | sim
    bbox: Mapped[str] = mapped_column(String(128))   # lưu "[x1,y1,x2,y2]"
    snapshot_url: Mapped[str | None] = mapped_column(String(256), nullable=True)
    person_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
