# app/models.py
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, Text, Boolean

Base = declarative_base()


class User(Base):
    __tablename__ = "users"   # tên bảng, có thể đổi cho khớp DB của bạn

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    # MẬT KHẨU PLAIN TEXT (phục vụ dev), KHÔNG HASH
    password = Column(String(128), nullable=False)
    is_admin = Column(Boolean, default=False)

    def check_password(self, password: str) -> bool:
        # So sánh trực tiếp
        return self.password == password


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True)

    # ID logic của event, dùng cho UI / API
    evt_id = Column(String(64), index=True, nullable=False, unique=True)

    # timestamp (epoch seconds)
    ts = Column(Float, index=True, nullable=False)

    camera_id = Column(String(64), index=True, nullable=False)
    source = Column(String(32), default="esp32")

    # bounding box người, JSON string "[x1,y1,x2,y2]"
    bbox = Column(Text)

    # đường dẫn ảnh tương đối, ví dụ "/snapshots/front_gate_xxx.jpg"
    snapshot_url = Column(Text)

    # dành cho face recognition sau này
    person_id = Column(String(64), nullable=True)

    # ===== PHẦN MỚI: session / phân loại thời gian đứng =====

    # thời gian đứng (giây) trong phiên đó
    duration_sec = Column(Float, nullable=True)

    # loại cảnh báo: "none" | "yellow" | "red"
    #  - "none"  : loại 1 (10–15s, không gửi alert lên UI)
    #  - "yellow": loại 2 (15–60s, alert vàng)
    #  - "red"   : loại 3 (>60s, alert đỏ)
    alert_type = Column(String(16), nullable=True)
