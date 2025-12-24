# models.py
from sqlalchemy import Column, Integer, String, Date, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import ARRAY, REAL

Base = declarative_base()

# -------------------------------
# PERSON
# -------------------------------
class Person(Base):
    __tablename__ = "person"

    roll_number = Column(String(20), primary_key=True)
    name = Column(String(100))


# -------------------------------
# FACE EMBEDDING (CSV STYLE)
# -------------------------------
class FaceEmbedding(Base):
    __tablename__ = "face_embedding"

    embedding_id = Column(Integer, primary_key=True)
    roll_number = Column(
        String(20),
        ForeignKey("person.roll_number", ondelete="CASCADE"),
        nullable=False
    )

    # CSV-equivalent storage
    embedding = Column(ARRAY(REAL), nullable=False)

    created_at = Column(DateTime, server_default=func.now())


# -------------------------------
# ATTENDANCE
# -------------------------------
class Attendance(Base):
    __tablename__ = "attendance"

    attendance_id = Column(Integer, primary_key=True)
    roll_number = Column(
        String(20),
        ForeignKey("person.roll_number"),
        nullable=False
    )

    attendance_date = Column(Date, server_default=func.current_date())
    marked_at = Column(DateTime, server_default=func.now())
