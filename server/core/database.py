import os
import uuid
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./workspace/app.db"

# Create a robust SQLite engine
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class OAuthToken(Base):
    __tablename__ = "oauth_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String, unique=True, index=True) # e.g., 'youtube', 'tiktok', 'instagram'
    access_token = Column(Text, nullable=True)
    refresh_token = Column(Text, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Scenario(Base):
    __tablename__ = "scenarios"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    premise = Column(Text, nullable=False)
    title = Column(String, nullable=False)
    status = Column(String, nullable=False, default="created", index=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    current_stage = Column(String, nullable=True)
    job_id = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    research_data = Column(Text, nullable=True)
    analysis_data = Column(Text, nullable=True)
    timeline_data = Column(Text, nullable=True)
    script_data = Column(Text, nullable=True)
    visual_plan_data = Column(Text, nullable=True)
    artifact_dir = Column(String, nullable=False)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
