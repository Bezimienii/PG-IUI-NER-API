from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from ..config import settings

engine = create_engine(settings.DATABASE_URL)

Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Get a database session."""
    db = Session()
    try:
        yield db
    finally:
        db.close()

db_context = contextmanager(get_db)
