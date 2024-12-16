from ..db.db import engine, Base
from ..db.models import AIModel

def initialize_db():
    """Initializes the database."""

    print("Initializing database...")

    Base.metadata.create_all(bind=engine)

    print("Database initialized.")

if __name__ == "__main__":
    initialize_db()