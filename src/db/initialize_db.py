import os
from .db import engine, Base, db_context
from ..db.models import AIModel
from ..config import settings
from ..utils.enum import BaseModels
from ..utils.crud import create_model, get_model_by_name
import datetime

def initialize_db():
    """Initializes the database."""
    database_url = settings.DATABASE_URL
    # Ensure the directory exists
    db_dir = os.path.dirname(database_url[10:])  # Remove 'sqlite:///' prefix
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    print(f"Database URL: {database_url}")
    Base.metadata.create_all(bind=engine)
    print("Database initialized.")

def create_models():
    """Creates the default models in the database."""
    print("Creating models...")
    with db_context() as session:
        for model_name in BaseModels:
            model = get_model_by_name(session, model_name.value)
            if model:
                print(f"Model {model_name.value} already exists.")
                continue
            model = create_model(
                session=session,
                base_model=model_name.value,
                file_path=settings.MODEL_PATH,
                date_created=datetime.datetime.now(),
            )
            print(f"Model {model.base_model} created.")
    print("Models created.")