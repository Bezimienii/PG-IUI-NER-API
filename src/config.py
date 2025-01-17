from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class Settings(BaseSettings):
    """Application settings. This class is used to store the application settings and/or load them from a .env file."""
    DATABASE_URL : str ="sqlite:///databases/models.db"
    TOKENIZER_PATH : str ="models"
    MODEL_PATH : str ="models"
    DEBUG : bool = True
    ENG_MODEL : str = "Birband/roberta_ner_eng"
    PL_MODEL : str ="Birband/roberta_ner_pl"
    UPLOAD_DIR : str = 'tmp/files'

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
