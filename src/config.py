from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    DATABASE_URL : str ="sqlite:///databases/models.db"
    TOKENIZER_PATH : str ="models"
    MODEL_PATH : str ="models"
    DEBUG : bool = True
    # Base models 
    ENG_MODEL : str = "Birband/roberta_ner_eng"
    PL_MODEL : str ="Birband/roberta_ner_pl"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()