from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    DATABASE_URL : str ="sqlite:///./src/databases/models.db"
    TOKENIZER_PATH : str ="./src/tokenizers"
    MODEL_PATH : str ="./src/models"
    DEBUG : bool = True
    # Base models 
    ENG_MODEL : str = "Birband/roberta_ner_eng"
    PL_MODEL : str ="Birband/roberta_ner_pl"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()