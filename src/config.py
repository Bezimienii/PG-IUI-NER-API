from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    DATABASE_URL : str ="sqlite:///./databases/models.db"
    TOKENIZER_PATH : str ="./src/tokenizers"
    MODEL_PATH : str ="./src/models"
    DEBUG : bool = False
    # Base models 
    ROBERTA : str = "Jean-Baptiste/roberta-large-ner-english"
    HERBERT : str ="pietruszkowiec/herbert-base-ner"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()