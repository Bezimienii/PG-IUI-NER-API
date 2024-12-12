from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL : str ="sqlite:///./databases/models.db"
    MODEL_PATH : str ="./models"
    DEBUG : bool = False
    # Base models 
    ROBERTA : str = "roberta-base"
    HERBERT : str ="herbert-base-cased"
    XLM_ROBERTA : str ="xlm-roberta"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
