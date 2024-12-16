import os 
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from ..utils.enum import BaseModels
from ..config import settings

def download_model(model_name: str) -> None:
    """
    Downloads a pre-trained model from HuggingFace.

    Args:
        model_name (str): The name of the model to be downloaded.

    Returns:
        bool: True if the model was downloaded successfully, False otherwise.
    """
    if os.path.exists(f"{settings.MODEL_PATH}/{model_name}"):
        print(f"Model {model_name} already exists in {settings.MODEL_PATH}/{model_name}")
        return
    
    try:
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        model.save_pretrained(f"{settings.MODEL_PATH}/{model_name}")
        print(f"Model {model_name} downloaded successfully")
        return
    except Exception as e:
        print(f"Error downloading model: {e}")
        return

def download_tokenizer(model_name: str) -> None:
    """
    Downloads a pre-trained tokenizer from HuggingFace.

    Args:
        model_name (str): The name of the tokenizer to be downloaded.

    Returns:
        bool: True if the tokenizer was downloaded successfully, False otherwise.
    """
    if os.path.exists(f"{settings.TOKENIZER_PATH}/{model_name}"):
        print(f"Tokenizer {model_name} already exists in {settings.TOKENIZER_PATH}/{model_name}")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(f"{settings.TOKENIZER_PATH}/{model_name}")
        print(f"Tokenizer {model_name} downloaded successfully")
        return 
    except Exception as e:
        print(f"Error downloading tokenizer: {e}")
        return

def download_default_models() -> None:
    """
    Loads the default models from the settings file. And downloads them if they are not already downloaded.
    """
    print("Debug mode is set to:", settings.DEBUG)
    if settings.DEBUG:
        print(f"Tokenizer path: {settings.TOKENIZER_PATH}")
        print(f"Model path: {settings.MODEL_PATH}")

    for model_name in BaseModels:
        download_model(model_name.value)
        download_tokenizer(model_name.value)
