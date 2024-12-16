from transformers import AutoTokenizer, AutoModelForTokenClassification
from ..utils.enum import BaseModels
from ..config import settings

def load_model_and_tokenizer(model_name: str) -> tuple[AutoTokenizer, AutoModelForTokenClassification]:
    """
    Loads a pre-trained model and tokenizer from the specified model name.

    Args:
        model_name (str): The name of the model to be loaded.

    Returns:
        tuple[AutoTokenizer, AutoModelForTokenClassification]: A tuple containing the tokenizer and model.
    """
    
    model_path = f"{settings.MODEL_PATH}/{model_name}"
    tokenizer_path = f"{settings.TOKENIZER_PATH}/{model_name}"

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        print(f"Model {model_name} loaded successfully")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
