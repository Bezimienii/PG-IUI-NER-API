from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from .enum import BaseModels
from ..config import settings
from ..db.models import AIModel

def load_model_and_tokenizer(model: AIModel, train: bool = False) -> tuple[RobertaTokenizerFast, RobertaForTokenClassification]:
    """
    Loads a pre-trained model and tokenizer from the specified model name.

    Args:
        model_name (str): The name of the model to be loaded.

    Returns:
        tuple[AutoTokenizer, AutoModelForTokenClassification]: A tuple containing the tokenizer and model.
    """
    
    model_path = f"{settings.MODEL_PATH}/{model.base_model if train else model.model_name}"

    try:

        tokenizer = RobertaTokenizerFast.from_pretrained(model_path, ignore_mismatched_sizes=True, add_prefix_space=True)
        model = RobertaForTokenClassification.from_pretrained(model_path, ignore_mismatched_sizes=True)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
def save_model(model, tokenizer, model_name):
    """
    Saves a pre-trained model and tokenizer to the specified model name.

    Args:
        model (AutoModelForTokenClassification): The model to be saved.
        tokenizer (AutoTokenizer): The tokenizer to be saved.
        model_name (str): The name of the model to be saved.

    Returns:
        None
    """
    model_path = f"{settings.MODEL_PATH}/{model_name}"

    model.save_pretrained(model_path)
    model.config.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    print(f"Model {model_name} saved successfully")