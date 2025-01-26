import json
import os
import uuid
from datetime import datetime
from multiprocessing import Process

import numpy as np
from fastapi import APIRouter, Form, HTTPException, UploadFile, Depends
from pydantic import BaseModel
from transformers import pipeline

from ..config import settings
from ..database.context_manager import get_db, Session
from ..database.models import AIModel
from ..model.training import execute_training
from ..utils.crud import create_model, get_model
from ..utils.models_utils import load_model_and_tokenizer

router = APIRouter(prefix='/api/model', tags=['AI Models'])


# ----------------- Training -----------------

UPLOAD_DIR = settings.UPLOAD_DIR

def save_file(file: UploadFile, name: str) -> str:
    """Save an uploaded file to the UPLOAD_DIR with a new name.

    Args:
        file (UploadFile): The file to be saved.
        name (str): The new name for the file.

    Returns:
       file_path (str): The path of the saved file.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_extension = os.path.splitext(file.filename)[-1]
    new_filename = f'{name}{file_extension}'
    file_path = os.path.join(UPLOAD_DIR, new_filename)

    print(file_path)
    with open(file_path, 'wb') as buffer:
        buffer.write(file.file.read())

    return file_path

@router.post('/train', summary='Train a model')
def train_model(
    model_name: str = Form(...),
    base_model: int = Form(...),
    train_data: UploadFile = Form(...),
    valid_data: UploadFile = Form(...),
    test_data: UploadFile = Form(...)
):
    """Initiates the training process for a Named-Entity Recognition (NER) model.

    **Args**:
    - **model_name** (str): The name of the model to be trained.
    - **model_language** (str): The language for the model (e.g., "en", "pl").
    - **train_data** (UploadFile): The training data for the model.
    - **valid_data** (UploadFile): The validation data for the model.
    - **test_data** (UploadFile): The test data for the model.

    **Returns**:
    - **dict**: A dictionary confirming the successful initiation of the training process with the following structure:
        ```json
            {
                "message": "Successfully loaded model.",
                "model_name": "<model_name>"
            }
        ```

    **Raises**:
    - **HTTPException (400)**: If a model with the provided name already exists.
    """
    with Session() as db:
        existing_model = db.query(AIModel).filter_by(model_name=model_name).first()
        if existing_model:
            return {
                "message": "Model with the given name already exists.",
                "model_name": model_name,
                "is_trained": existing_model.is_trained
            }
        model_info = get_model(db, base_model)

    files_uuid = uuid.uuid4()
    train_path = save_file(train_data, f'train_{files_uuid}')
    valid_path = save_file(valid_data, f'valid_{files_uuid}')
    test_path = save_file(test_data, f'test_{files_uuid}')

    create_model(
        session=db,
        base_model=model_info.model_name,
        file_path = f'{settings.MODEL_PATH}/{model_name}',
        model_name=model_name,
        train_file_path=train_path,
        valid_file_path=valid_path,
        test_file_path=test_path,
        training_process_id=0,
        is_training=False,
        is_trained=False,
        date_created=datetime.now(),
    )

    return {'message': 'Successfully loaded model.', 'model_name': model_name}


# ----------------- NER -----------------

def float32_to_float(obj):
    """Converts np.float32 objects to float."""
    if isinstance(obj, np.float32):
        return float(obj)
    return obj

class CreateRequestNER(BaseModel):
    """Request model for NER."""
    input_text: str

@router.post('/{model_id}/ner', summary='Perform Named Entity Recognition (NER)')
def get_ai_model(model_id: int, request: CreateRequestNER, db: Session = Depends(get_db)) -> dict:
    """Processes input text for Named Entity Recognition using a specified model.

    **Args**:
    - **model_id** (int): The ID of the AI model to use for NER.
    - **request** (CreateRequestNER): Input text for the NER process.
    - **db** (Session): The database session.

    **Returns**:
    - **dict**: A dictionary containing the NER results with the following structure:
        ```json
            {
                "sentence": "<input_text>",
                "words": [
                    {
                        "entity_group": str,
                        "score": float,
                        "word": str,
                        "start": int,
                        "end": int
                    }
                ],
                "status": "success"
            }
        ```

    **Raises**:
    - **HTTPException (404)**: If the model is not found or not trained.
    - **HTTPException (400)**: If there is an error processing the input text.
    """

    input_text = request.input_text

    model = get_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail='Model not found')

    if not model.is_trained:
        raise HTTPException(status_code=404, detail='Model not trained')

    model_name = model.base_model

    model, tokenizer = load_model_and_tokenizer(model)

    if model is None or tokenizer is None:
        raise HTTPException(status_code=404, detail=f'Model not found: {model_name}')

    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    processed_text = nlp(input_text)

    try:
        processed_text_json = [ {
            key: float(value) if isinstance(value, np.float32) else value
                for key, value in item.items()
                }
            for item in processed_text
            ]
        return {'sentence': input_text, 'words': processed_text_json, 'status': 'success'}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON input"}

# ----------------- STATE -----------------

@router.get('/{model_id}/state', summary='Get model training state')
def get_ai_model_state(model_id: int, db: Session = Depends(get_db)) -> dict:
    """Checks the training state of a specific AI model.

    **Args**:
    - **model_id** (int): The ID of the AI model to check.
    - **db** (Session): The database session.

    **Returns**:
    - **dict**: A dictionary containing the training state of the model with the following structure:
      ```json
      {
          "is_training": bool,
          "is_trained": bool
      }
      ```

    **Raises**:
    - **HTTPException (404)**: If the model with the specified ID is not found in the database.
    """

    model = get_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail='Model not found')

    return {'is_training': model.is_training, 'is_trained': model.is_trained}
