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
    test_data: UploadFile = Form(...),
    db: Session = Depends(get_db)
):
    """Initiates the training process for a Named-Entity Recognition (NER) model.

    **Args**:
    - **model_name** (str): The name of the model to be trained.
    - **model_language** (str): The language for the model (e.g., "en", "pl").
    - **train_data** (UploadFile): The training data for the model.
    - **valid_data** (UploadFile): The validation data for the model.
    - **test_data** (UploadFile): The test data for the model.
    - **db** (Session): The database session.

    **Returns**:
    - **message** (str): A message indicating the status of the training process.
    - **training_id** (UUID): A unique ID for the training process.
    - **model_name** (str): The name of the model being trained.
    """
    with Session() as db:
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

@router.post('/{model_id}/ner', summary='Pass input for a model to do NER')
def get_ai_model(model_id: int, request: CreateRequestNER, db: Session = Depends(get_db)) -> dict:
    """Pass input for a model to do NER.

    Args:
        model_id (int): The ID of the AI model to get.
        request (CreateRequestNER): NER request with input_text to process
        db (Session): The database session.

    Returns:
        dict: answer for NER process
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

@router.get('/{model_id}/state', summary='Check if model is training')
def get_ai_model_state(model_id: int, db: Session = Depends(get_db)) -> dict:
    """Check if model is training.

    Args:
        model_id (int): The ID of the AI model to get.

    Returns:
        dict: answer for NER process
    """

    model = get_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail='Model not found')

    return {'is_training': model.is_training, 'is_trained': model.is_trained}
