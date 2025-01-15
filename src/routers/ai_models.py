from datetime import datetime, date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..config import settings
from ..db.db import get_db
from ..utils.crud import create_model, delete_model, get_model, get_models, get_model_by_name
from ..utils.models_utils import load_model_and_tokenizer, save_model
from ..model.model import getBaseModel, getTokeniser, classifyText, classifyPolishText, execute_training
from ..utils.enum import BaseModels
from ..utils.model_training_example import fine_tune_model
from pydantic import BaseModel, Field
from typing import List, Dict
from transformers import pipeline
import numpy as np
import json

router = APIRouter(prefix='/ai-models', tags=['AI Models'])


class CreateRequest(BaseModel):
    base_model: str

class CreateRequestNER(BaseModel):
    input_text: str


@router.post('/', summary='Create a new AI model')
def create_ai_model(request: CreateRequest, db: Session = Depends(get_db)) -> dict:
    """Creates a new AI model in the database.

    Args:
        base_model (BaseModels): The base model of the AI model.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the ID and message of the created AI model.
    """
    # string to enum
    # try:
    #     base_model = BaseModels(request.base_model)
    # except ValueError:
    #     raise HTTPException(status_code=400, detail="Invalid base model")

    new_model = create_model(
        session=db,
        base_model=request.base_model,
        file_path=settings.MODEL_PATH,
        date_created=datetime.now(),
    )
    return {'id': new_model.id, 'message': 'AI Model created successfully'}

# ---------------------------------------------- EXAMPLE TRAINING -----------------------------------------------

class FineTuneRequestNER(BaseModel):
    name: str = Field(..., description="Name of the model")  # Required: Name of the model
    fine_tune_data: Dict[str, List[List]] = Field(
        ..., 
        description="Fine-tuning data including texts and labels"
    )

@router.post('/example_train/{model_id}', summary='Learning example')
def fine_tune_ai_model(model_id: int, request: FineTuneRequestNER, db: Session = Depends(get_db)) -> dict:
    """
    Fine-tunes the specified AI model with new NER data.

    Args:
        model_id (int): The ID of the AI model to fine-tune.
        request (FineTuneRequestNER): New fine-tuning data.
        db (Session): The database session.

    Returns:
        dict: Success message confirming fine-tuning.
    """

    # Get the model from the database (if it exists)
    model_record = get_model(db, model_id)
    if not model_record:
        raise HTTPException(status_code=404, detail='Model not found')
    
    # Get model name from the database record
    model_name = model_record.base_model
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    if model is None or tokenizer is None:
        raise HTTPException(status_code=404, detail=f'Model not found: {model_name}')

    # Fine-tune the model with new data
    try:
        new_model = fine_tune_model(request.fine_tune_data["texts"], request.fine_tune_data["labels"], model, tokenizer)
        # Save the fine-tuned model and tokenizer
        model_new_name = request.name

        save_model(new_model, tokenizer, model_new_name)
        create_model(
            session=db,
            base_model=model_new_name,
            file_path=settings.MODEL_PATH,
            date_created=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fine-tuning failed: {e}")

    try:
        new_model_id = get_model_by_name(db, model_new_name).id
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fine-tuning failed: {e}")

    return {
        "message": f"Model {model_name} fine-tuned successfully.",
        "model_id": new_model_id
    }

# ---------------------------------------------- END EXAMPLE TRAINING -----------------------------------------------

@router.get('/{model_id}', summary='Get an AI model by ID')
def get_ai_model(model_id: int, db: Session = Depends(get_db)) -> dict:
    """Gets an AI model by its ID.

    Args:
        model_id (int): The ID of the AI model to get.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the AI model data.
    """
    model = get_model(db, model_id)
    if model:
        return {
            'id': model.id,
            'base_model': model.base_model,
            'file_path': model.file_path,
            'date_created': model.date_created,
            'is_training': model.is_training,
            'is_trained': model.is_trained,
            'version': model.version,
        }
    else:
        raise HTTPException(status_code=404, detail='Model not found')
    

def float32_to_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    return obj

@router.post('/{model_id}', summary='Pass input for a model to do NER')
def get_ai_model(model_id: int, request: CreateRequestNER, db: Session = Depends(get_db)) -> dict:
    """Pass input for a model to do NER.

    Args:
        model_id (int): The ID of the AI model to get.
        request (CreateRequestNER): NER request with input_text to process
        db (Session): The database session.

    Returns:
        dict: answer for NER process
    """

    # model = get_model(db, model_id)
    # if not model:
    #     raise HTTPException(status_code=404, detail='Model not found')
    
    # # get input text from json
    input_text = request.input_text

    model = get_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail='Model not found')

    model_name = model.base_model

    model, tokenizer = load_model_and_tokenizer(model_name)

    if model == None or tokenizer == None:
        raise HTTPException(status_code=404, detail=f'Model not found: {model_name}')

    # if model_id == 1:
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    processed_text = nlp(input_text)
    # else:
    #     processed_text = classifyText(model, tokenizer, input_text)
    try:
        processed_text_json = [ {key: float(value) if isinstance(value, np.float32) else value for key, value in item.items()} for item in processed_text ]
        return {'words': processed_text_json}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON input"}
    # return {"input_text": input_text,'message': 'NER Processing finished successfully', }
    

@router.get('/', summary='Get all AI models')
def get_ai_model(db: Session = Depends(get_db)) -> dict:
    """Gets all AI models.

    Args:
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the all available AI models data.
    """
    models = get_models(db)
    if models:
        response = [
            {
                'id': model.id,
                'base_model': model.base_model,
                'file_path': model.file_path,
                'date_created': model.date_created.isoformat() if isinstance(model.date_created, date) else model.date_created,
                'is_training': model.is_training,
                'is_trained': model.is_trained,
                'version': model.version,
            }
            for model in models
        ]
        return {"models": response}
    else:
        raise HTTPException(status_code=404, detail='Model not found')

@router.delete('/{model_id}', summary='Delete an AI model by ID')
def delete_ai_model(model_id: int, db: Session = Depends(get_db)) -> dict:
    """Deletes an AI model by its ID.

    Args:
        model_id (int): The ID of the AI model to delete.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the message of the deletion.
    """
    model = delete_model(db, model_id)
    if model:
        return {'message': f'Model {model.base_model} deleted successfully'}
    else:
        raise HTTPException(status_code=404, detail='Model not found')

@router.put('/{model_id}', summary='Zaczynamy show')
def execute_model(model_id: int):
    execute_training(model_id)