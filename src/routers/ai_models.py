from datetime import datetime, date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..config import settings
from ..db.db import get_db
from ..utils.crud import create_model, delete_model, get_model, get_models

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
    model = get_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail='Model not found')
    
    # get input text from json
     # Access input_text from the request
    input_text = request.input_text
    
    # pass data to model which id is passed by url param

    # processing
    processed_text = "PROCESSED TEXT"

    return {"input_text": input_text, "processed_text": processed_text, 'message': 'NER Processing finished successfully', }
    

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
