from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..config import settings
from ..db.db import get_db
from ..utils.crud import create_model, delete_model, get_model, update_training_status

router = APIRouter(prefix='/ai-models', tags=['AI Models'])



@router.get('/train', summary='Start training an AI model')
def start_training(model_id: int, db: Session = Depends(get_db)) -> dict:
    """Starts training an AI model.

    Args:
        model_id (int): The ID of the AI model to start training.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the message of the training status update.
    """
    training_status = update_training_status(db, model_id, True)
    if training_status:
        return {'message': 'Training started successfully'}
    else:
        raise HTTPException(status_code=404, detail='Model not found')


@router.get('/stop', summary='Stop training an AI model')
def stop_training(model_id: int, db: Session = Depends(get_db)) -> dict:
    """Stops training an AI model.

    Args:
        model_id (int): The ID of the AI model to stop training.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the message of the training status update.
    """
    training_status = update_training_status(db, model_id, False)
    if training_status:
        return {'message': 'Training stopped successfully'}
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
