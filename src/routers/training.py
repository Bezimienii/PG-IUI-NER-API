from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..config import settings
from ..db.db import get_db
from ..utils.crud import create_model, delete_model, get_model, update_training_status

router = APIRouter(prefix='/ai-training', tags=['AI Training'])


@router.get('/status/{model_id}', summary='Get training status of AI model training process')
def start_training(model_id: int, db: Session = Depends(get_db)) -> dict:
    """Get training status of AI model training process

    Args:
        model_id (int): The ID of the AI model to start training.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the message of the training status update.
    """

    # the idea is to set is_trained to false when training process started, and after this process is finished
    # set the is_trained to true
    # this endpoint only gives information about training process

    # is_trainded = None -> training process not started
    # is_trainded = False -> training process started, but not finished yet
    # is_trainded = True -> training process finished, model ready to use

    model = get_model(db, model_id)
    if model:
        return {"is_trained": model.is_trained}
    else:
        raise HTTPException(status_code=404, detail='Model not found')