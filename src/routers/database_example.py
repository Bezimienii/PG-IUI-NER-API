from fastapi import APIRouter, Depends, HTTPException
from contextlib import contextmanager
from sqlalchemy.orm import Session
from ..db.db import get_db
from ..utils.crud import create_model, update_training_status, get_model, delete_model
from ..utils.enum import BaseModels
from ..config import settings
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(prefix="/ai-models", tags=["AI Models"])

class CreateRequest(BaseModel):
    base_model: str


@router.post("/", summary="Create a new AI model")
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
    return {"id": new_model.id, "message": "AI Model created successfully"}

@router.get("/train", summary="Start training an AI model")
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
        return {"message": "Training started successfully"}
    else:
        raise HTTPException(status_code=404, detail="Model not found")

@router.get("/stop", summary="Stop training an AI model")
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
        return {"message": "Training stopped successfully"}
    else:
        raise HTTPException(status_code=404, detail="Model not found")

@router.get("/{model_id}", summary="Get an AI model by ID")
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
        return {"id": model.id, "base_model": model.base_model, "file_path": model.file_path, "date_created": model.date_created, "is_training": model.is_training, "version": model.version}
    else:
        raise HTTPException(status_code=404, detail="Model not found")

@router.delete("/{model_id}", summary="Delete an AI model by ID")
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
        return {"message": f"Model {model.base_model} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Model not found")
