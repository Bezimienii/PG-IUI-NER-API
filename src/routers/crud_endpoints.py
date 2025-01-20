from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from requests import Session

from ..database.context_manager import get_db
from ..utils.crud import delete_model, get_model, get_models, get_subprocesses

router = APIRouter(prefix='/api/model', tags=['AI Models'])


@router.get('/', summary='Get all AI models')
def get_ai_models(db: Session = Depends(get_db)) -> dict:
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
            'date_created': model.date_created.isoformat()
                    if isinstance(model.date_created, date)
                    else model.date_created,
            'is_training': model.is_training,
            'is_trained': model.is_trained,
            'version': model.version,
            }
            for model in models
        ]
        return {"models": response}
    else:
        raise HTTPException(status_code=404, detail='Model not found')

@router.get('/subs', summary='Get all AI models')
def get_ai_subs(db: Session = Depends(get_db)) -> dict:
    """Gets all AI models.

    Args:
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the all available AI models data.
    """

    models = get_subprocesses(db)
    if models:
        response = [
            {
            'id': model.pid,
            'name': model.name
            }
            for model in models
        ]
        return {"models": response}
    else:
        raise HTTPException(status_code=404, detail='Model not found')




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
            'model_name': model.model_name,
            'base_model': model.base_model,
            'file_path': model.file_path,
            'date_created': model.date_created,
            'is_training': model.is_training,
            'is_trained': model.is_trained,
            'version': model.version,
        }
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
