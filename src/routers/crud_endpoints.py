from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from requests import Session

from ..database.context_manager import get_db
from ..sync.sync_functions import delete_sub
from ..utils.crud import delete_model, get_model, get_models, get_subprocesses

router = APIRouter(prefix='/api/model', tags=['AI Models'])


@router.get('/', summary='Fetches all AI models from the database.')
def get_ai_models(db: Session = Depends(get_db)) -> dict:
    """This endpoint retrieves metadata for all available AI models in the system, including their IDs, base models, file paths, creation dates, training status, and versions.

    **Args**:
    - **db** (Session): The database session.

    **Returns**:
    - **dict**: A dictionary containing the data of all available AI models with the following structure:
        ```json
            {
                "models": [
                    {
                        "id": int,
                        "base_model": str,
                        "file_path": str,
                        "date_created": str,
                        "is_training": bool,
                        "is_trained": bool,
                        "version": int
                    }
                ]
            }
        ```
    **Raises**:
    - **HTTPException (404)**: If no models are found in the database.
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


@router.get('/subs', summary='Fetches all AI model subprocesses stored in the database.')
def get_ai_subs(db: Session = Depends(get_db)) -> dict:
    """This endpoint retrieves information about all subprocesses associated with AI models, including their unique process IDs and names.

    **Args**:
    - **db** (Session): The database session.

    **Returns**:
    - **dict**: A dictionary with the following structure:
        ```json
            {
                "models": [
                    {
                        "id": str,
                        "name": str
                    }
                ]
            }
        ```

    **Raises**:
    - **HTTPException (404)**: If no subprocesses are found in the database.
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


@router.delete('/subs/{sub}', summary='Deletes a specific AI model subprocess by its ID.')
def get_ai_subs(sub:int, db: Session = Depends(get_db)) -> dict:
    """This endpoint removes an AI model subprocess from the system, identified by its unique process ID.

    **Args**:
    - **sub** (int): The unique ID of the subprocess to delete.
    - **db** (Session): The database session.

    **Returns**:
    - **dict**: A message confirming the deletion of the subprocess, with the following structure:
        ```json
            {
                "message": "Subprocess with ID {sub} deleted successfully"
            }
        ```

    **Raises**:
    - **HTTPException (404)**: If the subprocess with the specified ID is not found in the database.
    """

    models = delete_sub(sub, db)
    if models:
        return {'message': f'Deleted syb'}
    else:
        raise HTTPException(status_code=404, detail='Model not found')


@router.get('/{model_id}', summary='Fetches a specific AI model by its unique ID.')
def get_ai_model(model_id: int, db: Session = Depends(get_db)) -> dict:
    """This endpoint retrieves detailed information about an individual AI model, including its metadata such as ID, name, base model, file path, creation date, training status, and version.

    **Args**:
    - **model_id** (int): The unique ID of the AI model to retrieve.
    - **db** (Session): The database session.

    **Returns**:
    - **dict**: A dictionary containing the data of the specified AI model, with the following structure:
        ```json
            {
                "id": int,
                "model_name": str,
                "base_model": str,
                "file_path": str,
                "date_created": str,
                "is_training": bool,
                "is_trained": bool,
                "version": int
            }
        ```

    **Raises**:
    - **HTTPException (404)**: If the model with the specified ID is not found in the database.
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


@router.delete('/{model_id}', summary='Deletes a specific AI model by its unique ID.')
def delete_ai_model(model_id: int, db: Session = Depends(get_db)) -> dict:
    """This endpoint removes an AI model from the system, identified by its unique ID.

    **Args**:
    - **model_id** (int): The unique ID of the AI model to delete.
    - **db** (Session): The database session used to perform the deletion.

    **Returns**:
    - **dict**: A message confirming the deletion of the AI model, with the following structure:
        ```json
            {
                "message": "Model with ID {model_id} deleted successfully"
            }
        ```

    **Raises**:
    - **HTTPException (404)**: If the model with the specified ID is not found in the database.
    """

    model = delete_model(db, model_id)
    if model:
        return {'message': f'Model by id {model_id} deleted successfully'}
    else:
        raise HTTPException(status_code=404, detail='Model not found')
