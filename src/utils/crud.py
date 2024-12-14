from datetime import datetime
from typing import List

from sqlalchemy.orm import Session

from ..db.models import AIModel


def update_training_status(session: Session, model_id: int, is_training: bool) -> bool:
    """Updates the training status of a model in the database.

    Args:
        session (Session): The database session.
        model_id (int): The ID of the model to update.
        is_training (bool): The new training status.

    Returns:
        None
    """
    model = session.query(AIModel).filter_by(id=model_id).first()
    if model:
        model.is_training = is_training
        session.commit()
        return True
    else:
        print(f'Model with ID {model_id} not found.')
        return False


def get_model(session: Session, model_id: int) -> AIModel | None:
    """Retrieves a model from the database by its ID.

    Args:
        session (Session): The database session.
        model_id (int): The ID of the model to retrieve.

    Returns:
        AIModel: The model with the specified ID.
    """
    model = session.query(AIModel).filter_by(id=model_id).first()
    if model:
        return model
    else:
        print(f'Model with ID {model_id} not found.')
        return None


def get_models(session: Session) -> List[AIModel] | None:
    """Retrieves a model from the database by its ID.

    Args:
        session (Session): The database session.
        model_id (int): The ID of the model to retrieve.

    Returns:
        AIModel: The model with the specified ID.
    """
    models = session.query(AIModel).all()
    if models:
        return models
    else:
        print(f'No available AI models')
        return None


def create_model(session: Session, base_model: str, file_path: str, date_created: datetime) -> AIModel:
    """Creates a new model in the database.

    Args:
        session (Session): The database session.
        base_model (str): The base model of the model.
        file_path (str): The file path of the model.
        date_created (str): The date the model was created.

    Returns:
        AIModel: The created model.
    """
    # print type of session
    print(session)

    model = AIModel(base_model=base_model, file_path=file_path, date_created=date_created)
    session.add(model)
    session.commit()
    return model


def delete_model(session: Session, model_id: int) -> bool:
    """Deletes a model from the database.

    Args:
        session (Session): The database session.
        model_id (int): The ID of the model to delete.

    Returns:
        None
    """
    model = session.query(AIModel).filter_by(id=model_id).first()
    if model:
        session.delete(model)
        session.commit()
        return True
    else:
        print(f'Model with ID {model_id} not found.')
        return False
