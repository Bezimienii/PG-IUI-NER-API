from datetime import datetime

from sqlalchemy.orm import Session

from ..database.models import AIModel


def update_training_status(session: Session, model_id: int, is_training: bool, is_trained) -> AIModel | None:
    """Updates the training status of a model in the database.

    Args:
        session (Session): The database session.
        model_id (int): The ID of the model to update.
        is_training (bool): The new training status of the model.
        is_trained (bool): The new trained status of the model.

    Returns:
        None
    """
    model = get_model(session, model_id)
    if model:
        model.is_training = is_training
        model.is_trained = is_trained
        session.commit()
        return model
    else:
        print(f'Model with ID {model_id} not found.')
        return None

def update_training_process_id(session: Session, model_id: int, training_process_id: int) -> AIModel | None:
    """Updates the training process id of a model in the database.

    Args:
        session (Session): The database session.
        model_id (int): The ID of the model to update.
        training_process_id (int): The new training process id.

    Returns:
        None
    """
    model = get_model(session, model_id)
    if model:
        model.training_process_id = training_process_id
        session.commit()
        return model
    else:
        print(f'Model with ID {model_id} not found.')
        return None


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

def get_model_by_name(session: Session, model_name: str) -> AIModel | None:
    """Retrieves a model from the database by its name.

    Args:
        session (Session): The database session.
        model_name (int): The name of the model to retrieve.

    Returns:
        AIModel: The model with the specified ID.
    """
    model = session.query(AIModel).filter_by(base_model=model_name).first()
    if model:
        return model
    else:
        print(f'Model with name {model_name} not found.')
        return None

def get_model_by_model_name(session: Session, model_name: str) -> AIModel | None:
    """Retrieves a model from the database by its name.

    Args:
        session (Session): The database session.
        model_name (int): The name of the model to retrieve.

    Returns:
        AIModel: The model with the specified ID.
    """
    model = session.query(AIModel).filter_by(name=model_name).first()
    if model:
        return model
    else:
        print(f'Model with name {model_name} not found.')
        return None


def get_models(session: Session) -> list[AIModel] | None:
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
        print('No available AI models')
        return None


def create_model(session: Session,
                 base_model: str,
                 file_path: str,
                 date_created: datetime,
                 is_training: bool = False,
                 is_trained: bool = False,
                 model_name: str = "",
                 train_file_path: str = "",
                 valid_file_path: str = "",
                 test_file_path: str = "",
                 training_process_id: int = 0) -> AIModel:
    """Creates a new model in the database.

    Args:
        session (Session): The database session.
        base_model (str): The base model of the new model.
        file_path (str): The file path of the new model.
        date_created (datetime): The creation date of the new model.
        is_training (bool): The training status of the new model.
        is_trained (bool): The trained status of the new model.
        model_name (str): The name of the new model.
        train_file_path (str): The training file path of the new model.
        valid_file_path (str): The validation file path of the new model.
        test_file_path (str): The test file path of the new model.
        training_process_id (int): The training process ID of the new model.

    Returns:
        AIModel: The created model.
    """
    model = AIModel(
        base_model=base_model,
        file_path=file_path,
        date_created=date_created,
        is_training=is_training,
        is_trained=is_trained,
        model_name=model_name,
        train_file_path=train_file_path,
        valid_file_path=valid_file_path,
        test_file_path=test_file_path,
        training_process_id=training_process_id
    )
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
