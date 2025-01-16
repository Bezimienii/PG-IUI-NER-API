import os
import uuid
from datetime import datetime
from multiprocessing import Process

from fastapi import APIRouter, Depends, Form, UploadFile

from src.db.db import Session, get_db
from ..model.training import execute_training
from ..utils.crud import create_model, get_model
from ..config import settings
router = APIRouter(prefix='/processing', tags=['processing'])

# UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../files'))
UPLOAD_DIR = 'tmp\\files'

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

@router.post('', summary='Train a model')
def train_model(
    model_name: str = Form(...),
    base_model: int = Form(...),
    train_data: UploadFile = Form(...),
    valid_data: UploadFile = Form(...),
    test_data: UploadFile = Form(...),
    db: Session = Depends(get_db),
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

    print("XD")

    with Session() as db:
        model_info = get_model(db, base_model)
    print(model_info)

    files_uuid = uuid.uuid4()
    train_path = save_file(train_data, f'train_{files_uuid}')
    valid_path = save_file(valid_data, f'valid_{files_uuid}')
    test_path = save_file(test_data, f'test_{files_uuid}')

    model = create_model(
        session=db,
        base_model=model_info.model_name,
        file_path = f'{settings.MODEL_PATH}/{model_name}',
        model_name=model_name,
        train_file_path=train_path,
        valid_file_path=valid_path,
        test_file_path=test_path,
        training_process_id=0,
        is_training=True,
        is_trained=False,
        date_created=datetime.now(),
    )

    model_id = model.id

    p = Process(target=execute_training, args=(model_id,)) # process independent of parent
    p.start()

    return {'message': 'Successfully started training model.', 'model_name': model_name, 'training_process_id': p.pid}
