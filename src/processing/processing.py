import os
import uuid
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.db.db import Session, get_db
from src.model.model import getBaseModel, getTokeniser, convert_2_dataframe, split_data, dataset_2_list, form_input, \
    train_engine
from src.utils.crud import get_model_by_model_name
from torch.utils.data import DataLoader

router = APIRouter(prefix='/api/processing', tags=['processing'])


class TrainingRequest(BaseModel):
    model_name: str
    model_language: str
    data: list


@router.post('/', summary='Train a model')
def train_model(
    model_name: str = Form(...),
    model_language: str = Form(...),
    train_data: UploadFile = Form(...),
    valid_data: UploadFile = Form(...),
    test_data: UploadFile = Form(...),
    db: Session = Depends(get_db),
):
    """Initiates the training process for a Named-Entity Recognition (NER) model.

    Args:
        model_name (str): The name of the model to be trained.
        model_language (str): The language for the model (e.g., "en", "pl").
        train_data (UploadFile): The training data for the model.
        valid_data (UploadFile): The validation data for the model.
        test_data (UploadFile): The test data for the model.
        db (Session): The database session.

    Returns:
        message (str):
        training_id (UUID):
        model_name (str):
    """

    files_uuid = uuid.uuid4()
    train_path = save_file(train_data, f'train_{files_uuid}')
    valid_path = save_file(valid_data, f'valid_{files_uuid}')
    test_path = save_file(test_data, f'test_{files_uuid}')

    training_id = uuid.uuid4()
    train_process = Process(target=train_model_simulation, args=("Example data",))
    train_process.start()

    # Wait until train process ended
    train_process.join()

    # with open('./train.conllu', "r", encoding="utf-8") as f:
    #     data = f.read()
    #
    # if get_model_by_model_name(db, model_name):
    #     raise HTTPException(status_code=400, detail='Model name already exists. Use a different name.')
    #
    # supported_languages = ['en', 'pl']
    # if model_language not in supported_languages:
    #     raise HTTPException(status_code=400, detail='Unsupported language.')
    #
    # model = getBaseModel(training_request.model_language)
    # tokenizer = getTokeniser(training_request.model_language)
    #
    # train_Id_list = [item['id'] for item in data]
    # train_sentences = [item['sentence'] for item in data]
    # train_keywords = [item['keyword'] for item in data]
    # train_labels = [item['label'] for item in data]
    #
    # # Example data
    # # {
    # #     "id": 1,
    # #     "sentence": "This is a sample sentence.",
    # #     "keyword": "sample",
    # #     "label": "O"
    # # }
    #
    # unique_df = convert_2_dataframe(
    #     train_Id_list=train_Id_list,
    #     train_sentences=train_sentences,
    #     train_keywords=train_keywords,
    #     train_labels=train_labels
    # )
    #
    # train_df, valid_df = split_data(unique_df)
    #
    # train_id_list, train_sentences, train_keywords, train_labels = dataset_2_list(train_df)
    # valid_id_list, valid_sentences, valid_keywords, valid_labels = dataset_2_list(valid_df)
    #
    # train_dataset = form_input(
    #     ID=train_id_list,
    #     sentence=train_sentences,
    #     kword=train_keywords,
    #     label=train_labels,
    #     tokenizer=tokenizer,
    #     data_type='train'
    # )
    #
    # valid_dataset = form_input(
    #     ID=valid_id_list,
    #     sentence=valid_sentences,
    #     kword=valid_keywords,
    #     label=valid_labels,
    #     tokenizer=tokenizer,
    #     data_type='valid'
    # )
    #
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config['batch_size'], # BATCH_SIZE
    #     shuffle=True)
    #
    # valid_loader = DataLoader(
    #     valid_dataset,
    #     batch_size=config['batch_size'], # BATCH_SIZE
    #     shuffle=True)
    #
    # model, val_predictions, val_true_labels = train_engine(
    #     model=model,
    #     epoch=config['Epoch'], # EPOCH
    #     train_data=train_loader,
    #     valid_data=valid_loader,
    #     model_name=model_name
    # )

    # TODO: save model

    model = create_model(
        session=db,
        base_model=model_language,
        file_path=model_name,
        is_training=True,
        is_trained=False,
        date_created=datetime.now(),
    )

    # id of created model for future usage
    model_id = model.id

    return {
        'message': 'Successfully started training model.',
        'model_name': model_name,
        'training_id': training_id
    }
