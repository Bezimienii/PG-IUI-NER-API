import os
import uuid

from fastapi import APIRouter, Depends, Form, UploadFile
from pydantic import BaseModel

from src.db.db import Session, get_db
from src.model.model import label2id

router = APIRouter(prefix='/api/processing', tags=['processing'])

# UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../files'))
UPLOAD_DIR = '/tmp/files'
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_file(file: UploadFile, name: str) -> str:
    """
    Save an uploaded file to the UPLOAD_DIR with a new name.

    Args:
        file (UploadFile): The file to be saved.
        name (str): The new name for the file.

    Returns:
       file_path (str): The path of the saved file.
    """
    file_extension = os.path.splitext(file.filename)[-1]
    new_filename = f'{name}{file_extension}'
    file_path = os.path.join(UPLOAD_DIR, new_filename)

    with open(file_path, 'wb') as buffer:
        buffer.write(file.file.read())

    return file_path


class Sentence(BaseModel):
    id: int
    ner_tags: list[int]
    tokens: list[str]

    @classmethod
    def get_empty_sentence(cls) -> "Sentence":
        """Creates and returns an empty Sentence object."""
        return cls(id=0, ner_tags=[], tokens=[])


def process_input_file(file_path: str):
    """
    Process input conllu file info the desired format for training.

    Args:
        file_path (str): The path of the file to be processed.

    Returns:
        sentences (list[Sentence]): A list of Sentence objects.
    """
    sentences: list[Sentence] = []
    current_sentence: Sentence = Sentence.get_empty_sentence()

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                if current_sentence.tokens:
                    current_sentence.id = len(sentences)
                    sentences.append(current_sentence)
                current_sentence = Sentence.get_empty_sentence()
            else:
                parts = line.split("\t")
                if len(parts) == 3:
                    _, token, label = parts
                    current_sentence.tokens.append(token)
                    current_sentence.ner_tags.append(label2id[label])

        if current_sentence.tokens:
            current_sentence.id = len(sentences)
            sentences.append(current_sentence)

    return sentences


@router.post('/', summary='Train a model')
def train_model(
    model_name: str = Form(...),
    model_language: str = Form(...),
    train_data: UploadFile = Form(...),
    valid_data: UploadFile = Form(...),
    test_data: UploadFile = Form(...),
    db: Session = Depends(get_db),
):
    """
    Initiates the training process for a Named-Entity Recognition (NER) model.

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

    return {
        'message': 'Successfully started training model.',
        'model_name': model_name,
        'training_id': training_id
    }
