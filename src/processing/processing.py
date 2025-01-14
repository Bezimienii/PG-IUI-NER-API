import os
import uuid

from fastapi import APIRouter, Depends, Form, UploadFile
from pydantic import BaseModel

from src.db.db import Session, get_db
from ..model.model import execute_training

from ..utils.crud import create_model
from datetime import datetime
from multiprocessing import Process

router = APIRouter(prefix='/api/processing', tags=['processing'])

# UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../files'))
UPLOAD_DIR = '/tmp/files'
os.makedirs(UPLOAD_DIR, exist_ok=True)

label2id = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}

def save_file(file: UploadFile, name: str) -> str:
    """Save an uploaded file to the UPLOAD_DIR with a new name.

    Args:
        file (UploadFile): The file to be saved.
        name (str): The new name for the file.

    Returns:
       file_path (str): The path of the saved file.
    """
    file_extension = os.path.splitext(file.filename)[-1]
    new_filename = f'{name}{file_extension}'
    file_path = os.path.join(UPLOAD_DIR, new_filename)

    print(file_path)
    with open(file_path, 'wb') as buffer:
        buffer.write(file.file.read())

    return file_path


class Sentence(BaseModel):
    id: int
    ner_tags: list[int]
    tokens: list[str]

    @classmethod
    def get_empty_sentence(cls) -> 'Sentence':
        """Creates and returns an empty Sentence object."""
        return cls(id=0, ner_tags=[], tokens=[])

    def to_dict(self) -> dict:
        """Converts the Sentence object into a simple dictionary."""
        return {
            "id": self.id,
            "ner_tags": self.ner_tags,
            "tokens": self.tokens,
        }

def process_input_file(file_path: str):
    """Process input conllu file into the desired format for training.

    Args:
        file_path (str): The path of the file to be processed.

    Returns:
        sentences (list[Sentence]): A list of Sentence objects.
    """
    sentences: list[Sentence] = []
    current_sentence: Sentence = Sentence.get_empty_sentence()

    with open(file_path, encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                if current_sentence.tokens:
                    current_sentence.id = len(sentences)
                    sentences.append(current_sentence)
                current_sentence = Sentence.get_empty_sentence()
            else:
                parts = line.split('\t')
                if len(parts) == 3:
                    _, token, label = parts
                    current_sentence.tokens.append(token)
                    current_sentence.ner_tags.append(label2id[label])

        if current_sentence.tokens:
            current_sentence.id = len(sentences)
            sentences.append(current_sentence)

    merged_sentences: list[Sentence] = []
    for i in range(0, len(sentences), 5):
        batch = sentences[i:i + 5]
        merged_tokens = []
        merged_ner_tags = []

        for sentence in batch:
            merged_tokens.extend(sentence.tokens)
            merged_ner_tags.extend(sentence.ner_tags)

        merged_sentence = Sentence(
            id=len(merged_sentences),
            tokens=merged_tokens,
            ner_tags=merged_ner_tags,
        )
        merged_sentences.append(merged_sentence)

    return [sentence.to_dict() for sentence in merged_sentences]


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
    files_uuid = uuid.uuid4()
    train_path = save_file(train_data, f'train_{files_uuid}')
    valid_path = save_file(valid_data, f'valid_{files_uuid}')
    test_path = save_file(test_data, f'test_{files_uuid}')

    model = create_model( # TODO: create a training model (missing fields for train_path, valid_path, test_path, model_name, training_process_id
        session=db,
        base_model=model_language,
        file_path=model_name,
        is_training=True,
        is_trained=False,
        date_created=datetime.now(),
    )
    # model = create_model(
    #     # expected entity
    #     session=db,
    #     base_model=model_language,
    #     model_name=model_name,
    #     train_file_path=train_path,
    #     valid_file_path=valid_path,
    #     test_file_path=test_path,
    #     training_process_id=None, # to be updated in child process
    #     is_training=True,
    #     is_trained=False,
    #     date_created=datetime.now(),
    # )

    model_id = model.id

    p = Process(target=execute_training, args=(model_id,)) # process independent from parent
    p.start()
    # TODO: save pid in the child project to model entity

    return {'message': 'Successfully started training model.', 'model_name': model_name, 'training_process_id': p.pid}
