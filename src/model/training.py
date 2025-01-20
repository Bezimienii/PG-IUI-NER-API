import os

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from ..config import settings
from ..database.context_manager import Session
from ..utils.crud import get_model, update_training_process_id, update_training_status
from ..utils.models_utils import label2id, load_model_and_tokenizer
from ..utils.process_input_file import process_stream_file

MAX_LEN = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

id2label = {v: k for k, v in label2id.items()}

def tokenize_and_align_labels(examples, tokenizer):
        """Tokenize the input texts and align the labels with the tokenized input."""
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            is_split_into_words=True
        )
        labels = []

        """
        {
            "ner_tags": [
                            [0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 5, 0, 0]
                            [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 7, 8, 0, 0, 7, 8, 0, 0]
                        ]
            "tokens": [
                        ['This', 'division', 'also', 'contains', 'the', 'Ventana', 'Wilderness', ',', 'home', 'to', 'the', 'California', 'condor', '.']
                        ['"', 'So', 'here', 'is', 'the', 'balance', 'NBC', 'has', 'to', 'consider', ':', 'The', 'Who', ',', "'", 'Animal', 'Practice', "'", '.']
                    ]
        }
        """

        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


def train(model, tokenizer, train_file_path, valid_file_path, output_model_path, num_epochs=3):
    """Train the model using the provided model, tokenizer, train and val data stream, and output path."""
    training_args = TrainingArguments(
        output_dir=output_model_path,
        save_strategy='no',
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        tokenizer=tokenizer
    )

    for epoch in range(num_epochs):
        train_data_stream = process_stream_file(train_file_path, batch_size=100)
        for train_batch in train_data_stream:
            tokenized_train_batch = tokenize_and_align_labels(train_batch, tokenizer)
            train_dataset = Dataset.from_dict(tokenized_train_batch)

            trainer.train_dataset = train_dataset
            trainer.train()

        val_data_stream = process_stream_file(valid_file_path, batch_size=100)

        all_metrics = []
        for val_batch in val_data_stream:
            tokenized_val_batch = tokenize_and_align_labels(val_batch, tokenizer)
            val_dataset = Dataset.from_dict(tokenized_val_batch)

            metrics = trainer.evaluate(eval_dataset=val_dataset)
            all_metrics.append(metrics)
        print(f"Validation Metrics for Epoch {epoch + 1}: {all_metrics}")

    return model, tokenizer


def execute_training(model_id):
    """Execute the training process for the specified model ID."""
    training_process_id = os.getpid()

    with Session() as db:
        update_training_process_id(db, model_id, training_process_id)
        model_info = get_model(db, model_id)

    model, tokenizer = load_model_and_tokenizer(model_info, train=True)

    output_model_path = f'{settings.MODEL_PATH}/{model_info.model_name}'

    model, tokenizer = train(
        model,
        tokenizer,
        model_info.train_file_path,
        model_info.valid_file_path,
        output_model_path
    )

    tokenizer.save_pretrained(output_model_path)
    model.save_pretrained(output_model_path)

    with Session() as db:
        update_training_status(db, model_id, is_training=False, is_trained=True)
