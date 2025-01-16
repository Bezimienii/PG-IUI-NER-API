import os
import pandas as pd
import numpy as np
import json
import re
from nltk.tokenize import sent_tokenize
from transformers import RobertaForTokenClassification, AutoModelForTokenClassification, AutoTokenizer, pipeline, \
    DataCollatorForTokenClassification, PretrainedConfig, RobertaConfig
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
import glob
import evaluate
import datetime
import warnings
from ..db.db import get_db, Session
from ..processing.process_input_file import process_input_file
from ..utils.crud import create_model, delete_model, get_model, get_models, update_training_process_id
from ..utils.models_utils import load_model_and_tokenizer
from datasets import Dataset, DatasetDict
from ..config import settings
MAX_LEN = 512 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

id2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 
            5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
label2id = {v: k for k, v in id2label.items()}

def tokenize_and_align_labels(examples, tokenizer):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding="max_length", max_length=MAX_LEN, is_split_into_words=True)
        labels = []

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

def train(model, tokenizer, dataset, output_model_path):
    training_args = TrainingArguments(
        output_dir=output_model_path,
        save_strategy='no',
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        tokenizer=tokenizer
    )

    trainer.train()
    tokenizer.save_pretrained(output_model_path)
    model.save_pretrained(output_model_path)

def process_files(path: str):
    texts = []
    labels = []

    with open(path, 'r') as f:
        for line in f:
            text, label = line.strip().split(';')
            texts.append(text.split(','))  # Split text into words
            labels.append([int(x) for x in label.split(',')])  # Convert labels to integers
    
    return {'tokens': texts, 'ner_tags': labels}

def execute_training(model_id):
    training_process_id = os.getpid()

    with Session() as db:
        update_training_process_id(db, model_id, training_process_id)
        model_info = get_model(db, model_id)

    print(model_info)
    # Load base model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_info, train=True)
    
    # Set ouput file
    output_model_path = f'{settings.MODEL_PATH}/{model_info.model_name}'

    train_data = process_files(model_info.train_file_path)
    val_data = process_files(model_info.valid_file_path)

    # Convert to Dataset and map over elements using tokenize_and_align_labels
    datasets = DatasetDict({
        'train': Dataset.from_dict(train_data),
        'val': Dataset.from_dict(val_data)
    })
    tokenized_datasets = datasets.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    train(model, tokenizer, tokenized_datasets, output_model_path)