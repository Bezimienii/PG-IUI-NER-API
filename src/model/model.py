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
from sklearn.model_selection import train_test_split
import evaluate
import datetime
import warnings
from ..db.db import get_db, Session
from ..utils.crud import create_model, delete_model, get_model, get_models
from ..processing.processing import process_input_file

MAX_LEN=128 
TEST_SIZE=0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

id2label = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}
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

def getBaseModel(lang):
    if lang == "en":
        return RobertaForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    elif lang == "pl":
        return AutoModelForTokenClassification.from_pretrained("pietruszkowiec/herbert-base-ner")
    else:
        return RobertaForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")

def getBaseModelForTraining(lang):
    if lang == "en":
        return RobertaForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    elif lang == "pl":
        return AutoModelForTokenClassification.from_pretrained("pietruszkowiec/herbert-base-ner", num_labels=7)
    else:
        return RobertaForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")

def prepare_tokenization(tokenizer):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            print(word_ids)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels

        return tokenized_inputs
    return tokenize_and_align_labels

def preprocess_data(tokenizer, data):
    tokenize_func = prepare_tokenization(tokenizer)
    return list(map(lambda x: tokenize_func(x), data))

def prepare_compute_metrics(label_list, seqeval):
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    return compute_metrics

def train_network(output_path, model, tokenizer, train, val, data_collator, label_list):
    training_args = TrainingArguments(
        output_dir=output_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    seqeval = evaluate.load("seqeval")
    compute_metrics = prepare_compute_metrics(label_list, seqeval)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=tokenizer,
    )
    trainer.train()

def getModelPath(lang):
    if lang == "en":
        return "Jean-Baptiste/roberta-large-ner-english"
    elif lang == "pl":
        return "pietruszkowiec/herbert-base-ner"
    else:
        return "Jean-Baptiste/roberta-large-ner-english"

def getTokeniser(lang):
    if lang == "en":
        return AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    elif lang == "pl":
        return AutoTokenizer.from_pretrained("pietruszkowiec/herbert-base-ner")
    else:
        return AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")

def getTokenizerFromPath(path):
    return AutoTokenizer.from_pretrained(path)

def getModelFromPath(path):
    print(path)
    #config = RobertaConfig()
    print(id2label)
    print(label2id)
    return AutoModelForTokenClassification.from_pretrained(path, num_labels=9, id2label=id2label, label2id=label2id,
                                                           ignore_mismatched_sizes=True)

def classifyPolishText(model, tokenizer, input):
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    return nlp(input)

def classifyText(model, tokenizer, input):
    inputs = tokenizer(
        input, add_special_tokens=False, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_token_class_ids = logits.argmax(-1)
    return [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

def train(model_path, output_path, train, val):
    label_list = list(id2label.values())
    model_name = getModelFromPath(model_path)
    model = getModelFromPath(model_name)
    tokenizer = getTokenizerFromPath(model_path)
    pre_train = preprocess_data(tokenizer, train)
    pre_val = preprocess_data(tokenizer, val)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    print(train[0])
    print(val[10])
    train_network(output_path, model, tokenizer, pre_train, pre_val, data_collator, label_list)

def execute_training(model_id):
    training_process_id = os.getpid()

    with Session() as db:
        # update_training_process_id(db, model_id, training_process_id)
        model_info = get_model(db, model_id)

    model_path = model_info.base_model
    output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models/"+ model_info.file_path)
    train_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../trainInfo/model1/train.conllu")
    val_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../trainInfo/model1/val.conllu")
    train_data = process_input_file(train_path)
    val_data = process_input_file(val_path)
    print(train_data[0])
    print(len(train_data))
    print(len(val_data))
    print(val_data[1])
    train(model_path, output_path, train_data[0:5000], val_data[0:500])