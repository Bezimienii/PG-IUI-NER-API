import os
import pandas as pd
import numpy as np
import json
import re
from nltk.tokenize import sent_tokenize
from transformers import RobertaForTokenClassification, AutoModelForTokenClassification
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split
import datetime
import warnings

MAX_LEN=128
TEST_SIZE=0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def getBaseModel(lang):
    if lang == "en":
        return RobertaForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    elif lang == "pl":
        return AutoModelForTokenClassification.from_pretrained("pczarnik/herbert-base-ner")
    else:
        return RobertaForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")

def classifyText(model, tokenizer, input):
    inputs = tokenizer(
        input, add_special_tokens=False, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_token_class_ids = logits.argmax(-1)
    return [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

def clean_text(txt):
    '''
    This is text cleaning function
    '''
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())


def data_joining(data_dict_id):
    '''
    This function is to join all the text data from different
    sections in the json to a single text file.
    '''
    data_length = len(data_dict_id)

    #     temp = [clean_text(data_dict_id[i]['text']) for i in range(data_length)]
    temp = [data_dict_id[i]['text'] for i in range(data_length)]
    temp = '. '.join(temp)

    return temp


def make_shorter_sentence(sentence, max_len=MAX_LEN):
    '''
    This function is to split the long sentences into chunks of shorter sentences upto the
    maximum length of words specified in config['MAX_LEN']
    '''
    sent_tokenized = sent_tokenize(sentence)

    max_length = max_len
    overlap = 20

    final_sentences = []

    for tokenized_sent in sent_tokenized:
        sent_tokenized_clean = clean_text(tokenized_sent)
        sent_tokenized_clean = sent_tokenized_clean.replace('.', '').rstrip()

        tok_sent = sent_tokenized_clean.split(" ")

        if len(tok_sent) < max_length:
            final_sentences.append(sent_tokenized_clean)
        else:
            #             print("Making shorter sentences")
            start = 0
            end = len(tok_sent)

            for i in range(start, end, max_length - overlap):
                temp = tok_sent[i: (i + max_length)]
                final_sentences.append(" ".join(i for i in temp))

    return final_sentences

def form_labels(sentence, labels_list, tokenizer):
    '''
    This function labels the training data
    '''
    matched_kwords = []
    matched_token = []
    un_matched_kwords = []
    label = []

    # Since there are many sentences which are more than 512 words,
    # Let's make the max length to be 128 words per sentence.
    tokens = make_shorter_sentence(sentence)

    for tok in tokens:
        tok_split = tokenizer.tokenize(tok)

        z = np.array(['O'] * len(tok_split))  # Create final label == len(tokens) of each sentence
        matched_keywords = 0  # Initially no kword matched

        for kword in labels_list:
            if kword in tok:  # This is to first check if the keyword is in the text and then go ahead
                kword_split = tokenizer.tokenize(kword)
                for i in range(len(tok_split)):
                    if tok_split[i: (i + len(kword_split))] == kword_split:
                        matched_keywords += 1

                        if (len(kword_split) == 1):
                            z[i] = 'B'
                        else:
                            z[i] = 'B'
                            z[(i + 1): (i + len(kword_split))] = 'B'

                        if matched_keywords > 1:
                            label[-1] = (z.tolist())
                            matched_token[-1] = tok
                            matched_kwords[-1].append(kword)
                        else:
                            label.append(z.tolist())
                            matched_token.append(tok)
                            matched_kwords.append([kword])
                    else:
                        un_matched_kwords.append(kword)

    return matched_token, matched_kwords, label, un_matched_kwords

def labelling(dataset, data_dict, train_df):
    '''
    This function is to iterate each of the training data and get it labelled
    from the form_labels() function.
    '''

    Id_list_ = []
    sentences_ = []
    key_ = []
    labels_ = []
    un_mat = []
    un_matched_reviews = 0

    for i, Id in tqdm(enumerate(dataset.Id), total=len(dataset.Id)):

        sentence = data_joining(data_dict[Id])
        labels = train_df.label[train_df.Id == Id].tolist()[0].split("|")

        s, k, l, un_matched = form_labels(sentence=sentence, labels_list=labels)

        if len(s) == 0:
            un_matched_reviews += 1
            un_mat.append(un_matched)
        else:
            sentences_.append(s)
            key_.append(k)
            labels_.append(l)
            Id_list_.append([Id] * len(l))

    print("Total unmatched keywords:", un_matched_reviews)
    sentences = [item for sublist in sentences_ for item in sublist]
    final_labels = [item for sublist in labels_ for item in sublist]
    keywords = [item for sublist in key_ for item in sublist]
    Id_list = [item for sublist in Id_list_ for item in sublist]

    return sentences, final_labels, keywords, Id_list

def convert_2_dataframe(train_Id_list, train_sentences, train_keywords, train_labels):
    unique_df = pd.DataFrame({'id': train_Id_list,
                              'train_sentences': train_sentences,
                              'kword': train_keywords,
                              'label': train_labels})
    unique_df.label = unique_df.label.astype('str')
    unique_df.kword = unique_df.kword.astype('str')
    unique_df['sent_len'] = unique_df.train_sentences.apply(lambda x: len(x.split(" ")))
    unique_df = unique_df.drop_duplicates()

def split_data(unique_df, test_size=TEST_SIZE):
    train_df, valid_df = train_test_split(unique_df, test_size=test_size)
    return train_df, valid_df

def dataset_2_list(df):
    tags_2_idx = {'O': 0, 'B': 1, 'P': 2}
    id_list = df.id.values.tolist()
    sentences_list = df.train_sentences.values.tolist()
    keywords_list = df.kword.apply(lambda x: eval(x)).values.tolist()

    labels_list = df.label.apply(lambda x: eval(x)).values.tolist()
    labels_list = [list(map(tags_2_idx.get, lab)) for lab in labels_list]

    return id_list, sentences_list, keywords_list, labels_list

class form_input():
    def __init__(self, ID, sentence, kword, label, tokenizer, data_type='test', max_len=MAX_LEN):
        self.id = ID
        self.sentence = sentence
        self.kword = kword
        self.label = label
        self.max_length = max_len
        self.tokenizer = tokenizer
        self.data_type = data_type

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, item, tokenizer):
        toks = tokenizer.tokenize(self.sentence[item])
        label = self.label[item]

        if len(toks) > self.max_length:
            toks = toks[:self.max_length]
            label = label[:self.max_length]

        ########################################
        # Forming the inputs
        ids = tokenizer.convert_tokens_to_ids(toks)
        tok_type_id = [0] * len(ids)
        att_mask = [1] * len(ids)

        # Padding
        pad_len = self.max_length - len(ids)
        ids = ids + [2] * pad_len
        tok_type_id = tok_type_id + [0] * pad_len
        att_mask = att_mask + [0] * pad_len

        ########################################
        # Forming the label
        if self.data_type != 'test':
            label = label + [2] * pad_len
        else:
            label = 1

        return {'ids': torch.tensor(ids, dtype=torch.long),
                'tok_type_id': torch.tensor(tok_type_id, dtype=torch.long),
                'att_mask': torch.tensor(att_mask, dtype=torch.long),
                'target': torch.tensor(label, dtype=torch.long)
                }


def train_fn(data_loader, model, optimizer, device=DEVICE):
    '''
    Functiont to train the model
    '''

    train_loss = 0
    for index, dataset in enumerate(tqdm(data_loader, total=len(data_loader))):
        batch_input_ids = dataset['ids'].to(device, dtype=torch.long)
        batch_att_mask = dataset['att_mask'].to(device, dtype=torch.long)
        batch_tok_type_id = dataset['tok_type_id'].to(device, dtype=torch.long)
        batch_target = dataset['target'].to(device, dtype=torch.long)

        output = model(batch_input_ids,
                       token_type_ids=None,
                       attention_mask=batch_att_mask,
                       labels=batch_target)

        step_loss = output[0]
        prediction = output[1]

        step_loss.sum().backward()
        optimizer.step()
        train_loss += step_loss
        optimizer.zero_grad()

    return train_loss.sum()


def eval_fn(data_loader, model, max_len=MAX_LEN, device=DEVICE):
    '''
    Functiont to evaluate the model on each epoch.
    We can also use Jaccard metric to see the performance on each epoch.
    '''

    model.eval()

    eval_loss = 0
    predictions = np.array([], dtype=np.int64).reshape(0, max_len)
    true_labels = np.array([], dtype=np.int64).reshape(0, max_len)

    with torch.no_grad():
        for index, dataset in enumerate(tqdm(data_loader, total=len(data_loader))):
            batch_input_ids = dataset['ids'].to(device, dtype=torch.long)
            batch_att_mask = dataset['att_mask'].to(device, dtype=torch.long)
            batch_tok_type_id = dataset['tok_type_id'].to(device, dtype=torch.long)
            batch_target = dataset['target'].to(device, dtype=torch.long)

            output = model(batch_input_ids,
                           token_type_ids=None,
                           attention_mask=batch_att_mask,
                           labels=batch_target)

            step_loss = output[0]
            eval_prediction = output[1]

            eval_loss += step_loss

            eval_prediction = np.argmax(eval_prediction.detach().to('cpu').numpy(), axis=2)
            actual = batch_target.to('cpu').numpy()

            predictions = np.concatenate((predictions, eval_prediction), axis=0)
            true_labels = np.concatenate((true_labels, actual), axis=0)

    return eval_loss.sum(), predictions, true_labels

def train_engine(model, epoch, train_data, valid_data, model_name):
    model = nn.DataParallel(model)
    model = model.to(DEVICE)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=3e-5)

    best_eval_loss = 1000000
    for i in range(epoch):
        train_loss = train_fn(data_loader=train_data,
                              model=model,
                              optimizer=optimizer)
        eval_loss, eval_predictions, true_labels = eval_fn(data_loader=valid_data,
                                                           model=model)

        print(f"Epoch {i} , Train loss: {train_loss}, Eval loss: {eval_loss}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss

            print("Saving the model")
            torch.save(model.state_dict(), model_name)

    return model, eval_predictions, true_labels