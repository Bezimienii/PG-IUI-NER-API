from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from transformers import DataCollatorForTokenClassification

def fine_tune_model(train_texts, train_labels, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """
    Fine-tunes a token classification model without IOB format.

    Args:
        train_texts (list): List of training sentences (tokenized words).
        train_labels (list): List of training labels (integer IDs).
        model (PreTrainedModel): Initialized Hugging Face model.
        tokenizer (PreTrainedTokenizer): Initialized tokenizer corresponding to the model.

    Returns:
        model: The fine-tuned model.
    """
    id2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 
                5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    label2id = {v: k for k, v in id2label.items()}

    # Tokenize and align labels
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding="max_length", max_length=512, is_split_into_words=True)
        labels = []

        for i, label in enumerate(examples['labels']):
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

    # Prepare dataset
    dataset = Dataset.from_dict({"tokens": train_texts, "labels": train_labels})
    tokenized_datasets = dataset.map(
        lambda x: tokenize_and_align_labels(x), batched=True
    )
    
    training_args = TrainingArguments(
        output_dir='models/tmp',
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        do_eval=False,
        eval_strategy='no'
    )

    # Initialize Trainer with data collator instead of directly passing tokenizer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
    )

    trainer.train()

    return model
