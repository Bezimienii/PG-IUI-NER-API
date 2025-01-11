# Define the align_target function
def align_target(labels, word_ids):
    """
    Aligns the target labels with the subword tokenization.
    
    Args:
        labels (list): List of integer labels corresponding to words.
        word_ids (list): List of word IDs from tokenization.
    
    Returns:
        list: Aligned labels for each token.
    """
    align_labels = []  # Store aligned labels
    last_word = None   # Track the last word ID

    for word in word_ids:
        if word is None:  # Special tokens like [CLS] and [SEP]
            label = -100  # Ignore labels for these tokens
        elif word != last_word:  # If it's a new word
            label = labels[word]  # Assign the label corresponding to the word
        else:
            label = labels[word]  # For subwords, keep the same label

        align_labels.append(label)  # Add to aligned labels
        last_word = word  # Update the last word ID

    return align_labels


# Fine-tune model function
from transformers import Trainer, TrainingArguments
from datasets import Dataset

def fine_tune_model(train_texts, train_labels, id2label, model, tokenizer):
    """
    Fine-tunes a token classification model without IOB format.

    Args:
        train_texts (list): List of training sentences (tokenized words).
        train_labels (list): List of training labels (integer IDs).
        id2label (dict): Mapping from label IDs to label names.
        model (PreTrainedModel): Initialized Hugging Face model.
        tokenizer (PreTrainedTokenizer): Initialized tokenizer corresponding to the model.

    Returns:
        model: The fine-tuned model.
    """
    # Ensure id2label is passed to the model config
    model.config.id2label = id2label
    model.config.label2id = {v: k for k, v in id2label.items()}

    # Tokenize and align labels
    def tokenize_and_align_labels(texts, labels):
        tokenized_inputs = tokenizer(texts, truncation=True, padding=True, is_split_into_words=True)
        aligned_labels = []
        for i, label in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned_labels.append(align_target(label, word_ids))
        tokenized_inputs["labels"] = aligned_labels
        return tokenized_inputs

    # Prepare dataset
    dataset = Dataset.from_dict({"tokens": train_texts, "labels": train_labels})
    tokenized_datasets = dataset.map(
        lambda x: tokenize_and_align_labels(x["tokens"], x["labels"]), batched=True
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results/fine_tune",
        learning_rate=2e-3,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="no"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()
    print("Model fine-tuning completed!")
    return model