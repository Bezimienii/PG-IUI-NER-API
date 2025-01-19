from pydantic import BaseModel

from .models_utils import label2id


class Sentence(BaseModel):
    """A class used to represent a sentence."""
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


def process_stream_file(file_path: str, batch_size: int = 100_000):
    """Process conllu file into the desired format for training in a streaming manner.

    Args:
        file_path (str): The path of the file to be processed.
        batch_size (int): The number of sentences to merge into a single batch.

    Returns:
        sentences (list[Sentence]): A list of Sentence objects.
    """
    current_sentence: Sentence = Sentence.get_empty_sentence()
    batch: list[Sentence] = []

    with open(file_path, encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                if current_sentence.tokens:
                    current_sentence.id = len(batch)
                    batch.append(current_sentence)
                current_sentence = Sentence.get_empty_sentence()

                if len(batch) >= batch_size:
                    yield merge_sentences(batch)
                    batch = []
            else:
                parts = line.split('\t')
                if len(parts) == 3:
                    _, token, label = parts
                    current_sentence.tokens.append(token)
                    current_sentence.ner_tags.append(label2id[label])

        if current_sentence.tokens:
            current_sentence.id = len(batch)
            batch.append(current_sentence)
        if batch:
            yield merge_sentences(batch)


def merge_sentences(batch: list[Sentence]) -> dict:
    """Merge a batch of Sentence objects into a single sentence.

    Args:
        batch (list[Sentence]): A list of Sentence objects to merge.

    Returns:
        dict: A dictionary representing the merged sentences.
    """
    merged_tokens = []
    merged_ner_tags = []

    for sentence in batch:
        merged_tokens.extend(sentence.tokens)
        merged_ner_tags.extend(sentence.ner_tags)

    merged_sentence = Sentence(
        id=0,
        tokens=merged_tokens,
        ner_tags=merged_ner_tags,
    )
    return merged_sentence.to_dict()