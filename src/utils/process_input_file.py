from pydantic import BaseModel
from .models_utils import label2id


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
