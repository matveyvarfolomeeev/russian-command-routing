"""Protect native pair inputs from label leakage and silent truncation."""

import pytest

from scripts.train_modernbert_kaggle import MAX_PAIR_TOKENS, encode_all


class RecordingTokenizer:
    def __init__(self, size=8):
        self.size = size
        self.calls = []

    def __call__(self, questions, cards, **kwargs):
        self.calls.append((questions, cards, kwargs))
        return {"input_ids": [[len(q), len(c)]+[1]*(self.size-2)
                              for q, c in zip(questions, cards, strict=True)]}


def test_training_labels_and_provenance_cannot_change_model_inputs():
    tokenizer = RecordingTokenizer()
    row = {"id": "row", "query": "открой редактор", "documents": [f"card-{i}" for i in range(20)],
           "target": "object-1", "candidate_ids": [f"object-{i}" for i in range(20)], "source": "clean"}
    first = encode_all(tokenizer, [row])[0]
    changed = {**row, "target": "object-19", "id": "another", "source": "stt",
               "candidate_ids": [f"different-{i}" for i in range(20)]}
    assert encode_all(tokenizer, [changed])[0] == first
    assert tokenizer.calls[0] == tokenizer.calls[1]
    questions, cards, options = tokenizer.calls[0]
    assert questions == [row["query"]]*20 and cards == row["documents"]
    assert options["truncation"] is False


def test_overlong_pair_aborts_instead_of_dropping_metadata():
    row = {"id": "too-long", "query": "request", "documents": ["card"]*20}
    with pytest.raises(ValueError, match="truncate"):
        encode_all(RecordingTokenizer(MAX_PAIR_TOKENS+1), [row])


def test_bert_pair_segments_survive_encoding():
    class BertTokenizer(RecordingTokenizer):
        def __call__(self, questions, cards, **kwargs):
            values = super().__call__(questions, cards, **kwargs)
            values["token_type_ids"] = [[0, 0] + [1]*(self.size-2) for _ in cards]
            return values

    row = {"id": "pair", "query": "открой редактор", "documents": ["редактор кода"]*20}
    encoded, _ = encode_all(BertTokenizer(), [row])
    assert all(pair["token_type_ids"] == [0, 0, 1, 1, 1, 1, 1, 1] for pair in encoded[0])
