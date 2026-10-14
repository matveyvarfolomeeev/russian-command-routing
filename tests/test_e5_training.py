"""Protect E5 prefixes, masking, target isolation and retrieval gradients."""

from types import SimpleNamespace

import pytest

from scripts.prepare_e5_experiment import passage_text
from scripts.train_e5_kaggle import (
    MAX_TOKENS,
    evaluate,
    mean_pool_normalize,
    prepare_training,
    prepare_validation,
    text_bank,
    training_logits,
)


class Tokenizer:
    def __init__(self, size=1):
        self.mapping = {}
        self.size = size

    def __call__(self, texts, **kwargs):
        assert kwargs == {"padding": False, "truncation": False}
        for text in texts:
            self.mapping.setdefault(text, len(self.mapping))
        ids = [[self.mapping[text]] * self.size for text in texts]
        return {"input_ids": ids, "attention_mask": [[1] * self.size for _ in texts]}

    def pad(self, data, **kwargs):
        import torch

        return {key: torch.tensor([row[key] for row in data]) for key in data[0]}


def row():
    return {"id": "first", "query": "открой редактор", "documents": [f"карточка-{i}" for i in range(20)],
            "candidate_ids": [f"id-{i}" for i in range(20)], "target": "id-3", "source": "clean"}


def test_labels_ids_and_provenance_do_not_enter_encoder_inputs():
    value = row()
    bank, _ = prepare_training(Tokenizer(), [value])
    changed = {**value, "id": "changed", "source": "stt", "group_id": "different",
               "candidate_ids": [f"other-{i}" for i in range(20)], "target": "other-19"}
    other, _ = prepare_training(Tokenizer(), [changed])
    assert bank["texts"] == other["texts"] and bank["encoded"] == other["encoded"]
    assert bank["texts"][0] == "query: открой редактор"
    assert bank["texts"][1] == "passage: карточка-0"


def test_overlong_text_is_rejected():
    with pytest.raises(ValueError, match="512"):
        text_bank(Tokenizer(MAX_TOKENS + 1), ["query: длинный текст"])


def test_public_card_renderer_matches_frozen_training_documents():
    import json
    from pathlib import Path
    from zipfile import ZipFile

    root = Path(__file__).resolve().parents[1]
    with ZipFile(root / "data/prepared/launcher-choice-v1.zip") as archive:
        validation = [json.loads(line) for line in archive.read("prepared-validation.jsonl").splitlines()]
    catalogs = {}
    for row in validation:
        profile = row["profile"]
        if profile not in catalogs:
            path = root / "data/benchmarks" / profile / "synthetic/catalog.json"
            catalogs[profile] = {obj["object_id"]: obj for obj in json.loads(path.read_text("utf-8"))["objects"]}
        expected = [passage_text(catalogs[profile][oid]).removeprefix("passage: ")
                    for oid in row["candidate_ids"]]
        assert row["documents"] == expected


def test_masked_padding_cannot_change_embedding():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[3., 4.], [100., -200.]]])
    mask = torch.tensor([[1, 0]])
    assert torch.allclose(mean_pool_normalize(hidden, mask), torch.tensor([[.6, .8]]))
    assert torch.allclose(mean_pool_normalize(hidden[:, :1], mask[:, :1]), mean_pool_normalize(hidden, mask))


def toy_model(torch, count):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(count, 8)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    torch.manual_seed(8)
    return Model()


def test_contrastive_loss_updates_query_and_positive_document():
    torch = pytest.importorskip("torch")
    tokenizer = Tokenizer()
    bank, prepared = prepare_training(tokenizer, [row()])
    model = toy_model(torch, len(bank["texts"]))
    logits = training_logits(model, tokenizer, bank, prepared, [0], "cpu")
    torch.nn.functional.cross_entropy(logits, torch.tensor([prepared[0]["label"]])).backward()
    grad = model.embedding.weight.grad
    assert torch.isfinite(grad).all()
    assert grad[prepared[0]["query"]].abs().sum() > 0
    assert grad[prepared[0]["documents"][prepared[0]["label"]]].abs().sum() > 0


def test_candidate_permutation_only_permutes_scores():
    torch = pytest.importorskip("torch")
    tokenizer = Tokenizer()
    bank, prepared = prepare_training(tokenizer, [row()])
    model = toy_model(torch, len(bank["texts"]))
    scores = training_logits(model, tokenizer, bank, prepared, [0], "cpu")
    reverse = [{**prepared[0], "documents": list(reversed(prepared[0]["documents"]))}]
    shuffled = training_logits(model, tokenizer, bank, reverse, [0], "cpu")
    assert torch.allclose(scores.flip(-1), shuffled)


def test_full_catalog_recall_and_nll_include_a_target_beyond_top20(tmp_path):
    torch = pytest.importorskip("torch")
    tokenizer = Tokenizer()
    value = {"id": "one", "query": "запрос", "target": "id-20", "group_id": "group",
             "profile": "validation", "catalog_key": "catalog"}
    catalog = {"candidate_ids": [f"id-{i}" for i in range(21)], "documents": [f"doc-{i}" for i in range(21)]}
    bank = prepare_validation(tokenizer, [value], {"catalog": catalog})
    model = toy_model(torch, len(bank["texts"]))
    with torch.no_grad():
        model.embedding.weight.zero_()
        model.embedding.weight[:, 0] = 1
        model.embedding.weight[bank["index"]["passage: doc-20"], 0] = -1
    result = evaluate(model, tokenizer, bank, [value], {"catalog": catalog}, "cpu", tmp_path, "full")
    assert result["metrics"]["validation"]["recall20"] == 0
    assert result["metrics"]["validation"]["accuracy"] == 0
    assert result["selection_loss"] > 20
