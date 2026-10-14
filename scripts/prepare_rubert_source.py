"""Pin RuBERT-tiny2 and attach one deterministic scalar choice head locally."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "cointegrated/rubert-tiny2"
REVISION = "e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae"
HEAD_SEED = 20261007


def prepare():
    os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_DISABLE_XET="1")
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from train_modernbert_kaggle import load_model, sha
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    output = ROOT/"artifacts/kaggle/rubert-source-v1"
    original, checkpoint = output/"original", output/"checkpoint"
    if (output/"source.json").exists():
        raise FileExistsError("Pinned source already exists; inspect it instead of replacing it")
    snapshot_download(
        MODEL_ID, revision=REVISION, token=False, local_dir=original,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "README.md"])
    torch.set_num_threads(4)
    torch.manual_seed(HEAD_SEED)
    model, info = AutoModelForSequenceClassification.from_pretrained(
        original, num_labels=1, local_files_only=True, output_loading_info=True,
        attn_implementation="sdpa")
    assert set(info["missing_keys"]) == {"classifier.weight", "classifier.bias"}, info
    assert not info["mismatched_keys"] and not info["error_msgs"], info
    assert all(key.startswith("cls.") or key == "bert.embeddings.position_ids"
               for key in info["unexpected_keys"]), info
    with safe_open(original/"model.safetensors", framework="pt") as saved:
        encoder_keys = [key for key in model.state_dict() if key.startswith("bert.")]
        for key in encoder_keys:
            assert torch.equal(model.state_dict()[key].cpu(), saved.get_tensor(key)), key
    tokenizer = AutoTokenizer.from_pretrained(original, local_files_only=True)
    probe = tokenizer("открой редактор", "Редактор кода. Приложение.", return_tensors="pt")
    assert "token_type_ids" in probe and set(probe["token_type_ids"][0].tolist()) == {0, 1}
    with torch.inference_mode():
        assert torch.isfinite(model(**probe).logits).all()
    model.float().save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    (checkpoint/"README.md").write_bytes((original/"README.md").read_bytes())
    reloaded, _ = load_model(checkpoint, "cpu")
    reloaded.eval()
    model.eval()
    with torch.inference_mode():
        assert torch.equal(model(**probe).logits, reloaded(**probe).logits)
    source = {
        "model_id": MODEL_ID, "revision": REVISION,
        "source_url": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
        "license": "MIT", "head_seed": HEAD_SEED,
        "original_files": {p.name: sha(p) for p in original.iterdir() if p.is_file()},
        "initial_choice_weight_sha256": sha(checkpoint/"model.safetensors"),
        "encoder_parameters_preserved_exactly": encoder_keys,
        "new_parameters": ["classifier.weight", "classifier.bias"],
        "removed_pretraining_head": sorted(info["unexpected_keys"]),
        "parameters": sum(p.numel() for p in model.parameters()),
        "layers": model.config.num_hidden_layers, "hidden_size": model.config.hidden_size,
        "maximum_pair_tokens": 2048, "pair_segments_verified": True,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "data_used_for_initialization": False,
    }
    (output/"source.json").write_text(json.dumps(source, indent=2)+"\n", encoding="utf8")
    print(json.dumps({k: v for k, v in source.items()
                      if k not in {"original_files", "encoder_parameters_preserved_exactly",
                                   "removed_pretraining_head"}}), flush=True)


if __name__ == "__main__":
    prepare()
