"""Download the exact E5 revision used by the application, without a new head."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def prepare():
    os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_XET="1")
    from huggingface_hub import snapshot_download
    from train_modernbert_kaggle import sha

    output = ROOT / "artifacts/kaggle/e5-source-v1"
    original = output / "original"
    if (output / "source.json").exists():
        raise FileExistsError("Source is already pinned; do not replace it")
    snapshot_download(MODEL_ID, revision=REVISION, token=False, local_dir=original,
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "README.md",
                                      "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"])
    from train_e5_kaggle import load_model

    model, _ = load_model(original, "cpu")
    source = {"model_id": MODEL_ID, "revision": REVISION,
              "source_url": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}", "license": "MIT",
              "files": {path.name: sha(path) for path in original.iterdir() if path.is_file()},
              "parameters": sum(value.numel() for value in model.parameters()),
              "layers": model.config.num_hidden_layers, "embedding_size": model.config.hidden_size,
              "maximum_tokens": 512, "head_added": False,
              "embedding": "attention-mask mean pooling, L2 normalization; query:/passage: prefixes",
              "transformers": __import__("transformers").__version__,
              "torch": __import__("torch").__version__, "checkpoint_loaded_exactly": True}
    (output / "source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in source.items() if key != "files"}), flush=True)


if __name__ == "__main__":
    prepare()
