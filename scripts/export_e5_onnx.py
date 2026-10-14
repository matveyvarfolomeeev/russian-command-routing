"""Export the selected E5 hidden states without changing its pooling or precision."""

import argparse
import json
import os
from pathlib import Path

from train_e5_kaggle import load_model, mean_pool_normalize
from train_modernbert_kaggle import sha


def run(args):
    os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
    import numpy as np
    import onnxruntime as ort
    import torch

    torch.set_num_threads(4)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh export directory")
    args.output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.checkpoint, "cpu", attention="eager")
    model.requires_grad_(False)

    class HiddenStates(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = model

        def forward(self, input_ids, attention_mask):
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    example = tokenizer(["query: открой редактор", "passage: Блокнот. Приложение. Запись заметок"],
                        padding=True, return_tensors="pt")
    destination = args.output / "model.onnx"
    torch.onnx.export(HiddenStates().eval(), (example["input_ids"], example["attention_mask"]),
                      destination, input_names=["input_ids", "attention_mask"],
                      output_names=["last_hidden_state"], opset_version=17, dynamo=False,
                      external_data=False,
                      dynamic_axes={"input_ids": {0: "batch", 1: "tokens"},
                                    "attention_mask": {0: "batch", 1: "tokens"},
                                    "last_hidden_state": {0: "batch", 1: "tokens"}})
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(destination), sess_options=options, providers=["CPUExecutionProvider"])
    probes = [["query: код"], ["query: открой редактор кода", "passage: Заметки. Приложение"],
              ["query: " + "документ " * 30, "query: музыка", "passage: " + "редактор " * 50]]
    differences = []
    with torch.inference_mode():
        for texts in probes:
            batch = tokenizer(texts, padding=True, return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            actual = session.run(None, {key: batch[key].numpy() for key in ("input_ids", "attention_mask")})[0]
            reference = mean_pool_normalize(hidden, batch["attention_mask"]).numpy()
            embedded = mean_pool_normalize(torch.from_numpy(actual), batch["attention_mask"]).numpy()
            difference = float(np.max(np.abs(reference - embedded)))
            if difference > 1e-5 or not np.isfinite(actual).all():
                raise ValueError(f"ONNX pooling parity failed: {difference}")
            differences.append({"batch": len(texts), "tokens": batch["input_ids"].shape[1],
                                "maximum_embedding_difference": difference})
    report = {"precision": "fp32", "head_added": False, "opset": 17,
              "checkpoint_sha256": sha(args.checkpoint / "model.safetensors"),
              "onnx_sha256": sha(destination), "tokenizer_sha256": sha(args.checkpoint / "tokenizer.json"),
              "probes": differences, "exporter": "legacy torch.onnx, eager attention, dynamic batch/tokens"}
    (args.output / "export.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
