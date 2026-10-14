"""Independently check the scalar fit with Torch autograd and frozen prediction files."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "artifacts/benchmarks/laya-temperature-v1"


def read(path):
    return json.loads(path.read_text("utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run():
    import torch
    from torch.nn import functional as functional

    torch.set_num_threads(1)
    protocol = read(DIRECTORY / "protocol.json")
    result = read(DIRECTORY / "temperature.json")
    assert sha(DIRECTORY / "protocol.json") == result["protocol_sha256"]
    assert sha(Path(__file__).with_name("calibrate_laya_temperature.py")) == protocol["script_sha256"]
    source = ROOT / protocol["input"]
    assert sha(source) == protocol["input_sha256"]
    raw = {row["id"]: row for row in rows(source) if row["scope"] == "known800"}
    assert len(raw) == 800
    calibration = [raw[value] for value in protocol["calibration"]["ids"]]
    test = [raw[value] for value in protocol["test"]["ids"]]
    assert {row["group_id"] for row in calibration}.isdisjoint(row["group_id"] for row in test)
    assert len({row["group_id"] for row in calibration}) == 10
    assert len({row["group_id"] for row in test}) == 30
    logits = torch.tensor([row["scores"] for row in calibration], dtype=torch.float64).log()
    labels = torch.tensor([row["candidate_ids"].index(row["target"]) for row in calibration])
    log_temperature = torch.tensor(math.log(result["temperature"]), dtype=torch.float64,
                                   requires_grad=True)
    loss = functional.cross_entropy(logits / log_temperature.exp(), labels)
    loss.backward()
    gradient = float(log_temperature.grad)
    assert abs(gradient) < 1e-8, gradient
    for value in (0.1, 1.0, result["temperature"] * 0.99, result["temperature"] * 1.01):
        assert float(functional.cross_entropy(logits / value, labels)) > float(loss.detach())
    maximum_difference, verified = 0.0, 0
    for scope, filename in (("calibration100", "calibration100-predictions.jsonl"),
                            ("test300", "test300-predictions.jsonl"),
                            ("old_stt100", "old_stt100-predictions.jsonl")):
        frozen = rows(DIRECTORY / filename)
        original = [raw[row["id"]] for row in frozen]
        original_logits = torch.tensor([row["scores"] for row in original], dtype=torch.float64).log()
        probabilities = functional.softmax(original_logits / result["temperature"], dim=-1)
        recorded = torch.tensor([row["scores"] for row in frozen], dtype=torch.float64)
        difference = float((probabilities - recorded).abs().max())
        assert difference < 1e-12
        maximum_difference = max(maximum_difference, difference)
        for reference, current, winner in zip(original, frozen, original_logits.argmax(-1).tolist()):
            assert current["candidate_ids"] == reference["candidate_ids"]
            assert current["predicted"] == reference["predicted"] == current["candidate_ids"][winner]
            assert current["correct"] == (current["predicted"] == current["target"])
            assert len(current["candidate_ids"]) == 20
        covered = [i for i, row in enumerate(original) if row["target_in_top20"]]
        targets = torch.tensor([original[i]["candidate_ids"].index(original[i]["target"]) for i in covered])
        for label, temperature in (("original", 1.0), ("fitted", result["temperature"])):
            measured = float(functional.cross_entropy(original_logits[covered] / temperature, targets))
            assert abs(measured - result["scopes"][scope][label]["conditional_nll"]) < 1e-10
        verified += len(frozen)
    audit = {"passed": True, "temperature": result["temperature"],
             "calibration_gradient_log_temperature": gradient,
             "probabilities_audited": verified * 20, "predictions_audited": verified,
             "maximum_torch_probability_difference": maximum_difference,
             "calibration_test_group_overlap": 0, "torch": torch.__version__,
             "source_sha256": sha(Path(__file__)), "report_sha256": sha(DIRECTORY / "temperature.json")}
    (DIRECTORY / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    run()
