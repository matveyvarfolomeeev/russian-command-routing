"""Verify inference equivalence and source integrity for file-backed CPU weights."""

import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file

    from scripts.laya_cpu_weights import cpu_weights, sha
except ImportError as error:
    raise unittest.SkipTest("Requires the model environment and model worktree on PYTHONPATH") from error


class CpuWeightTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[1] / "artifacts/kaggle"
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="weight-test-", dir=self.scratch)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(self.scratch.resolve()))
        self.addCleanup(self.temporary.cleanup)

    def test_fp32_linear_inference_matches_copied_weights(self):
        expected = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7
        source = self.root / "source.safetensors"
        save_file({"weight": expected}, str(source))
        original_sha = sha(source)
        weights, metadata = cpu_weights(source, original_sha, self.root / "cache")
        linear = torch.nn.Linear(4, 3, bias=False)
        linear.weight = torch.nn.Parameter(weights.get_tensor("weight"), requires_grad=False)
        query = torch.tensor([[0.5, -1, 2, 3]], dtype=torch.float32)
        with torch.inference_mode():
            self.assertTrue(torch.equal(linear(query), torch.nn.functional.linear(query, expected)))
        self.assertEqual(metadata["storage"], "read-only file mapping")
        self.assertEqual(sha(source), original_sha)

    def test_half_widening_is_exact_and_preserves_existing_fp32(self):
        source = self.root / "source.safetensors"
        values = {"half": torch.tensor([[-3.25, 0.125, 0, 65504]], dtype=torch.float16),
                  "single": torch.tensor([1.1234567], dtype=torch.float32)}
        save_file(values, str(source))
        original_sha = sha(source)
        weights, metadata = cpu_weights(source, original_sha, self.root / "cache")
        for name, expected in values.items():
            self.assertEqual(weights.get_tensor(name).dtype, torch.float32)
            self.assertTrue(torch.equal(weights.get_tensor(name), expected.float()))
        cached, repeated = cpu_weights(source, original_sha, self.root / "cache")
        self.assertTrue(torch.equal(cached.get_tensor("half"), values["half"].float()))
        self.assertEqual(repeated, metadata)
        self.assertEqual(sha(source), original_sha)

    def test_changed_cache_cannot_be_used_for_inference(self):
        source = self.root / "source.safetensors"
        save_file({"weight": torch.ones(2, dtype=torch.float16)}, str(source))
        original_sha = sha(source)
        weights, _ = cpu_weights(source, original_sha, self.root / "cache")
        # Drop the file view before changing the derived test fixture on Windows.
        weights.mapping.close()
        derived = self.root / "cache" / original_sha / "model.safetensors"
        with derived.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "cached FP32 weights changed"):
            cpu_weights(source, original_sha, self.root / "cache")

    def test_bfloat_widening_matches_torch_bit_for_bit(self):
        source = self.root / "bfloat.safetensors"
        values = torch.tensor([-3.25, 0.125, -0.0, 1e-30, 1e30], dtype=torch.bfloat16)
        save_file({"weight": values}, str(source))
        original_sha = sha(source)
        weights, _ = cpu_weights(source, original_sha, self.root / "cache")
        self.assertTrue(torch.equal(weights.get_tensor("weight").view(torch.int32),
                                    values.float().view(torch.int32)))
        self.assertEqual(sha(source), original_sha)

    def test_truncated_source_is_rejected_before_mapping(self):
        source = self.root / "truncated.safetensors"
        save_file({"weight": torch.ones(2)}, str(source))
        with source.open("r+b") as stream:
            stream.truncate(source.stat().st_size - 1)
        with self.assertRaisesRegex(ValueError, "data length mismatch"):
            cpu_weights(source, sha(source), self.root / "cache")


if __name__ == "__main__":
    unittest.main()
