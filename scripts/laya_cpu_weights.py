"""Keep CPU inference weights file-backed; widen original half weights on disk."""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import struct
import warnings
from pathlib import Path

from jev_launcher.local_weights import CpuSafeFile


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ReadOnlyCpuWeights(CpuSafeFile):
    def __init__(self, filename):
        # safe_open's Windows map reserves commit for the entire file, even for
        # metadata validation. Validate the pinned format by streaming its header.
        with open(filename, "rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated safetensors header")
            size = struct.unpack("<Q", prefix)[0]
            file_size = Path(filename).stat().st_size
            if size > 100_000_000 or size > file_size - 8:
                raise ValueError("Invalid safetensors header size")
            self.records = json.loads(stream.read(size))
        self.filename, self.data_start = filename, size + 8
        self.file_metadata = self.records.pop("__metadata__", None)
        widths = {"F32": 4, "F16": 2, "BF16": 2}
        cursor = 0
        for record in sorted(self.records.values(), key=lambda r: r["data_offsets"]):
            shape, kind = record["shape"], record["dtype"]
            start, end = record["data_offsets"]
            if kind not in widths or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError("Invalid floating-point tensor descriptor")
            if start != cursor or end - start != math.prod(shape) * widths[kind]:
                raise ValueError("Invalid safetensors data offsets")
            cursor = end
        if self.data_start + cursor != file_size:
            raise ValueError("Safetensors data length mismatch")
        with open(filename, "rb") as stream:
            self.mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)

    def get_tensor(self, key):
        import torch
        from safetensors.torch import _getdtype

        record = self.records[key]
        count = math.prod(record["shape"])
        dtype = _getdtype(record["dtype"])
        if not count:
            return torch.empty(record["shape"], dtype=dtype)
        # Evaluation never writes these buffers. Torch retains the Python buffer owner.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given buffer is not writable")
            tensor = torch.frombuffer(self.mapping, dtype=dtype, count=count,
                                      offset=self.data_start + record["data_offsets"][0])
        return tensor.reshape(record["shape"])


def cpu_weights(filename, source_sha, cache):
    original = ReadOnlyCpuWeights(filename)
    kinds = {record["dtype"] for record in original.records.values()}
    if kinds == {"F32"}:
        return original, {"storage": "read-only file mapping", "source_sha256": source_sha}
    if not kinds <= {"F16", "BF16", "F32"}:
        raise ValueError(f"Unsupported floating-point weight dtypes: {sorted(kinds)}")
    destination = cache / source_sha / "model.safetensors"
    manifest_path = destination.with_suffix(".json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_sha256"] != source_sha or sha(destination) != manifest["derived_sha256"]:
            raise ValueError("The cached FP32 weights changed")
    else:
        import numpy as np

        destination.parent.mkdir(parents=True, exist_ok=True)
        header, offset = {}, 0
        for key, record in original.records.items():
            size = math.prod(record["shape"]) * 4
            header[key] = {"dtype": "F32", "shape": record["shape"], "data_offsets": [offset, offset + size]}
            offset += size
        if original.file_metadata is not None:
            header["__metadata__"] = original.file_metadata
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        partial = destination.with_suffix(".part")
        with open(filename, "rb") as source, partial.open("wb") as target:
            target.write(struct.pack("<Q", len(encoded)))
            target.write(encoded)
            for record in original.records.values():
                start, end = record["data_offsets"]
                source.seek(original.data_start + start)
                remaining = end - start
                while remaining:
                    block = source.read(min(4 * 1024 * 1024, remaining))
                    if not block:
                        raise ValueError("The original weight file became truncated")
                    remaining -= len(block)
                    if record["dtype"] == "F16":
                        target.write(np.frombuffer(block, dtype="<f2").astype("<f4").tobytes())
                    elif record["dtype"] == "BF16":
                        # BF16 is the upper 16 bits of IEEE float32: widening is exact.
                        widened = np.frombuffer(block, dtype="<u2").astype("<u4") << 16
                        target.write(widened.tobytes())
                    else:
                        target.write(block)
        partial.replace(destination)
        manifest = {"source_sha256": source_sha, "derived_sha256": sha(destination),
                    "conversion": "Exact F16/BF16 to F32 widening; original F32 bytes retained"}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return ReadOnlyCpuWeights(destination), {"storage": "read-only file mapping of exact FP32 widening", **manifest}
