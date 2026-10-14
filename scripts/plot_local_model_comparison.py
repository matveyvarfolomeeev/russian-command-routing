"""Plot measured local profiles separately, retaining retrieval failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = [
        ("E5", "e5-reference-v1", "#64748b"),
        ("Nano: clean-first", "clean-first-v1", "#2563a6"),
        ("Laya: clean-first", "laya-clean-first-v1", "#238055"),
        ("Kev: исходный", "kev-baseline-v1", "#bc6b25"),
    ]
    if args.include_trained_kev:
        sources.append(("Kev: выбранный", "kev-clean-first-v1", "#8b4b9d"))
    results = [(name, json.loads((args.local / folder / "result.json").read_text()), color)
               for name, folder, color in sources]
    expected_hash = "2e2d77f81b28214e6143258406d310d403f0d76446da62f6f256f5c246e35f97"
    if any(result["input_sha256"] != expected_hash for _, result, _ in results):
        raise ValueError("Different evaluation inputs")
    profiles = ["launcher-basic-v1", "launcher-v1", "launcher-asr-v1"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", sharey=True)
    width = 0.8 / len(results)
    for axis, catalog, title in zip(axes, ("real", "synthetic"),
                                     ("Реальный каталог", "Вымышленный каталог"), strict=True):
        for index, (name, result, color) in enumerate(results):
            counts = [result["metrics"][f"{profile}/{catalog}"] for profile in profiles]
            if any(row["rows"] != 50 or row["accuracy"] != row["correct"] / 50 for row in counts):
                raise ValueError("Inconsistent displayed counts")
            positions = [i - 0.4 + width * (index + 0.5) for i in range(3)]
            bars = axis.bar(positions, [row["correct"] for row in counts], width * 0.9,
                            label=name, color=color)
            axis.bar_label(bars, fontsize=9, padding=3)
        axis.set(title=title, xticks=range(3), xticklabels=["Простые", "Сложные", "ASR-тексты"],
                 ylim=(0, 55), yticks=range(0, 51, 10))
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Правильных ID из 50 запросов")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, fontsize=10)
    fig.supxlabel("Одинаковые E5 top-20 и тексты; ошибки retrieval включены. Только качество: скорость здесь не проверяется.",
                  fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)
    print(json.dumps({"plot": str(args.output.resolve()), "models": [name for name, _, _ in results],
                      "input_sha256": expected_hash}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, default=ROOT / "artifacts/kaggle/local-validation")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/kaggle/local-model-comparison.png")
    parser.add_argument("--include-trained-kev", action="store_true")
    main(parser.parse_args())
