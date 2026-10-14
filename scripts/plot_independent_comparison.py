"""Plot audited clean/ASR results with separate validation and test denominators."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = ROOT / "artifacts/kaggle/independent-v1"
    report = json.loads((root / "comparison.json").read_text())
    if report["audit"] != "passed":
        raise ValueError("Only audited results can be plotted")
    names = [("E5", "e5"), ("Nano\nисходный", "original-nano"), ("Nano\nmixed", "mixed-nano"),
             ("Nano\nclean-first", "clean-nano"), ("Laya\nисходная", "original-laya"),
             ("Laya\nдообученная", "trained-laya"), ("Kev\nисходный", "original-kev"),
             ("Kev\nдообученный", "trained-kev")]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained", sharey=True)
    for axis, split, total in zip(axes, ("validation", "test"), (50, 150), strict=True):
        for offset, representation, label, color in ((-0.18, "clean", "Чистый текст", "#23765b"),
                                                     (0.18, "asr", "После ASR", "#4268b0")):
            counts = [(report["e5"] if key == "e5" else report["models"][key]["metrics"])
                      [f"independent/{representation}/{split}"] for _, key in names]
            if any(r["rows"] != total or r["accuracy"] != r["correct"] / total for r in counts):
                raise ValueError("Inconsistent displayed denominator")
            bars = axis.bar([i + offset for i in range(len(names))], [r["accuracy"] * 100 for r in counts],
                            0.34, label=label, color=color)
            axis.bar_label(bars, labels=[str(r["correct"]) for r in counts], padding=3, fontsize=9)
        axis.set(title=f"{split}: правильных ID из {total}", xticks=range(len(names)),
                 xticklabels=[label for label, _ in names], ylim=(0, 110), yticks=range(0, 101, 20))
        axis.tick_params(axis="x", labelsize=9)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Точность, %")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=2)
    fig.supxlabel("200 новых команд: clean и ASR — связанные версии. Фиксированные веса и E5 top-20; все ошибки включены.", fontsize=10)
    for suffix in ("png", "svg"):
        fig.savefig(root / f"comparison.{suffix}", dpi=160)
    plt.close(fig)
    print(json.dumps({"plot": str(root / "comparison.png")}))


if __name__ == "__main__":
    main()
