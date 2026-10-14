"""Plot the completed run's actual losses and per-profile validation accuracy."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path


def main(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    result = json.loads((args.run / "result.json").read_text())
    training = [json.loads(line) for line in (args.run / "training.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in (args.run / "validation.jsonl").read_text().splitlines()]
    if len(training) != result["steps"] or validation[-1]["step"] != result["steps"]:
        raise ValueError("Plot requires complete training and validation ledgers")
    window, smoothed = deque(maxlen=50), []
    for row in training:
        window.append(row["loss"])
        smoothed.append(sum(window) / len(window))
    steps = [row["step"] for row in training]
    checks = [row["step"] for row in validation]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    axes[0].semilogy(steps, [max(row["loss"], 1e-8) for row in training], color="#4682b4",
                     alpha=0.2, linewidth=0.6, label="Train CE: отдельные batch")
    axes[0].semilogy(steps, [max(value, 1e-8) for value in smoothed], color="#246394",
                     linewidth=1.8, label="Train CE: среднее за 50 шагов")
    axes[0].semilogy(checks, [row["selection_loss"] for row in validation], color="#c35c21",
                     marker="o", linewidth=1.8, label="Validation NLL: равный вес групп")
    axes[0].set(xlabel="Шаг оптимизатора", ylabel="Loss: логарифмическая шкала, минимум 1e-8", title="Ошибка обучения и выбор весов")
    profiles = (("launcher-basic-v1", "Простые", "#267a4e"),
                ("launcher-v1", "Сложные", "#9658a5"),
                ("launcher-asr-v1", "ASR-тексты", "#246394"))
    for profile, label, color in profiles:
        axes[1].plot(checks, [row["metrics"][profile]["accuracy"] for row in validation],
                     label=label, color=color, marker="o", linewidth=1.8)
    axes[1].set(xlabel="Шаг оптимизатора", ylabel="Правильно выбранные ID", ylim=(0, 1.025),
                title="Synthetic validation: по 50 запросов")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for axis in axes:
        prefix_step = result["plan"].get("clean_prefix_rows", 0) // 16
        if prefix_step:
            axis.axvline(prefix_step, color="#777777", linestyle=":", linewidth=1.2, label="Конец чистого старта")
        axis.axvline(result["best_step"], color="#222222", linestyle="--", linewidth=1.2,
                     label=f"Выбранные веса: шаг {result['best_step']}")
        axis.grid(alpha=0.15)
        legend_location = "upper right" if axis is axes[1] else "lower right"
        if result["plan"].get("model_family") == "e5":
            legend_location = "lower right"
        axis.legend(fontsize=8, loc=legend_location)
    reason = "3 проверки без улучшения" if result["stop_reason"] == "validation_patience" else "лимит одного прохода"
    fig.suptitle(f"{args.model_label}: {result['steps']} шагов, остановка — {reason}", fontsize=14)
    if result["plan"].get("model_family") == "e5":
        caption = "Train CE: закреплённые 20 карточек. Validation NLL: полный каталог 177, равный вес групп. Третий каталог здесь не используется."
    else:
        caption = "Train CE: batch по 16 запросов. Validation NLL: достижимые цели, равный вес групп. Третий каталог здесь не используется."
    fig.supxlabel(caption, fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)
    print(json.dumps({"plot": str(args.output.resolve()), "matplotlib": matplotlib.__version__,
                      "training_steps": len(training), "validation_checks": len(validation)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/kaggle/laya-learning-curves.png"))
    parser.add_argument("--model-label", default="Laya")
    main(parser.parse_args())
