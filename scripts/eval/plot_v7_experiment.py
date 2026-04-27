#!/usr/bin/env python3
import ast
import json
import os
import subprocess
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_metrics(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _score_path(predictions_path: str) -> str:
    if predictions_path.endswith(".json"):
        return predictions_path[:-5] + ".score.json"
    return predictions_path + ".score.json"


def _ensure_score(dataset: str, predictions_path: str) -> Dict:
    score_path = _score_path(predictions_path)
    if os.path.exists(score_path):
        return _load_metrics(score_path)
    result = subprocess.run(
        [
            "python",
            "scripts/eval/score_predictions.py",
            "--dataset",
            dataset,
            "--predictions-path",
            predictions_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metrics = json.loads(result.stdout)
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def _parse_log_records(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{") or not line.endswith("}"):
                continue
            try:
                record = ast.literal_eval(line)
            except Exception:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _split_loss_records(records: List[Dict]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    train_points = []
    eval_points = []
    for record in records:
        epoch = float(record.get("epoch", 0.0))
        if "loss" in record:
            train_points.append((epoch, float(record["loss"])))
        if "eval_loss" in record:
            eval_points.append((epoch, float(record["eval_loss"])))
    return train_points, eval_points


def _plot_loss_curves() -> None:
    sft_records = _parse_log_records("logs/qwen_bfcl_sft_v7.log")
    dpo_records = _parse_log_records("logs/qwen_bfcl_sft_v7_dpo_mix_v1.log")
    sft_train, sft_eval = _split_loss_records(sft_records)
    dpo_train, dpo_eval = _split_loss_records(dpo_records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if sft_train:
        axes[0].plot([x for x, _ in sft_train], [y for _, y in sft_train], label="train_loss", color="#1f77b4", linewidth=1.5)
    if sft_eval:
        axes[0].plot([x for x, _ in sft_eval], [y for _, y in sft_eval], label="eval_loss", color="#d62728", linewidth=2.0)
    axes[0].set_title("v7 SFT Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if dpo_train:
        axes[1].plot([x for x, _ in dpo_train], [y for _, y in dpo_train], label="train_loss", color="#1f77b4", linewidth=1.5)
    if dpo_eval:
        axes[1].plot([x for x, _ in dpo_eval], [y for _, y in dpo_eval], label="eval_loss", color="#d62728", linewidth=2.0)
    axes[1].set_title("v7 DPO Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    os.makedirs("eval_results/figures", exist_ok=True)
    fig.savefig("eval_results/figures/v7_training_loss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_bars() -> None:
    baseline_when2call = _load_metrics("eval_results/baseline/qwen_when2call_full_vllm.score.json")
    baseline_bfcl = _load_metrics("eval_results/baseline/qwen_bfcl_full_vllm.score.json")
    v7_sft_when2call = _ensure_score("when2call", "eval_results/sft/qwen_bfcl_sft_v7_when2call.json")
    v7_sft_bfcl = _ensure_score("bfcl", "eval_results/sft/qwen_bfcl_sft_v7_bfcl.json")
    v7_dpo_when2call = _ensure_score("when2call", "eval_results/dpo/qwen_bfcl_sft_v7_dpo_mix_v1_when2call.json")
    v7_dpo_bfcl = _ensure_score("bfcl", "eval_results/dpo/qwen_bfcl_sft_v7_dpo_mix_v1_bfcl.json")

    labels = ["Baseline", "v7 SFT", "v7 DPO"]
    when2call_accuracy = [
        baseline_when2call["accuracy"],
        v7_sft_when2call["accuracy"],
        v7_dpo_when2call["accuracy"],
    ]
    when2call_macro_f1 = [
        baseline_when2call["macro_f1"],
        v7_sft_when2call["macro_f1"],
        v7_dpo_when2call["macro_f1"],
    ]
    bfcl_exact = [
        baseline_bfcl["exact_match"],
        v7_sft_bfcl["exact_match"],
        v7_dpo_bfcl["exact_match"],
    ]
    bfcl_tool_name = [
        baseline_bfcl["tool_name_accuracy"],
        v7_sft_bfcl["tool_name_accuracy"],
        v7_dpo_bfcl["tool_name_accuracy"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = range(len(labels))
    width = 0.35

    axes[0].bar([i - width / 2 for i in x], when2call_accuracy, width=width, label="Accuracy", color="#4c78a8")
    axes[0].bar([i + width / 2 for i in x], when2call_macro_f1, width=width, label="Macro F1", color="#f58518")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 0.8)
    axes[0].set_title("When2Call")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar([i - width / 2 for i in x], bfcl_exact, width=width, label="Exact Match", color="#54a24b")
    axes[1].bar([i + width / 2 for i in x], bfcl_tool_name, width=width, label="Tool Name Acc.", color="#e45756")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 0.8)
    axes[1].set_title("BFCL v3 Strict")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    os.makedirs("eval_results/figures", exist_ok=True)
    fig.savefig("eval_results/figures/v7_final_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _plot_loss_curves()
    _plot_metric_bars()
    print("Saved figures to eval_results/figures/")


if __name__ == "__main__":
    main()
