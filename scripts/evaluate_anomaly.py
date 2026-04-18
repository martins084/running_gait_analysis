"""
Evaluate anomaly reconstruction model on a chosen split.

Outputs:
- metrics.json
- thresholds.json
- per_sample_scores.csv
- plots/error_histogram.png (if matplotlib available)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ric_dataset import (
    RICAnomalyDataset,
    SequenceSpec,
    collate_ric_anomaly,
    compute_max_feature_dim,
)
from models.gait_classifier import GaitAnomalyDetector


def _load_cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _quantile_threshold(scores: np.ndarray, q: float) -> float:
    q = min(max(float(q), 0.0), 1.0)
    return float(np.quantile(scores, q))


@torch.no_grad()
def _collect_scores(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    out_rows = []
    all_scores = []
    all_labels = []
    for batch in loader:
        x = batch["x"].to(device)
        decoded, _ = model(x)
        scores = ((decoded - x) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()
        labels = batch["is_injured"].cpu().numpy()
        for i in range(len(scores)):
            out_rows.append(
                {
                    "subject_id": batch["subject_id"][i],
                    "session_id": batch["session_id"][i],
                    "mode": batch["mode"][i],
                    "is_injured": int(labels[i]),
                    "recon_error": float(scores[i]),
                }
            )
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())
    return {"rows": out_rows, "scores": np.array(all_scores), "labels": np.array(all_labels)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "session_id", "mode", "is_injured", "recon_error"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anomaly model reconstruction errors.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "anomaly_train_v1.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    seq = SequenceSpec(
        seq_len=int(cfg["data"]["seq_len"]),
        prefer_mode=str(cfg["data"].get("prefer_mode", "run")),
        normalize=str(cfg["data"].get("normalize", "zscore")),
        train_random_window=False,
    )

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Single CPU load: need `input_size` before the dataset (variable marker counts per session).
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "input_size" in ckpt:
        input_size = int(ckpt["input_size"])
    else:
        input_size = compute_max_feature_dim(
            ROOT / cfg["data"]["session_split_csv"],
            ROOT / cfg["data"]["json_root"],
            seq,
        )

    dataset = RICAnomalyDataset(
        session_split_csv=ROOT / cfg["data"]["session_split_csv"],
        json_root=ROOT / cfg["data"]["json_root"],
        split=args.split,
        seq=seq,
        include_injured=True,
        seed=int(cfg["experiment"]["seed"]),
        feature_dim=input_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_ric_anomaly,
    )

    hidden_size = int(ckpt.get("hidden_size", cfg["train"].get("hidden_size", 128)))
    model = GaitAnomalyDetector(input_size=input_size, hidden_size=hidden_size).to(device)
    model.load_state_dict(ckpt["model_state"])

    coll = _collect_scores(model, loader, device)
    scores = coll["scores"]
    labels = coll["labels"]
    rows = coll["rows"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(args.output_dir / "per_sample_scores.csv", rows)

    thr = _quantile_threshold(scores, args.threshold_quantile)
    y_pred = (scores >= thr).astype(int)
    acc = float((y_pred == labels).mean()) if len(labels) else float("nan")
    positive_rate = float(y_pred.mean()) if len(y_pred) else float("nan")

    metrics = {
        "split": args.split,
        "num_samples": int(len(scores)),
        "mean_recon_error": float(scores.mean()) if len(scores) else float("nan"),
        "std_recon_error": float(scores.std()) if len(scores) else float("nan"),
        "threshold_quantile": float(args.threshold_quantile),
        "threshold_value": float(thr),
        "threshold_accuracy": acc,
        "predicted_positive_rate": positive_rate,
    }
    if len(np.unique(labels)) >= 2:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
        metrics["auprc"] = float(average_precision_score(labels, scores))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    thresholds = {
        "q90": _quantile_threshold(scores, 0.90),
        "q95": _quantile_threshold(scores, 0.95),
        "q975": _quantile_threshold(scores, 0.975),
        "q99": _quantile_threshold(scores, 0.99),
    }

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        plots_dir = args.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(scores, bins=40, alpha=0.75)
        ax.axvline(thr, color="red", linestyle="--", label=f"q{int(args.threshold_quantile * 100)}={thr:.4f}")
        ax.set_title(f"Reconstruction Error Histogram ({args.split})")
        ax.set_xlabel("Reconstruction Error")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "error_histogram.png", dpi=130)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot skip: {exc}", file=sys.stderr)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

