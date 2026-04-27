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
from collections import defaultdict
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
    _motion_stats_extra_channels,
    collate_ric_anomaly,
    compute_max_feature_dim,
)
from models.gait_classifier import build_anomaly_model


def _load_cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _quantile_threshold(scores: np.ndarray, q: float) -> float:
    q = min(max(float(q), 0.0), 1.0)
    return float(np.quantile(scores, q))


def _load_threshold_from_file(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"Threshold file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "threshold" in payload:
            return float(payload["threshold"])
        if "val_at_selected_threshold" in payload and isinstance(payload["val_at_selected_threshold"], dict):
            if "threshold" in payload["val_at_selected_threshold"]:
                return float(payload["val_at_selected_threshold"]["threshold"])
    raise ValueError("Threshold file must contain `threshold` or `val_at_selected_threshold.threshold`.")


@torch.no_grad()
def _collect_scores(model: torch.nn.Module, loader: DataLoader, device: torch.device, label_col: str) -> dict:
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
                    label_col: int(labels[i]),
                    "recon_error": float(scores[i]),
                }
            )
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())
    return {"rows": out_rows, "scores": np.array(all_scores), "labels": np.array(all_labels)}


def _write_csv(path: Path, rows: list[dict], label_col: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "session_id", "mode", label_col, "recon_error"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _classification_metrics(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    y_pred = (scores >= thr).astype(int)
    out = {
        "num_samples": int(len(scores)),
        "mean_recon_error": float(scores.mean()) if len(scores) else float("nan"),
        "std_recon_error": float(scores.std()) if len(scores) else float("nan"),
        "threshold_accuracy": float((y_pred == y_true).mean()) if len(y_true) else float("nan"),
        "predicted_positive_rate": float(y_pred.mean()) if len(y_pred) else float("nan"),
    }
    if len(np.unique(y_true)) >= 2:
        out["auroc"] = float(roc_auc_score(y_true, scores))
        out["auprc"] = float(average_precision_score(y_true, scores))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def _aggregate_subject_p90(rows: list[dict], label_col: str) -> tuple[np.ndarray, np.ndarray]:
    by_subject: dict[str, list[float]] = defaultdict(list)
    subj_label: dict[str, int] = {}
    for r in rows:
        sid = str(r["subject_id"])
        by_subject[sid].append(float(r["recon_error"]))
        subj_label[sid] = max(subj_label.get(sid, 0), int(r[label_col]))
    subjects = sorted(by_subject.keys())
    subj_scores = np.array([float(np.quantile(by_subject[sid], 0.90)) for sid in subjects], dtype=np.float64)
    subj_labels = np.array([int(subj_label[sid]) for sid in subjects], dtype=np.int64)
    return subj_labels, subj_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anomaly model reconstruction errors.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "anomaly_train_v1.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--threshold-file", type=Path, default=None)
    parser.add_argument(
        "--allow-test-quantile",
        action="store_true",
        help="Allow split-local quantile thresholding on test split (diagnostics only).",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    label_col = str(cfg["data"].get("label_col", "is_injured"))
    fe = str(cfg["data"].get("feature_engineering", "none")).lower()
    seq = SequenceSpec(
        seq_len=int(cfg["data"]["seq_len"]),
        prefer_mode=str(cfg["data"].get("prefer_mode", "run")),
        normalize=str(cfg["data"].get("normalize", "zscore")),
        train_random_window=False,
        feature_engineering=fe,
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
        base = compute_max_feature_dim(
            ROOT / cfg["data"]["session_split_csv"],
            ROOT / cfg["data"]["json_root"],
            seq,
            split_filter="train",
        )
        input_size = int(base + (_motion_stats_extra_channels() if fe == "motion_stats" else 0))

    dataset = RICAnomalyDataset(
        session_split_csv=ROOT / cfg["data"]["session_split_csv"],
        json_root=ROOT / cfg["data"]["json_root"],
        split=args.split,
        seq=seq,
        include_injured=True,
        seed=int(cfg["experiment"]["seed"]),
        feature_dim=input_size,
        label_col=label_col,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_ric_anomaly,
    )

    hidden_size = int(ckpt.get("hidden_size", cfg["train"].get("hidden_size", 128)))
    mcfg = cfg.get("model", {})
    model_variant = str(ckpt.get("model_variant", mcfg.get("variant", "baseline")))
    model_num_layers = int(ckpt.get("model_num_layers", mcfg.get("num_layers", 2)))
    model_dropout = float(ckpt.get("model_dropout", mcfg.get("dropout", 0.2)))
    model_bidirectional = bool(ckpt.get("model_bidirectional", mcfg.get("bidirectional", True)))
    model = build_anomaly_model(
        variant=model_variant,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=model_num_layers,
        dropout=model_dropout,
        bidirectional=model_bidirectional,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    coll = _collect_scores(model, loader, device, label_col=label_col)
    scores = coll["scores"]
    labels = coll["labels"]
    rows = coll["rows"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(args.output_dir / "per_sample_scores.csv", rows, label_col=label_col)

    threshold_source = "quantile_current_split"
    if args.fixed_threshold is not None and args.threshold_file is not None:
        raise ValueError("Use only one of --fixed-threshold or --threshold-file.")
    if args.fixed_threshold is not None:
        thr = float(args.fixed_threshold)
        threshold_source = "fixed_argument"
    elif args.threshold_file is not None:
        thr = _load_threshold_from_file(args.threshold_file)
        threshold_source = f"file:{args.threshold_file}"
    else:
        if args.split == "test" and not args.allow_test_quantile:
            raise ValueError(
                "Refusing split-local quantile threshold on test split. "
                "Provide --fixed-threshold or --threshold-file. "
                "If this is only for diagnostics, pass --allow-test-quantile."
            )
        thr = _quantile_threshold(scores, args.threshold_quantile)
        if args.split == "test":
            threshold_source = "quantile_test_diagnostic_override"

    session_metrics = _classification_metrics(labels, scores, thr)
    subj_labels, subj_scores = _aggregate_subject_p90(rows, label_col=label_col)
    subject_metrics = _classification_metrics(subj_labels, subj_scores, thr)
    diagnostic_only = bool(args.split == "test" and threshold_source == "quantile_test_diagnostic_override")
    reporting_warning = (
        "Diagnostic-only threshold on test split; do not use for final reporting."
        if diagnostic_only
        else ""
    )

    metrics = {
        "split": args.split,
        "label_col": label_col,
        "threshold_quantile": float(args.threshold_quantile),
        "threshold_value": float(thr),
        "threshold_source": threshold_source,
        "diagnostic_only": diagnostic_only,
        "reporting_warning": reporting_warning,
        "primary_level": "subject_p90",
        "subject_level": subject_metrics,
        "session_level": session_metrics,
    }

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

