"""
Train a supervised binary injury classifier (GaitInjuryClassifier) on RIC data.

This script uses the full labelled dataset (healthy + injured) and optimises
BCEWithLogitsLoss with pos_weight to handle class imbalance.

Because labels exist for 1,798 subjects, a supervised classifier will reach
higher AUROC than the unsupervised anomaly autoencoder.

Usage
-----
python scripts/train_classifier.py --config config/classifier_v1.yaml

Output (under results/<run_id>/):
  train_summary.json        best AUROC + AUPRC
  logs/epoch_metrics.csv    per-epoch train/val metrics
  logs/run_metadata.json    config hash, git commit, hyperparams
  checkpoints/best.pt       best checkpoint (by val AUROC)
  checkpoints/last.pt       most recent checkpoint
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ric_dataset import (
    RICClassifierDataset,
    SequenceSpec,
    _motion_stats_extra_channels,
    collate_ric_classifier,
    compute_max_feature_dim,
)
from models.gait_classifier import build_classifier_model


# ---------------------------------------------------------------------------
# Config + reproducibility
# ---------------------------------------------------------------------------

def _load_cfg(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")
    return cfg


def _cfg_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested CUDA but it is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _git_commit() -> str:
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True)
        return p.stdout.strip()
    except Exception:
        return ""


def _ensure_dirs(cfg: dict, run_id: str) -> dict[str, Path]:
    out = cfg["output"]
    dirs = {
        "checkpoints": ROOT / out["checkpoints_dir"] / run_id,
        "logs":        ROOT / out["logs_dir"]        / run_id,
        "results":     ROOT / out["results_dir"]     / run_id,
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _build_seq(cfg: dict) -> SequenceSpec:
    d = cfg["data"]
    fe = str(d.get("feature_engineering", "none")).lower()
    return SequenceSpec(
        seq_len=int(d["seq_len"]),
        prefer_mode=str(d.get("prefer_mode", "run")),
        normalize=str(d.get("normalize", "zscore")),
        train_random_window=bool(d.get("train_random_window", True)),
        feature_engineering=fe,
        lp_filter_hz=float(d.get("lp_filter_hz", 0.0)),
        lp_filter_source_hz=float(d.get("lp_filter_source_hz", 120.0)),
        aug_mirror=bool(d.get("aug_mirror", False)),
        aug_mirror_prob=float(d.get("aug_mirror_prob", 0.5)),
        aug_noise_sigma=float(d.get("aug_noise_sigma", 0.0)),
        aug_time_warp=bool(d.get("aug_time_warp", False)),
    )


def _build_loaders(
    cfg: dict, seed: int
) -> tuple[DataLoader, DataLoader, int, dict]:
    d   = cfg["data"]
    t   = cfg["train"]
    seq = _build_seq(cfg)
    fe  = seq.feature_engineering.lower()
    label_col = str(d.get("label_col", "is_injured"))

    base_dim = compute_max_feature_dim(
        ROOT / d["session_split_csv"], ROOT / d["json_root"], seq, split_filter="train"
    )
    feat_dim = int(base_dim + (_motion_stats_extra_channels() if fe == "motion_stats" else 0))

    def _worker_init(wid: int) -> None:
        s = seed + wid
        random.seed(s); np.random.seed(s); torch.manual_seed(s)

    gen = torch.Generator()
    gen.manual_seed(seed)

    train_ds = RICClassifierDataset(
        session_split_csv=ROOT / d["session_split_csv"],
        json_root=ROOT / d["json_root"],
        split="train",
        seq=seq,
        seed=seed,
        feature_dim=feat_dim,
        label_col=label_col,
    )
    val_ds = RICClassifierDataset(
        session_split_csv=ROOT / d["session_split_csv"],
        json_root=ROOT / d["json_root"],
        split="val",
        seq=seq,
        seed=seed,
        feature_dim=feat_dim,
        label_col=label_col,
    )

    # Class-balanced oversampling so each batch has roughly equal healthy/injured.
    n_healthy, n_injured = train_ds.class_counts()
    class_meta = {"n_healthy": n_healthy, "n_injured": n_injured}
    if n_injured > 0 and n_healthy > 0 and bool(t.get("oversample", True)):
        w0 = 1.0 / n_healthy
        w1 = 1.0 / n_injured
        labels = [int(r.get(label_col, 0)) for r in train_ds.rows]
        weights = [w0 if y == 0 else w1 for y in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=int(t["batch_size"]),
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=int(t["num_workers"]),
        collate_fn=collate_ric_classifier,
        drop_last=False,
        worker_init_fn=_worker_init,
        generator=gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(t["batch_size"]),
        shuffle=False,
        num_workers=int(t["num_workers"]),
        collate_fn=collate_ric_classifier,
        drop_last=False,
        worker_init_fn=_worker_init,
        generator=gen,
    )
    return train_loader, val_loader, feat_dim, class_meta


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def _build_scheduler(optimizer, scheduler_cfg: dict, n_epochs: int):
    if not bool(scheduler_cfg.get("enabled", True)):
        return None, "none"
    stype   = str(scheduler_cfg.get("type", "cosine")).lower()
    min_lr  = float(scheduler_cfg.get("min_lr", 1e-6))
    warmup  = int(scheduler_cfg.get("warmup_epochs", 5))

    if stype == "cosine":
        main_ep = max(1, n_epochs - warmup)
        cosine  = CosineAnnealingLR(optimizer, T_max=main_ep, eta_min=min_lr)
        if warmup > 0:
            wu    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
            sched = SequentialLR(optimizer, schedulers=[wu, cosine], milestones=[warmup])
        else:
            sched = cosine
        return sched, "cosine"

    sched = ReduceLROnPlateau(
        optimizer,
        mode="max",   # maximise AUROC
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 5)),
        min_lr=min_lr,
    )
    return sched, "plateau"


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: torch.Tensor | None,
    grad_clip: float,
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.detach()) * len(x)
        n += len(x)
    return total_loss / n if n else float("nan")


@torch.no_grad()
def _val_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n = 0
    y_true: list[float] = []
    y_score: list[float] = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss) * len(x)
        n += len(x)
        y_true.extend(y.cpu().numpy().tolist())
        y_score.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    out = {"val_loss": total_loss / n if n else float("nan")}
    if len(set(int(v) for v in y_true)) >= 2:
        out["val_auroc"] = float(roc_auc_score(y_true, y_score))
        out["val_auprc"] = float(average_precision_score(y_true, y_score))
    else:
        out["val_auroc"] = float("nan")
        out["val_auprc"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train supervised injury classifier on RIC data.")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config" / "classifier_v1.yaml")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    cfg  = _load_cfg(args.config)
    seed = int(cfg["experiment"]["seed"])
    _seed_all(seed)

    cfg_hash  = _cfg_hash(args.config)
    ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix    = str(cfg["experiment"].get("run_id_prefix", "clf"))
    run_id    = args.run_id or f"{prefix}-{ts}-{cfg_hash}"
    dirs      = _ensure_dirs(cfg, run_id)

    tcfg   = cfg["train"]
    mcfg   = cfg.get("model", {})
    device = _pick_device(str(tcfg.get("device", "auto")).lower())

    train_loader, val_loader, input_size, class_meta = _build_loaders(cfg, seed)
    n_epochs = int(args.epochs) if args.epochs > 0 else int(tcfg["epochs"])

    model = build_classifier_model(
        input_size=input_size,
        hidden_size=int(mcfg.get("hidden_size", 128)),
        num_layers=int(mcfg.get("num_layers", 2)),
        dropout=float(mcfg.get("dropout", 0.3)),
    ).to(device)

    # pos_weight = n_healthy / n_injured  (if not using oversampling)
    pw: torch.Tensor | None = None
    n0, n1 = class_meta["n_healthy"], class_meta["n_injured"]
    if not bool(tcfg.get("oversample", True)) and n0 > 0 and n1 > 0:
        pw = torch.tensor([n0 / n1], dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )

    start_epoch    = 1
    best_auroc     = float("-inf")
    best_auprc     = float("-inf")

    if args.resume_checkpoint is not None:
        state = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_auroc  = float(state.get("best_val_auroc", best_auroc))

    scheduler, sched_type = _build_scheduler(optimizer, tcfg.get("scheduler", {}), n_epochs)

    early_cfg  = tcfg.get("early_stopping", {})
    early_pat  = int(early_cfg.get("patience", 15))
    no_improve = 0

    grad_clip = float(tcfg.get("grad_clip_norm", 1.0))
    epoch_rows: list[dict] = []

    run_meta = {
        "run_id": run_id, "config_hash": cfg_hash, "seed": seed,
        "device": str(device), "git_commit": _git_commit(),
        "started_utc": ts, "class_balance_train": class_meta,
        "input_size": input_size, "model": mcfg, "train": tcfg,
    }
    (dirs["logs"] / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    for epoch in range(start_epoch, n_epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, device, pw, grad_clip)
        val_m      = _val_epoch(model, val_loader, device)
        val_loss   = float(val_m["val_loss"])
        val_auroc  = float(val_m["val_auroc"])
        val_auprc  = float(val_m.get("val_auprc", float("nan")))

        epoch_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 8),
            "val_loss":   round(val_loss,   8),
            "val_auroc":  "" if np.isnan(val_auroc) else round(val_auroc, 8),
            "val_auprc":  "" if np.isnan(val_auprc) else round(val_auprc, 8),
        })

        ckpt = {
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_auroc": best_auroc, "run_id": run_id,
            "config_hash": cfg_hash, "input_size": input_size,
        }
        torch.save(ckpt, dirs["checkpoints"] / "last.pt")

        improved = not np.isnan(val_auroc) and val_auroc > best_auroc
        if improved:
            best_auroc = val_auroc
            best_auprc = val_auprc
            torch.save(ckpt, dirs["checkpoints"] / "best.pt")
            no_improve = 0
        else:
            no_improve += 1

        if scheduler is not None:
            if sched_type == "plateau":
                scheduler.step(val_auroc if not np.isnan(val_auroc) else 0.0)
            else:
                scheduler.step()

        auroc_str = "n/a" if np.isnan(val_auroc) else f"{val_auroc:.4f}"
        auprc_str = "n/a" if np.isnan(val_auprc) else f"{val_auprc:.4f}"
        print(
            f"Epoch {epoch:03d}/{n_epochs}: "
            f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
            f"val_auroc={auroc_str}  val_auprc={auprc_str}"
        )
        if bool(early_cfg.get("enabled", True)) and no_improve >= early_pat:
            print(f"Early stopping at epoch {epoch} (no AUROC improvement for {early_pat} epochs).")
            break

    # Write epoch CSV
    with open(dirs["logs"] / "epoch_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_auroc", "val_auprc"])
        w.writeheader()
        w.writerows(epoch_rows)

    summary = {
        "run_id": run_id, "best_val_auroc": best_auroc,
        "best_val_auprc": best_auprc, "epochs_completed": len(epoch_rows),
        "checkpoints_dir": str(dirs["checkpoints"]),
    }
    (dirs["results"] / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nTraining complete.  Best val AUROC: {best_auroc:.4f}  |  Run: {run_id}")


if __name__ == "__main__":
    main()
