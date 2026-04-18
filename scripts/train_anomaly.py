"""
Train RIC anomaly model using existing `GaitAnomalyDetector`.

This is intentionally minimal and reproducible:
- deterministic seed
- config hash
- checkpoint best/last
- epoch metrics written to CSV/JSON
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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ric_dataset import RICAnomalyDataset, SequenceSpec, collate_ric_anomaly
from models.gait_classifier import GaitAnomalyDetector


def _load_cfg(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")
    return cfg


def _cfg_hash(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest()[:12]


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
    out_cfg = cfg["output"]
    ckpt = ROOT / out_cfg["checkpoints_dir"] / run_id
    logs = ROOT / out_cfg["logs_dir"] / run_id
    results = ROOT / out_cfg["results_dir"] / run_id
    for p in (ckpt, logs, results):
        p.mkdir(parents=True, exist_ok=True)
    return {"checkpoints": ckpt, "logs": logs, "results": results}


def _build_loaders(cfg: dict, seed: int) -> tuple[DataLoader, DataLoader, int]:
    d = cfg["data"]
    t = cfg["train"]
    seq = SequenceSpec(
        seq_len=int(d["seq_len"]),
        prefer_mode=str(d.get("prefer_mode", "run")),
        normalize=str(d.get("normalize", "zscore")),
        train_random_window=bool(d.get("train_random_window", True)),
    )

    train_ds = RICAnomalyDataset(
        session_split_csv=ROOT / d["session_split_csv"],
        json_root=ROOT / d["json_root"],
        split="train",
        seq=seq,
        include_injured=bool(d.get("train_include_injured", False)),
        seed=seed,
    )
    val_ds = RICAnomalyDataset(
        session_split_csv=ROOT / d["session_split_csv"],
        json_root=ROOT / d["json_root"],
        split="val",
        seq=seq,
        include_injured=bool(d.get("val_include_injured", True)),
        seed=seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(t["batch_size"]),
        shuffle=True,
        num_workers=int(t["num_workers"]),
        collate_fn=collate_ric_anomaly,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(t["batch_size"]),
        shuffle=False,
        num_workers=int(t["num_workers"]),
        collate_fn=collate_ric_anomaly,
        drop_last=False,
    )
    return train_loader, val_loader, train_ds.feature_dim


def _mse_per_sample(decoded: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Mean over sequence and feature dims -> one score per sample.
    return ((decoded - x) ** 2).mean(dim=(1, 2))


def _run_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        x = batch["x"].to(device)
        optimizer.zero_grad(set_to_none=True)
        decoded, _ = model(x)
        loss = ((decoded - x) ** 2).mean()
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def _run_epoch_val(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    losses = []
    y_true = []
    y_score = []
    for batch in loader:
        x = batch["x"].to(device)
        decoded, _ = model(x)
        per_sample = _mse_per_sample(decoded, x)
        losses.append(float(per_sample.mean().detach().cpu().item()))
        y_score.extend(per_sample.detach().cpu().numpy().tolist())
        y_true.extend(batch["is_injured"].cpu().numpy().tolist())

    out = {"val_loss": float(np.mean(losses)) if losses else float("nan")}
    if len(set(y_true)) >= 2:
        out["val_auroc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["val_auroc"] = float("nan")
    return out


def _write_epoch_metrics(path: Path, epoch_rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_auroc"])
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train anomaly model on RIC sessions.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "anomaly_train_v1.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--run-id", default="", help="Optional fixed run id.")
    parser.add_argument("--epochs", type=int, default=0, help="Override config epochs when > 0.")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    seed = int(cfg["experiment"]["seed"])
    _seed_all(seed)

    cfg_hash = _cfg_hash(args.config)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_prefix = str(cfg["experiment"].get("run_id_prefix", "run"))
    run_id = args.run_id or f"{run_prefix}-{ts}-{cfg_hash}"

    dirs = _ensure_dirs(cfg, run_id)
    tcfg = cfg["train"]
    device = _pick_device(str(tcfg.get("device", "auto")).lower())

    train_loader, val_loader, input_size = _build_loaders(cfg, seed=seed)
    model = GaitAnomalyDetector(input_size=input_size, hidden_size=int(tcfg.get("hidden_size", 128))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    start_epoch = 1
    best_val = float("inf")
    if args.resume_checkpoint is not None:
        state = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val = float(state.get("best_val_loss", best_val))

    n_epochs = int(args.epochs) if args.epochs > 0 else int(tcfg["epochs"])
    grad_clip_norm = float(tcfg.get("grad_clip_norm", 0.0))
    epoch_rows: list[dict] = []

    # Write run metadata before training starts.
    run_meta = {
        "run_id": run_id,
        "config_path": str(args.config),
        "config_hash": cfg_hash,
        "seed": seed,
        "device": str(device),
        "git_commit": _git_commit(),
        "started_utc": ts,
        "data": cfg["data"],
        "train": cfg["train"],
    }
    (dirs["logs"] / "run_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    for epoch in range(start_epoch, n_epochs + 1):
        train_loss = _run_epoch_train(model, train_loader, optimizer, device, grad_clip_norm)
        val_metrics = _run_epoch_val(model, val_loader, device)
        val_loss = float(val_metrics["val_loss"])
        val_auroc = float(val_metrics["val_auroc"])

        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 8),
                "val_loss": round(val_loss, 8),
                "val_auroc": "" if np.isnan(val_auroc) else round(val_auroc, 8),
            }
        )

        ckpt_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val,
            "run_id": run_id,
            "config_hash": cfg_hash,
            "input_size": input_size,
            "hidden_size": int(tcfg.get("hidden_size", 128)),
        }
        torch.save(ckpt_payload, dirs["checkpoints"] / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            ckpt_payload["best_val_loss"] = best_val
            torch.save(ckpt_payload, dirs["checkpoints"] / "best.pt")

        print(
            f"Epoch {epoch:03d}/{n_epochs}: "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_auroc={'n/a' if np.isnan(val_auroc) else f'{val_auroc:.4f}'}"
        )

    _write_epoch_metrics(dirs["logs"] / "epoch_metrics.csv", epoch_rows)
    summary = {
        "run_id": run_id,
        "best_val_loss": best_val,
        "epochs_completed": n_epochs,
        "checkpoints_dir": str(dirs["checkpoints"]),
        "logs_dir": str(dirs["logs"]),
        "results_dir": str(dirs["results"]),
    }
    (dirs["results"] / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Training complete. Best val loss: {best_val:.6f}")
    print(f"Run id: {run_id}")


if __name__ == "__main__":
    main()

