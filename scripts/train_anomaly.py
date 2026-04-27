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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
from torch.utils.data import WeightedRandomSampler
from models.gait_classifier import build_anomaly_model


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


def _validate_scheduler_cfg(scheduler_cfg: dict) -> None:
    """
    Guard against silent scheduler misconfiguration.

    ReduceLROnPlateau must use:
    - mode=min for val_loss
    - mode=max for val_auroc
    """
    if not bool(scheduler_cfg.get("enabled", True)):
        return
    metric = str(scheduler_cfg.get("metric", "val_loss")).lower()
    mode = str(scheduler_cfg.get("mode", "min")).lower()
    if metric not in {"val_loss", "val_auroc"}:
        raise ValueError("train.scheduler.metric must be one of: val_loss, val_auroc")
    if mode not in {"min", "max"}:
        raise ValueError("train.scheduler.mode must be one of: min, max")
    expected = "min" if metric == "val_loss" else "max"
    if mode != expected:
        raise ValueError(
            f"Scheduler mismatch: metric={metric} requires mode={expected}, but got mode={mode}. "
            "Fix config to avoid wrong LR updates."
        )


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


def _build_loaders(
    cfg: dict, seed: int
) -> tuple[DataLoader, DataLoader, int, str, torch.Tensor | None, dict[str, float | int]]:
    d = cfg["data"]
    t = cfg["train"]
    label_col = str(d.get("label_col", "is_injured"))
    fe = str(d.get("feature_engineering", "none")).lower()
    seq = SequenceSpec(
        seq_len=int(d["seq_len"]),
        prefer_mode=str(d.get("prefer_mode", "run")),
        normalize=str(d.get("normalize", "zscore")),
        train_random_window=bool(d.get("train_random_window", True)),
        feature_engineering=fe,
    )

    base_dim = compute_max_feature_dim(
        ROOT / d["session_split_csv"], ROOT / d["json_root"], seq, split_filter="train"
    )
    if fe == "motion_stats":
        feat_dim = int(base_dim + _motion_stats_extra_channels())
    else:
        feat_dim = int(base_dim)

    train_ds = RICAnomalyDataset(
        session_split_csv=ROOT / d["session_split_csv"],
        json_root=ROOT / d["json_root"],
        split="train",
        seq=seq,
        include_injured=bool(d.get("train_include_injured", False)),
        seed=seed,
        feature_dim=feat_dim,
        label_col=label_col,
    )
    val_ds = RICAnomalyDataset(
        session_split_csv=ROOT / d["session_split_csv"],
        json_root=ROOT / d["json_root"],
        split="val",
        seq=seq,
        include_injured=bool(d.get("val_include_injured", True)),
        seed=seed,
        feature_dim=feat_dim,
        label_col=label_col,
    )

    # Optional oversampling: only meaningful when the training set contains both classes.
    train_sampler: WeightedRandomSampler | None = None
    if bool(t.get("oversample_train_injured", False)) and bool(d.get("train_include_injured", False)):
        ys = [int(r.get(label_col, 0) or 0) for r in train_ds.rows]
        n0 = max(1, sum(1 for y in ys if y == 0))
        n1 = max(1, sum(1 for y in ys if y == 1))
        w0 = 1.0 / n0
        w1 = 1.0 / n1
        weights = [w0 if y == 0 else w1 for y in ys]
        train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    # sklearn-style class-balanced weights: n / (2 * n_c) for binary (used in weighted loss, not the sampler).
    w_policy = str(t.get("recon_class_weight", "none")).lower()
    n0 = n1 = 0
    class_w: dict[str, float | int] = {}
    if w_policy in ("balanced", "inverse") and bool(d.get("train_include_injured", False)):
        ys = [int(r.get(label_col, 0) or 0) for r in train_ds.rows]
        n0 = sum(1 for y in ys if y == 0)
        n1 = sum(1 for y in ys if y == 1)
        class_w = {"n_healthy_sessions": n0, "n_injured_sessions": n1}

    train_shuffle = train_sampler is None
    def _worker_init_fn(worker_id: int) -> None:
        # Keep dataloader randomness reproducible across workers while still unique per worker.
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    dl_generator = torch.Generator()
    dl_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(t["batch_size"]),
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=int(t["num_workers"]),
        collate_fn=collate_ric_anomaly,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        generator=dl_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(t["batch_size"]),
        shuffle=False,
        num_workers=int(t["num_workers"]),
        collate_fn=collate_ric_anomaly,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        generator=dl_generator,
    )
    # class_weight tensor: index 0->weight healthy, 1->weight injured; None if not used
    cwt: torch.Tensor | None = None
    if w_policy == "balanced" and n0 + n1 > 0 and n0 > 0 and n1 > 0:
        n = float(n0 + n1)
        cwt = torch.tensor(
            [n / (2.0 * n0), n / (2.0 * n1)], dtype=torch.float32
        )  # per sklearn balanced
    elif w_policy == "inverse" and n0 > 0 and n1 > 0:
        cwt = torch.tensor([1.0, float(n0) / float(n1)], dtype=torch.float32)  # injured gets n0/n1

    return train_loader, val_loader, feat_dim, w_policy, cwt, class_w


def _mse_per_sample(decoded: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Mean over sequence and feature dims -> one score per sample.
    return ((decoded - x) ** 2).mean(dim=(1, 2))


def _recon_loss(
    decoded: torch.Tensor,
    x: torch.Tensor,
    loss_name: str,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reconstruction loss for the autoencoder.

    When `sample_weight` is set (session-level), it implements a class-frequency reweighting
    analogous to `pos_weight` in BCE: only used if you train with injured sessions included
    (`train_include_injured: true`). It does not apply to the default healthy-only training.
    """
    name = (loss_name or "mse").lower()
    if name == "mse":
        err = (decoded - x) ** 2
        per = err.mean(dim=(1, 2))
    elif name == "huber":
        err = torch.nn.functional.smooth_l1_loss(decoded, x, reduction="none")
        per = err.mean(dim=(1, 2))
    else:
        raise ValueError(f"Unsupported train.loss: {loss_name}")
    if sample_weight is None:
        return per.mean()
    sw = sample_weight.to(per.device)
    return (per * sw).sum() / sw.sum().clamp_min(1e-8)


def _run_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
    loss_name: str,
    class_weight_tensor: torch.Tensor | None,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["is_injured"].to(device)
        optimizer.zero_grad(set_to_none=True)
        decoded, _ = model(x)
        sw = None
        if class_weight_tensor is not None:
            sw = class_weight_tensor[y]
        loss = _recon_loss(decoded, x, loss_name, sw)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def _run_epoch_val(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    loss_sum = 0.0
    n_samples = 0
    y_true = []
    y_score = []
    for batch in loader:
        x = batch["x"].to(device)
        decoded, _ = model(x)
        per_sample = _mse_per_sample(decoded, x)
        per_np = per_sample.detach().cpu().numpy()
        loss_sum += float(np.sum(per_np))
        n_samples += int(len(per_np))
        y_score.extend(per_sample.detach().cpu().numpy().tolist())
        y_true.extend(batch["is_injured"].cpu().numpy().tolist())

    out = {"val_loss": (loss_sum / n_samples) if n_samples else float("nan")}
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
    mcfg = cfg.get("model", {})
    device = _pick_device(str(tcfg.get("device", "auto")).lower())

    train_loader, val_loader, input_size, _wpol, class_w_tensor, class_w_meta = _build_loaders(cfg, seed=seed)
    model_variant = str(mcfg.get("variant", "baseline"))
    model = build_anomaly_model(
        variant=model_variant,
        input_size=input_size,
        hidden_size=int(tcfg.get("hidden_size", 128)),
        num_layers=int(mcfg.get("num_layers", 2)),
        dropout=float(mcfg.get("dropout", 0.2)),
        bidirectional=bool(mcfg.get("bidirectional", True)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_auroc = float("-inf")
    best_val_loss_at_best_auroc = float("inf")
    if args.resume_checkpoint is not None:
        state = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val_loss = float(state.get("best_val_loss", best_val_loss))
        best_val_auroc = float(state.get("best_val_auroc", best_val_auroc))
        best_val_loss_at_best_auroc = float(
            state.get("best_val_loss_at_best_auroc", best_val_loss_at_best_auroc)
        )

    scheduler_cfg = tcfg.get("scheduler", {})
    _validate_scheduler_cfg(scheduler_cfg)
    scheduler_enabled = bool(scheduler_cfg.get("enabled", True))
    scheduler_mode = str(scheduler_cfg.get("mode", "min")).lower()
    scheduler_metric = str(scheduler_cfg.get("metric", "val_loss")).lower()
    scheduler = None
    if scheduler_enabled:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 4)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )
    early_cfg = tcfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", True))
    early_patience = int(early_cfg.get("patience", 10))
    epochs_without_improve = 0

    n_epochs = int(args.epochs) if args.epochs > 0 else int(tcfg["epochs"])
    grad_clip_norm = float(tcfg.get("grad_clip_norm", 0.0))
    loss_name = str(tcfg.get("loss", "mse"))
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
        "model": mcfg,
        "train": cfg["train"],
        "class_balance_train": class_w_meta,
    }
    (dirs["logs"] / "run_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    for epoch in range(start_epoch, n_epochs + 1):
        train_loss = _run_epoch_train(
            model, train_loader, optimizer, device, grad_clip_norm, loss_name, class_w_tensor
        )
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
            "best_val_loss": best_val_loss,
            "best_val_auroc": best_val_auroc,
            "best_val_loss_at_best_auroc": best_val_loss_at_best_auroc,
            "run_id": run_id,
            "config_hash": cfg_hash,
            "input_size": input_size,
            "hidden_size": int(tcfg.get("hidden_size", 128)),
            "model_variant": model_variant,
            "model_num_layers": int(mcfg.get("num_layers", 2)),
            "model_dropout": float(mcfg.get("dropout", 0.2)),
            "model_bidirectional": bool(mcfg.get("bidirectional", True)),
        }
        torch.save(ckpt_payload, dirs["checkpoints"] / "last.pt")

        # Checkpoint selection: prioritize AUROC when available; otherwise fall back to loss.
        improved = False
        if not np.isnan(val_auroc):
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_val_loss_at_best_auroc = val_loss
                improved = True
            elif (
                np.isclose(val_auroc, best_val_auroc, atol=1e-12)
                and val_loss < best_val_loss_at_best_auroc
            ):
                best_val_loss_at_best_auroc = val_loss
                improved = True
        elif best_val_auroc == float("-inf") and val_loss < best_val_loss:
            improved = True

        if improved:
            best_val_loss = min(best_val_loss, val_loss)
            ckpt_payload["best_val_loss"] = best_val_loss
            ckpt_payload["best_val_auroc"] = best_val_auroc
            ckpt_payload["best_val_loss_at_best_auroc"] = best_val_loss_at_best_auroc
            torch.save(ckpt_payload, dirs["checkpoints"] / "best.pt")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if scheduler is not None:
            if scheduler_metric == "val_auroc":
                scheduler.step(val_auroc if not np.isnan(val_auroc) else 0.0)
            else:
                scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d}/{n_epochs}: "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_auroc={'n/a' if np.isnan(val_auroc) else f'{val_auroc:.4f}'}"
        )
        if early_enabled and epochs_without_improve >= early_patience:
            print(f"Early stopping at epoch {epoch}: no checkpoint improvement for {early_patience} epoch(s).")
            break

    _write_epoch_metrics(dirs["logs"] / "epoch_metrics.csv", epoch_rows)
    summary = {
        "run_id": run_id,
        "best_val_loss": best_val_loss,
        "best_val_auroc": None if best_val_auroc == float("-inf") else best_val_auroc,
        "best_val_loss_at_best_auroc": (
            None if best_val_loss_at_best_auroc == float("inf") else best_val_loss_at_best_auroc
        ),
        "epochs_completed": len(epoch_rows),
        "checkpoints_dir": str(dirs["checkpoints"]),
        "logs_dir": str(dirs["logs"]),
        "results_dir": str(dirs["results"]),
    }
    (dirs["results"] / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"Run id: {run_id}")


if __name__ == "__main__":
    main()

