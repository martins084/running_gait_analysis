# AMS3 Pre-GPU Runbook (Anomaly Pipeline)

This runbook is the minimal-risk sequence to avoid wasting GPU hours.

## 1) Build manifest and splits (CPU)

```bash
python scripts/build_ric_manifest.py \
  --s3-prefix s3://bakalaurs-ams/processed/reformat_data/ \
  --endpoint-url https://ams3.digitaloceanspaces.com \
  --output-csv data/processed/ric_manifest.csv \
  --summary-json data/processed/ric_manifest_summary.json
```

```bash
python scripts/build_subject_splits.py \
  --manifest-csv data/processed/ric_manifest.csv \
  --output-subject-split data/processed/splits/subject_split_v1.csv \
  --output-session-split data/processed/splits/session_split_v1.csv \
  --seed 42
```

## 2) Prepare local JSON root for training

`core/ric_dataset.py` expects local JSON files under:

- `data/ric/reformat_data/<subject>/<session>.json`

If needed, sync from Spaces before training.

## 3) CPU smoke training (must pass before GPU)

```bash
python scripts/train_anomaly.py \
  --config config/anomaly_train_v1.yaml \
  --epochs 1
```

Expected outputs:

- `checkpoints/<run_id>/last.pt`
- `checkpoints/<run_id>/best.pt`
- `logs/<run_id>/run_metadata.json`
- `logs/<run_id>/epoch_metrics.csv`
- `results/<run_id>/train_summary.json`

## 4) CPU evaluation smoke

```bash
python scripts/evaluate_anomaly.py \
  --config config/anomaly_train_v1.yaml \
  --checkpoint checkpoints/<run_id>/best.pt \
  --split val \
  --output-dir results/<run_id>/eval_val
```

Expected outputs:

- `results/<run_id>/eval_val/metrics.json`
- `results/<run_id>/eval_val/thresholds.json`
- `results/<run_id>/eval_val/per_sample_scores.csv`

## 5) Pre-GPU go/no-go checklist

- [ ] Manifest row count matches expected session count.
- [ ] Subject split has zero leakage (subject-level).
- [ ] CPU 1-epoch train run completes with checkpoints.
- [ ] Evaluation script produces metrics + thresholds.
- [ ] Paths in `config/anomaly_train_v1.yaml` are correct.
- [ ] Artifact sync command works in dry-run mode.

## 6) First GPU run (AMS3)

1. Create AMS3 GPU droplet with AI/ML-ready image.
2. Verify:
   - `nvidia-smi`
   - Python CUDA visibility.
3. Run short validation run first:
   - `python scripts/train_anomaly.py --config config/anomaly_train_v1.yaml --epochs 2`
4. Evaluate best checkpoint.
5. Sync artifacts to Spaces.

## 7) Artifact sync to AMS bucket

```bash
python scripts/sync_run_artifacts.py \
  --run-id <run_id> \
  --bucket bakalaurs-ams \
  --endpoint-url https://ams3.digitaloceanspaces.com
```

Targets:

- `s3://bakalaurs-ams/checkpoints/<run_id>/`
- `s3://bakalaurs-ams/logs/<run_id>/`
- `s3://bakalaurs-ams/results/<run_id>/`

