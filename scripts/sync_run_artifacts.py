"""
Sync run artifacts to AMS Spaces bucket.

Example:
  python scripts/sync_run_artifacts.py \
    --run-id ric-anom-20260415T120000Z-abcd1234 \
    --bucket bakalaurs-ams \
    --endpoint-url https://ams3.digitaloceanspaces.com
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sync_dir(local_dir: Path, s3_uri: str, endpoint_url: str, dry_run: bool) -> None:
    if not local_dir.is_dir():
        print(f"Skip missing dir: {local_dir}")
        return
    cmd = ["aws", "s3", "cp", str(local_dir), s3_uri, "--recursive"]
    if endpoint_url:
        cmd.extend(["--endpoint-url", endpoint_url])
    if dry_run:
        cmd.append("--dryrun")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload run artifacts (checkpoints/logs/results) to Spaces.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bucket", default="bakalaurs-ams")
    parser.add_argument("--endpoint-url", default="https://ams3.digitaloceanspaces.com")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id
    targets = [
        (ROOT / "checkpoints" / run_id, f"s3://{args.bucket}/checkpoints/{run_id}/"),
        (ROOT / "logs" / run_id, f"s3://{args.bucket}/logs/{run_id}/"),
        (ROOT / "results" / run_id, f"s3://{args.bucket}/results/{run_id}/"),
    ]

    try:
        for local_dir, s3_uri in targets:
            _sync_dir(local_dir, s3_uri, endpoint_url=args.endpoint_url, dry_run=args.dry_run)
    except subprocess.CalledProcessError as exc:
        print(f"Sync command failed: {exc}", file=sys.stderr)
        sys.exit(2)

    print("Artifact sync complete.")


if __name__ == "__main__":
    main()

