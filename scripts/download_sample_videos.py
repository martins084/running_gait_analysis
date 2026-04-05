"""
Batch-download videos from a text file of URLs (one per line) using yt-dlp.

Lines starting with # are ignored. Trailing comments after URL are stripped.

Usage:
  python scripts/download_sample_videos.py --urls-file video_urls.txt --output-dir data/raw/sample_videos
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_urls(path: Path) -> list[str]:
    urls = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # First URL-like token
            m = re.search(r"https?://\S+", line)
            if m:
                urls.append(m.group(0).rstrip(").,]\"'"))
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(description="Download videos from URL list via yt-dlp.")
    parser.add_argument("--urls-file", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/sample_videos"),
        help="Directory for downloaded MP4 files",
    )
    parser.add_argument(
        "--format",
        default="bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        help="yt-dlp -f argument (default prefers mp4)",
    )
    args = parser.parse_args()

    if not args.urls_file.is_file():
        print(f"Missing URLs file: {args.urls_file}", file=sys.stderr)
        sys.exit(1)

    urls = parse_urls(args.urls_file)
    if not urls:
        print("No URLs found in file.", file=sys.stderr)
        sys.exit(2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"download_log_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt"

    ok = fail = 0
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"urls_file={args.urls_file}\noutput_dir={args.output_dir}\n\n")
        for i, url in enumerate(urls, 1):
            # Unique filename stem per line order to avoid overwrites
            out_tmpl = str(args.output_dir / f"%(title).80B_{i}.%(ext)s")
            cmd = [
                "yt-dlp",
                "-f",
                args.format,
                "--merge-output-format",
                "mp4",
                "-o",
                out_tmpl,
                "--no-playlist",
                url,
            ]
            log.write(f"--- [{i}/{len(urls)}] {url}\n")
            log.write(f"CMD: {' '.join(cmd)}\n")
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                log.write(r.stdout or "")
                log.write(r.stderr or "")
                if r.returncode == 0:
                    ok += 1
                    log.write("RESULT: OK\n\n")
                else:
                    fail += 1
                    log.write(f"RESULT: FAIL code={r.returncode}\n\n")
            except FileNotFoundError:
                log.write("RESULT: FAIL yt-dlp not found on PATH\n\n")
                print(
                    "yt-dlp not found. Install: pip install yt-dlp (and ensure executable on PATH).",
                    file=sys.stderr,
                )
                sys.exit(3)
            except subprocess.TimeoutExpired:
                fail += 1
                log.write("RESULT: FAIL timeout\n\n")

    print(f"Done. success={ok} fail={fail} log={log_path}")


if __name__ == "__main__":
    main()
