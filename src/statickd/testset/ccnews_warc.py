"""Download raw Common Crawl CC-NEWS WARC files (2024-2026) for building the human test set.

Only downloads; nothing is parsed or filtered here. Files are picked evenly over each month
(deterministic), saved unchanged, and verified against the server's Content-Length, so an
interrupted run can simply be restarted. A manifest lists every file with its size.

Usage:
    python -m statickd.testset.ccnews_warc --start 2024-01 --end 2026-08 --files-per-month 1
"""
import argparse
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from statickd.config import RAW_DIR  # noqa: E402

import pandas as pd  # noqa: E402
import requests  # noqa: E402

BASE = "https://data.commoncrawl.org/"
WARC_DIR = RAW_DIR / "ccnews_recent" / "warc"


def month_files(month: str, k: int) -> list[str]:
    year, mon = month.split("-")
    r = requests.get(f"{BASE}crawl-data/CC-NEWS/{year}/{mon}/warc.paths.gz", timeout=60)
    r.raise_for_status()
    paths = gzip.decompress(r.content).decode().split()
    step = len(paths) / k
    return [paths[int(i * step + step / 2)] for i in range(k)]


def download(path: str, retries: int = 5) -> dict:
    dest = WARC_DIR / Path(path).name
    size = int(requests.head(BASE + path, timeout=60).headers["Content-Length"])
    if dest.exists() and dest.stat().st_size == size:
        return {"path": path, "bytes": size, "status": "exists"}
    for attempt in range(retries):
        try:
            with requests.get(BASE + path, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if dest.stat().st_size == size:
                return {"path": path, "bytes": size, "status": "ok"}
        except Exception:  # noqa: BLE001 - network hiccup: back off and retry
            pass
        time.sleep(15 * (attempt + 1))
    return {"path": path, "bytes": size, "status": "FAILED"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01")
    parser.add_argument("--end", default="2026-08")
    parser.add_argument("--files-per-month", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    WARC_DIR.mkdir(parents=True, exist_ok=True)
    months = [p.strftime("%Y-%m") for p in pd.period_range(args.start, args.end, freq="M")]
    paths = [p for m in months for p in month_files(m, args.files_per_month)]
    print(f"{len(paths)} WARC files to fetch into {WARC_DIR}", flush=True)
    with ThreadPoolExecutor(args.parallel) as pool:
        results = []
        for res in pool.map(download, paths):
            results.append(res)
            print(json.dumps(res), flush=True)
    (WARC_DIR.parent / "manifest.json").write_text(json.dumps(
        {"source": "Common Crawl CC-NEWS", "months": [args.start, args.end],
         "files_per_month": args.files_per_month, "files": results}, indent=1))
    failed = [r for r in results if r["status"] == "FAILED"]
    print(f"done: {len(results) - len(failed)} ok, {len(failed)} failed, "
          f"{sum(r['bytes'] for r in results) / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
