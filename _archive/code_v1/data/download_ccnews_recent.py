"""Candidate articles for the human-annotated test set, from raw Common Crawl CC-NEWS 2024-2026.

The test set must be disjoint from everything the students or the teacher saw:
  * time    - CC-NEWS crawls from 2024-01 onward (training data is CC-News 2019-2021)
  * sites   - every domain of the training / held-out pools is excluded
  * content - near-duplicates of training documents are removed later (prepare_human_test_pool.py)

For each month a few WARC files (~1 GB, ~30k pages each) spread over the month are downloaded,
processed and deleted again, so disk use stays at a few GB. Per page: HTML -> main text with
trafilatura, title + body as in the training data (download_ccnews.build_text), fastText language
ID, length filter and truncation to the first 512 words. One parquet shard per WARC file makes
the run resumable.

Usage:
    python data/download_ccnews_recent.py --start 2024-01 --end 2026-08 --files-per-month 2 --workers 6
"""
import argparse
import gzip
import io
import json
import random
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MIN_WORDS, MODELS_DIR, PROCESSED_DIR, RAW_DIR, truncate_words  # noqa: E402

import pandas as pd  # noqa: E402
import requests  # noqa: E402

BASE = "https://data.commoncrawl.org/"
OUT_DIR = RAW_DIR / "ccnews_recent"
SHARD_DIR = OUT_DIR / "shards"
TMP_DIR = OUT_DIR / "tmp"
LID_MODEL = MODELS_DIR / "lid" / "lid.176.bin"

# Candidate languages for the human test set (the final 5-7 are chosen from the yield report).
TARGET_LANGS = {"id", "en", "ms", "jv", "su", "ar", "tr", "zh", "hi", "ru"}
NO_SPACE_LANGS = {"zh", "ja", "th"}
MIN_CHARS_NO_SPACE, MAX_CHARS_NO_SPACE = 200, 2000
EN_KEEP = 0.12          # English is ~40% of CC-NEWS; a sample is plenty
MIN_LID_CONF = 0.70
HTML_LANG = re.compile(rb'<html[^>]*\slang\s*=\s*["\']?([a-zA-Z]{2,3})', re.I)

_lid = None
_excluded = None


def excluded_domains() -> set[str]:
    """All news sites of the training, extra and held-out CC-News pools (with and without www.)."""
    doms = set()
    for f in ("ccnews_train", "ccnews_train_extra", "ccnews_heldout"):
        doms |= set(pd.read_parquet(PROCESSED_DIR / "teacher_labels" / f"{f}.parquet", columns=["domain"])["domain"])
    return doms | {d[4:] for d in doms if d.startswith("www.")} | {"www." + d for d in doms}


def _init():
    global _lid, _excluded
    import fasttext
    fasttext.FastText.eprint = lambda *a, **k: None
    _lid = fasttext.load_model(str(LID_MODEL))
    _excluded = excluded_domains()


def detect_lang(text: str) -> tuple[str, float]:
    labels, probs = _lid.predict(text[:1500].replace("\n", " "), k=1)
    return labels[0].replace("__label__", ""), float(probs[0])


def month_files(month: str, k: int) -> list[str]:
    year, mon = month.split("-")
    r = requests.get(f"{BASE}crawl-data/CC-NEWS/{year}/{mon}/warc.paths.gz", timeout=60)
    r.raise_for_status()
    paths = gzip.decompress(r.content).decode().split()
    step = len(paths) / k  # evenly spread over the month
    return [paths[int(i * step + step / 2)] for i in range(k)]


def download(path: str, dest: Path, retries: int = 5) -> None:
    for attempt in range(retries):
        try:
            with requests.get(BASE + path, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            return
        except Exception:  # noqa: BLE001 - network hiccups: back off and retry
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"download failed: {path}")


def process(path: str) -> dict:
    from warcio.archiveiterator import ArchiveIterator
    import trafilatura

    name = Path(path).name.replace(".warc.gz", "")
    shard = SHARD_DIR / f"{name}.parquet"
    if shard.exists():
        return {"file": name, "skipped": True}
    rng = random.Random(name)
    tmp = TMP_DIR / f"{name}.warc.gz"
    t0 = time.time()
    download(path, tmp)
    t_dl = time.time() - t0
    rows, seen, n_pages = [], set(), 0
    with open(tmp, "rb") as f:
        for rec in ArchiveIterator(f):
            if rec.rec_type != "response" or not rec.http_headers:
                continue
            if rec.http_headers.get_statuscode() != "200":
                continue
            if "html" not in (rec.http_headers.get_header("Content-Type") or ""):
                continue
            n_pages += 1
            url = rec.rec_headers.get_header("WARC-Target-URI")
            domain = urlparse(url).netloc.lower()
            if domain in _excluded or url in seen:
                continue
            html = rec.content_stream().read()
            m = HTML_LANG.search(html[:4000])
            hint = m.group(1).decode().lower()[:2] if m else None
            if hint and hint not in TARGET_LANGS:
                continue
            if hint == "en" and rng.random() > EN_KEEP:
                continue
            try:
                extracted = trafilatura.extract(html, output_format="json", with_metadata=True,
                                                include_comments=False, include_tables=False)
            except Exception:  # noqa: BLE001 - malformed HTML
                continue
            if not extracted:
                continue
            meta = json.loads(extracted)
            body = (meta.get("text") or "").strip()
            title = (meta.get("title") or "").strip()
            if not body:
                continue
            lang, conf = detect_lang(body)
            if lang not in TARGET_LANGS or conf < MIN_LID_CONF:
                continue
            if lang == "en" and hint != "en" and rng.random() > EN_KEEP:
                continue
            text = f"{title}\n\n{body}" if title and not body.startswith(title) else body
            if lang in NO_SPACE_LANGS:
                if len(body) < MIN_CHARS_NO_SPACE:
                    continue
                text = text[:MAX_CHARS_NO_SPACE]
            else:
                if len(body.split()) < MIN_WORDS:
                    continue
                text = truncate_words(text)
            seen.add(url)
            rows.append({"url": url, "domain": domain, "path": urlparse(url).path,
                         "date": meta.get("date"), "warc_date": rec.rec_headers.get_header("WARC-Date"),
                         "lang": lang, "lang_conf": round(conf, 3), "html_lang": hint,
                         "title": title, "text": text, "n_words": len(body.split()), "warc": name})
    tmp.unlink(missing_ok=True)
    pd.DataFrame(rows).to_parquet(shard, index=False)
    counts = pd.Series([r["lang"] for r in rows]).value_counts().to_dict() if rows else {}
    return {"file": name, "pages": n_pages, "kept": len(rows), "download_s": round(t_dl),
            "total_s": round(time.time() - t0), "langs": counts}


def months(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01")
    parser.add_argument("--end", default="2026-08")
    parser.add_argument("--files-per-month", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    paths = [p for m in months(args.start, args.end) for p in month_files(m, args.files_per_month)]
    print(f"{len(paths)} WARC files ({args.start} .. {args.end}, {args.files_per_month}/month)", flush=True)
    log = open(OUT_DIR / "download_log.jsonl", "a", encoding="utf8")
    with Pool(args.workers, initializer=_init) as pool:
        for i, res in enumerate(pool.imap_unordered(process, paths), 1):
            res["done"] = f"{i}/{len(paths)}"
            print(json.dumps(res), flush=True)
            log.write(json.dumps(res) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
