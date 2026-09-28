"""Build a multilingual, unlabeled news corpus from CC-News (CloverSearch/cc-news-mutlilingual).

The teacher model later pseudo-labels this corpus, which is what lets a small student learn
many languages. For each language we stream the beginning of the 2019-2021 per-language files,
keep documents whose declared and fastText language agree, cap documents per source domain
(the files are grouped by domain), and split train / held-out BY DOMAIN so that the held-out
set measures generalisation to unseen news sites.

Usage:
    python -m statickd.corpus.ccnews --per-lang 3000 --heldout 300
"""
import argparse
import gzip
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from statickd.config import MAX_WORDS, MIN_WORDS, PROCESSED_DIR, RAW_DIR, truncate_words  # noqa: E402

BASE_URL = "https://huggingface.co/datasets/CloverSearch/cc-news-mutlilingual/resolve/main/{}/{}.jsonl.gz"
# Files are grouped by source domain, so several crawl years are sampled to diversify sources.
YEARS = ["2019", "2020", "2021"]

# Languages supported by XLM-R that have a reasonable amount of news in CC-News.
LANGUAGES = [
    "af", "am", "ar", "az", "be", "bg", "bn", "bs", "ca", "cs", "cy", "da", "de", "el", "en",
    "es", "et", "eu", "fa", "fi", "fr", "gl", "gu", "he", "hi", "hr", "hu", "hy", "id", "is",
    "it", "ja", "ka", "kk", "kn", "ko", "ky", "lt", "lv", "mk", "ml", "mn", "mr", "ms", "my",
    "ne", "nl", "no", "or", "pa", "pl", "ps", "pt", "ro", "ru", "sk", "sl", "so", "sq", "sr",
    "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "zh",
]
# Very large languages are split into several files; the first shard is enough for sampling.
SHARDED = {"en": "en00", "es": "es00"}
# Scripts written without spaces between words: length is measured in characters instead.
NO_SPACE_LANGS = {"ja", "zh", "th", "my", "km", "lo"}
MIN_CHARS_NO_SPACE = 200
MAX_CHARS_NO_SPACE = 2000

MAX_DOCS_PER_DOMAIN = 150
MAX_COMPRESSED_BYTES = 50_000_000  # per year
MAX_PARSED_DOCS = 25_000  # per year


def build_text(record: dict) -> str:
    """Title + main text, mirroring the MaCoCu texts used by Kuzman (title on the first line)."""
    title = (record.get("title") or "").strip()
    body = (record.get("maintext") or "").strip()
    if title and title != "None" and not body.startswith(title):
        return f"{title}\n\n{body}"
    return body


def is_long_enough(text: str, lang: str) -> bool:
    if lang in NO_SPACE_LANGS:
        return len(text) >= MIN_CHARS_NO_SPACE
    return len(text.split()) >= MIN_WORDS


def truncate(text: str, lang: str) -> str:
    if lang in NO_SPACE_LANGS:
        return text[:MAX_CHARS_NO_SPACE]
    return truncate_words(text, MAX_WORDS)


def stream_language(lang: str) -> list[dict]:
    """Stream the head of each yearly language file and return clean candidate documents."""
    docs = []
    for year in YEARS:
        try:
            docs.extend(_stream_file(BASE_URL.format(year, SHARDED.get(lang, lang)), lang))
        except requests.HTTPError:
            continue  # the language may be missing for a given year
    return docs


def _stream_file(url: str, lang: str) -> list[dict]:
    docs, parsed = [], 0
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        counter = _ByteCounter(resp.raw, MAX_COMPRESSED_BYTES)
        try:
            with gzip.GzipFile(fileobj=counter) as gz:
                for line in gz:
                    parsed += 1
                    if parsed > MAX_PARSED_DOCS:
                        break
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("language") != lang or record.get("fasttext_language") != lang:
                        continue
                    text = build_text(record)
                    if not is_long_enough(text, lang):
                        continue
                    docs.append({
                        "text": truncate(text, lang),
                        "lang": lang,
                        "domain": record.get("source_domain") or "unknown",
                        "url": record.get("url"),
                    })
        except (EOFError, OSError, _LimitReached):
            pass  # partial stream is fine: we only need a sample
    return docs


class _LimitReached(Exception):
    pass


class _ByteCounter:
    """File-like wrapper that stops reading after `limit` compressed bytes."""

    def __init__(self, raw, limit: int):
        self.raw, self.limit, self.read_bytes = raw, limit, 0

    def read(self, size=-1):
        if self.read_bytes >= self.limit:
            raise _LimitReached
        chunk = self.raw.read(size)
        self.read_bytes += len(chunk)
        return chunk


def domain_split(docs: list[dict], per_lang: int, heldout: int, seed: int) -> tuple[list, list]:
    """Deduplicate, cap docs per domain, then assign whole domains to train or held-out.

    Held-out never exceeds 20% of the language's documents, so low-resource languages keep
    most data for training. With fewer than 3 domains a document-level split is used.
    """
    rng = random.Random(seed)
    by_domain = defaultdict(list)
    seen_texts = set()
    for d in docs:
        key = d["text"][:300]
        if key in seen_texts:
            continue
        seen_texts.add(key)
        by_domain[d["domain"]].append(d)
    domains = list(by_domain)
    rng.shuffle(domains)
    for dom in domains:
        rng.shuffle(by_domain[dom])
        by_domain[dom] = by_domain[dom][:MAX_DOCS_PER_DOMAIN]
    n_total = sum(len(v) for v in by_domain.values())
    heldout_target = min(heldout, int(0.2 * n_total))

    if len(domains) < 3:
        pool = [d for dom in domains for d in by_domain[dom]]
        rng.shuffle(pool)
        return pool[heldout_target:heldout_target + per_lang], pool[:heldout_target]

    heldout_docs, train_docs = [], []
    for dom in domains:
        target = heldout_docs if len(heldout_docs) < heldout_target else train_docs
        target.extend(by_domain[dom])
    rng.shuffle(train_docs)
    return train_docs[:per_lang], heldout_docs[:heldout_target]


def load_or_stream(lang: str, cache_dir: Path) -> list[dict]:
    """Raw candidates are cached so that re-splitting never needs a new download."""
    cache = cache_dir / f"{lang}.candidates.jsonl.gz"
    if cache.exists():
        with gzip.open(cache, "rt", encoding="utf8") as f:
            return [json.loads(line) for line in f]
    docs = stream_language(lang)
    with gzip.open(cache, "wt", encoding="utf8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-lang", type=int, default=3000)
    parser.add_argument("--heldout", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--langs", nargs="*", default=LANGUAGES)
    args = parser.parse_args()

    out_dir = PROCESSED_DIR / "ccnews"
    cache_dir = RAW_DIR / "ccnews_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(load_or_stream, lang, cache_dir): lang for lang in args.langs}
        for fut in as_completed(futures):
            lang = futures[fut]
            try:
                docs = fut.result()
            except Exception as exc:  # network errors should not kill the whole run
                print(f"[{lang}] failed: {exc}", flush=True)
                continue
            train, held = domain_split(docs, args.per_lang, args.heldout, args.seed)
            for name, rows in (("train", train), ("heldout", held)):
                with open(out_dir / f"{lang}.{name}.jsonl", "w", encoding="utf8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats[lang] = {"candidates": len(docs), "train": len(train), "heldout": len(held)}
            print(f"[{lang}] candidates={len(docs)} train={len(train)} heldout={len(held)}", flush=True)

    with open(out_dir / "stats.json", "w", encoding="utf8") as f:
        json.dump(dict(sorted(stats.items())), f, indent=1)


if __name__ == "__main__":
    main()
