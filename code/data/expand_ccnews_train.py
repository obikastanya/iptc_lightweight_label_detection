"""Add more CC-News training documents from the cached candidates (distillation data scaling).

Extra documents never come from held-out domains and never duplicate existing train or held-out
documents, so the held-out evaluation remains a test on unseen news sites.

Usage:
    python data/expand_ccnews_train.py --extra 5000
"""
import argparse
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR, RAW_DIR  # noqa: E402

MAX_DOCS_PER_DOMAIN = 400


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf8") as f:
        return [json.loads(line) for line in f]


def expand_language(lang: str, extra: int, rng: random.Random) -> list[dict]:
    ccnews = PROCESSED_DIR / "ccnews"
    train = read_jsonl(ccnews / f"{lang}.train.jsonl")
    heldout = read_jsonl(ccnews / f"{lang}.heldout.jsonl")
    heldout_domains = {d["domain"] for d in heldout}
    seen = {d["text"][:300] for d in train + heldout}
    per_domain = Counter(d["domain"] for d in train)

    candidates = read_jsonl(RAW_DIR / "ccnews_candidates" / f"{lang}.candidates.jsonl.gz")
    rng.shuffle(candidates)
    out = []
    for d in candidates:
        key = d["text"][:300]
        if d["domain"] in heldout_domains or key in seen or per_domain[d["domain"]] >= MAX_DOCS_PER_DOMAIN:
            continue
        seen.add(key)
        per_domain[d["domain"]] += 1
        out.append(d)
        if len(out) >= extra:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    total = 0
    for cand in sorted((RAW_DIR / "ccnews_candidates").glob("*.candidates.jsonl.gz")):
        lang = cand.name.split(".")[0]
        docs = expand_language(lang, args.extra, rng)
        with open(PROCESSED_DIR / "ccnews" / f"{lang}.train_extra.jsonl", "w", encoding="utf8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        total += len(docs)
        print(f"[{lang}] +{len(docs)}", flush=True)
    print(f"total extra documents: {total}")


if __name__ == "__main__":
    main()
