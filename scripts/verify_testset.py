"""Check a rebuilt unified test set against the released manifest (text hashes and labels).

    python scripts/verify_testset.py [--manifest data/testset/unified_test_manifest.csv]
                                     [--rebuilt data/processed/testset_jsonld/unified_test.jsonl]
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def text_hash(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).encode("utf8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data/testset/unified_test_manifest.csv"))
    parser.add_argument("--rebuilt", default=str(ROOT / "data/processed/testset_jsonld/unified_test.jsonl"))
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    rebuilt = pd.read_json(args.rebuilt, lines=True)
    rebuilt["text_sha1"] = rebuilt["text"].map(text_hash)
    merged = manifest.merge(rebuilt[["text_sha1", "label"]], on="text_sha1", how="left", suffixes=("", "_rebuilt"))
    found = merged["label_rebuilt"].notna()
    same_label = (merged["label"] == merged["label_rebuilt"])[found]
    print(f"manifest articles: {len(manifest)}")
    print(f"found in rebuilt set: {found.sum()} ({found.mean():.1%})")
    print(f"same label when found: {same_label.mean():.1%}")
    print(merged[~found].groupby("source").size().rename("missing").to_string())


if __name__ == "__main__":
    main()
