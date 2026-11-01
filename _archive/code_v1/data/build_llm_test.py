"""Multilingual LLM-annotated test set (languages outside the teacher's training data).

Step 1 (`sample`): draw documents from the CC-News held-out split (news domains never used for
training) - a fixed number per language, spread round-robin over domains, natural topic distribution.
The batches are written without any model prediction so annotation stays independent.

Step 2 (annotation): each batch is labelled by an LLM using exactly the prompt and label
descriptions of Kuzman & Ljubesic (2025, Appendix B), replacing GPT-4o by Claude.

Step 3 (`merge`): collect the annotations into data/processed/llm_test.jsonl.

Usage:
    python data/build_llm_test.py sample
    python data/build_llm_test.py merge
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABEL2ID, PROCESSED_DIR  # noqa: E402

import pandas as pd  # noqa: E402

LANGS = ["en", "de", "fr", "es", "pt", "it", "ru", "pl", "tr", "ar", "zh", "ja", "id", "vi", "hi", "sw"]
PER_LANG = 20
BATCH_SIZE = 40
MAX_CHARS = 3000  # annotation sees the article start; long tails rarely change the topic
WORK_DIR = PROCESSED_DIR / "llm_test_work"


def sample(seed: int = 7) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for lang in LANGS:
        df = pd.read_json(PROCESSED_DIR / "ccnews" / f"{lang}.heldout.jsonl", lines=True)
        df = df.sample(frac=1.0, random_state=seed)
        # Round-robin over domains: 1st doc of every domain, then the 2nd, ... for source diversity.
        df = df.assign(rank=df.groupby("domain").cumcount()).sort_values("rank", kind="stable").head(PER_LANG)
        rows.append(df)
    df = pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["doc_id"] = [f"d{i:03d}" for i in range(len(df))]
    df.to_json(WORK_DIR / "sample.jsonl", orient="records", lines=True, force_ascii=False)
    for b, start in enumerate(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[start:start + BATCH_SIZE]
        with open(WORK_DIR / f"batch_{b:02d}.txt", "w", encoding="utf8") as f:
            for r in batch.itertuples():
                f.write(f"=== {r.doc_id} ===\n{r.text[:MAX_CHARS]}\n\n")
    print(f"{len(df)} docs in {b + 1} batches -> {WORK_DIR}")


def merge() -> None:
    df = pd.read_json(WORK_DIR / "sample.jsonl", lines=True)
    labels = {}
    for f in sorted(WORK_DIR.glob("labels_*.json")):
        labels.update(json.loads(f.read_text(encoding="utf8")))
    df["label"] = df["doc_id"].map(labels)
    missing = df["label"].isna().sum()
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin(LABEL2ID)]
    df["label_id"] = df["label"].map(LABEL2ID)
    df[["doc_id", "lang", "domain", "text", "label", "label_id"]].to_json(
        PROCESSED_DIR / "llm_test.jsonl", orient="records", lines=True, force_ascii=False)
    print(f"llm_test: {len(df)} docs ({missing} missing)\n{df['label'].value_counts().to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["sample", "merge"])
    step = parser.parse_args().step
    sample() if step == "sample" else merge()
