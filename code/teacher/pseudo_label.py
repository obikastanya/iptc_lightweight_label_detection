"""Annotate the unlabeled corpora with the teacher's logits (knowledge-distillation targets).

GPU is used here only to save time; it is an offline, one-off data preparation step. The
full 17-dimensional logit vector is stored so that students can learn from soft labels.

Outputs (data/processed/teacher_labels/):
    emmediatopic.parquet     21k EMMediaTopic docs (GPT-4o label + teacher logits)
    ccnews_train.parquet     multilingual CC-News training pool
    ccnews_heldout.parquet   multilingual CC-News held-out docs (unseen news domains)
    ccnews_train_extra.parquet  additional training docs (data-scaling experiment)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR  # noqa: E402

import pandas as pd  # noqa: E402

from data.emmediatopic import load_emmediatopic  # noqa: E402
from teacher.teacher import Teacher  # noqa: E402

OUT_DIR = PROCESSED_DIR / "teacher_labels"


def load_ccnews(split: str) -> pd.DataFrame:
    files = sorted((PROCESSED_DIR / "ccnews").glob(f"*.{split}.jsonl"))
    return pd.concat([pd.read_json(f, lines=True) for f in files if f.stat().st_size > 0], ignore_index=True)


def annotate(teacher: Teacher, df: pd.DataFrame, name: str, batch_size: int) -> None:
    start = time.perf_counter()
    logits = teacher.logits(df["text"].tolist(), batch_size=batch_size)
    df = df.copy()
    df["teacher_logits"] = list(logits)
    df["teacher_label_id"] = logits.argmax(axis=1)
    df.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
    elapsed = time.perf_counter() - start
    print(f"[{name}] {len(df)} docs in {elapsed / 60:.1f} min ({len(df) / elapsed:.1f} docs/s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", nargs="*", default=["emmediatopic", "ccnews_heldout", "ccnews_train"])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    teacher = Teacher(device="cuda", fp16=True)
    loaders = {
        "emmediatopic": lambda: load_emmediatopic(),
        "ccnews_heldout": lambda: load_ccnews("heldout"),
        "ccnews_train": lambda: load_ccnews("train"),
        "ccnews_train_extra": lambda: load_ccnews("train_extra"),
    }
    for name in args.corpora:
        annotate(teacher, loaders[name](), name, args.batch_size)


if __name__ == "__main__":
    main()
