"""Label the distillation corpora with the teacher's 17 logits (offline, once; GPU recommended).

Outputs data/processed/teacher_labels/{emmediatopic,ccnews_train,ccnews_train_extra,ccnews_heldout}.parquet

Usage (from the project root):
    python scripts/label_with_teacher.py --corpora emmediatopic ccnews_train ccnews_heldout ccnews_train_extra
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from statickd.data import CCNEWS_DIR, TEACHER_LABELS_DIR, load_emmediatopic  # noqa: E402
from statickd.teacher import Teacher, label_corpus  # noqa: E402


def load_ccnews(split: str) -> pd.DataFrame:
    files = sorted(CCNEWS_DIR.glob(f"*.{split}.jsonl"))
    return pd.concat([pd.read_json(f, lines=True) for f in files if f.stat().st_size > 0], ignore_index=True)


LOADERS = {
    "emmediatopic": lambda: load_emmediatopic(),
    "ccnews_train": lambda: load_ccnews("train"),
    "ccnews_heldout": lambda: load_ccnews("heldout"),
    "ccnews_train_extra": lambda: load_ccnews("train_extra"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", nargs="*", default=list(LOADERS))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    TEACHER_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    teacher = Teacher()
    for name in args.corpora:
        print(f"[{name}]", flush=True)
        label_corpus(teacher, LOADERS[name](), args.batch_size).to_parquet(TEACHER_LABELS_DIR / f"{name}.parquet", index=False)


if __name__ == "__main__":
    main()
