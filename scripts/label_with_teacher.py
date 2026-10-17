"""Label the distillation corpora with the teacher's 17 logits (offline, once; GPU recommended).

Outputs data/processed/teacher_labels/{emmediatopic,ccnews_train,ccnews_train_extra,ccnews_heldout}.parquet

Usage (from the project root):
    python scripts/label_with_teacher.py --corpora emmediatopic ccnews_train ccnews_heldout ccnews_train_extra

Timing only (the stored labels are not touched): label a fixed random sample of the distillation pool,
check that the logits match the stored ones, and write results/final/teacher_labeling_time.json:
    python scripts/label_with_teacher.py --time-sample 5000
"""
import argparse
import json
import sys
import time
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


def time_sample(teacher: Teacher, n: int, batch_size: int) -> None:
    """Measure labelling throughput on a fixed sample of the distillation pool (the stored labels stay)."""
    import numpy as np
    import torch

    from statickd.config import FINAL_DIR
    from statickd.data import load_distillation_pool

    pool = load_distillation_pool()
    sample = pool.sample(n=n, random_state=0)
    texts = sample["text"].tolist()
    teacher.logits(texts[:64], batch_size=batch_size)  # warm-up (CUDA kernels, tokenizer)
    if teacher.device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    logits = teacher.logits(texts, batch_size=batch_size)
    if teacher.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    stored = np.stack(sample["teacher_logits"].to_numpy())
    result = {
        "n_docs": n, "seconds": round(elapsed, 2), "docs_per_sec": round(n / elapsed, 2),
        "pool_docs": len(pool), "estimated_pool_hours": round(len(pool) / (n / elapsed) / 3600, 2),
        "device": torch.cuda.get_device_name(0) if teacher.device == "cuda" else "cpu",
        "fp16": teacher.device == "cuda", "batch_size": batch_size,
        "argmax_agreement_with_stored": float((logits.argmax(1) == stored.argmax(1)).mean()),
        "max_abs_logit_diff": float(np.abs(logits - stored).max()),
    }
    print(json.dumps(result), flush=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "teacher_labeling_time.json").write_text(json.dumps(result, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", nargs="*", default=list(LOADERS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--time-sample", type=int, default=0, help="timing run on N pool documents only")
    args = parser.parse_args()
    teacher = Teacher()
    if args.time_sample:
        time_sample(teacher, args.time_sample, args.batch_size)
        return
    TEACHER_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.corpora:
        print(f"[{name}]", flush=True)
        label_corpus(teacher, LOADERS[name](), args.batch_size).to_parquet(TEACHER_LABELS_DIR / f"{name}.parquet", index=False)


if __name__ == "__main__":
    main()
