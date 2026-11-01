"""Paired comparison of StaticKD variants against a reference model.

Reads results/accuracy/<name>.json (written by evaluate_accuracy.py) and reports, per evaluation
set, macro-F1 (dev, llm_test) or agreement with the teacher (ccnews held-out), the difference
to the reference with a paired-bootstrap 95% CI and two-sided p-value.

Usage:
    python evaluation/compare_variants.py --ref static_final_table --models static_bigram21_table ...
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABEL2ID, RESULTS_DIR  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from data.emmediatopic import load_emmediatopic  # noqa: E402
from evaluation.evaluate_accuracy import LLM_TEST, TEACHER_CCNEWS_HELDOUT  # noqa: E402


def gold() -> dict[str, np.ndarray]:
    return {
        "dev": load_emmediatopic("dev")["label_id"].to_numpy(),
        "ccnews": pd.read_parquet(TEACHER_CCNEWS_HELDOUT)["teacher_label_id"].to_numpy(),
        "llm_test": pd.read_json(LLM_TEST, lines=True)["label_id"].to_numpy(),
    }


def score(y, p, which):
    return float((y == p).mean()) if which == "ccnews" else f1_score(y, p, average="macro", labels=np.unique(y),
                                                                             zero_division=0)


def load_preds(name: str, which: str) -> np.ndarray:
    res = json.loads((RESULTS_DIR / "accuracy" / f"{name}.json").read_text(encoding="utf8"))
    return np.array([LABEL2ID[l] for l in res[which]["predictions"]])


def paired_bootstrap(y, a, b, which, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = score(y[idx], b[idx], which) - score(y[idx], a[idx], which)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return np.percentile(diffs, [2.5, 97.5]), min(p, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="static_final_table")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    labels = gold()
    rows = []
    for which in ("dev", "llm_test", "ccnews"):
        y = labels[which]
        ref = load_preds(args.ref, which)
        ref_score = score(y, ref, which)
        rows.append({"set": which, "model": args.ref, "score": ref_score})
        # ccnews has 13.5k docs; fewer bootstrap rounds keep it fast and still stable
        n_boot = args.n_boot if which != "ccnews" else min(args.n_boot, 500)
        for name in args.models:
            pred = load_preds(name, which)
            (lo, hi), p = paired_bootstrap(y, ref, pred, which, n_boot)
            s = score(y, pred, which)
            rows.append({"set": which, "model": name, "score": s, "diff": s - ref_score,
                         "ci_lo": lo, "ci_hi": hi, "p": p})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out = RESULTS_DIR / "variant_comparison.csv"
    df.to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
