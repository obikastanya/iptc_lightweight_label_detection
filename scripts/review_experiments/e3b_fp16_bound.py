"""E3b: how much can fp16 storage of the folded table change the StaticKD test predictions?

The per-seed tables T_k = E_k W_k^T were written in fp16 when each seed was saved and the fp32 products
were not kept, so an exact fp32-vs-fp16 comparison needs retraining. This script bounds the effect
without retraining:

1. Ensemble rounding (measured exactly): the three seed tables are averaged in fp32 and the resulting
   logits are compared with those of the released fp16 table.
2. Per-seed rounding (bounded): rounding to fp16 moves every entry by at most u|T_k| with u = 2^-11.
   The averaged logit of class c therefore moves by at most u * mean over its tokens of
   mean_k |T_k[t, c]| (plus the measured ensemble rounding). An argmax can only change where the gap
   between the top logit and another logit is below the sum of their bounds; the script counts these articles.

Usage (from the project root):
    env/Scripts/python.exe scripts/review_experiments/e3b_fp16_bound.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from statickd import evaluation as ev  # noqa: E402
from statickd.student import StaticKDModel, encode  # noqa: E402

FINAL = ROOT / "models" / "final" / "statickd"
SEEDS = [ROOT / "models" / "students" / f"full_s{i}" for i in range(3)]
OUT = ROOT / "results" / "review_experiments" / "e3b_fp16_bound.json"
U = 2.0 ** -11   # unit roundoff of IEEE half precision


def main() -> None:
    model = StaticKDModel(FINAL)
    seed_tables = [np.load(d / "model.npz")["token_table"].astype(np.float32) for d in SEEDS]
    seed_bias = np.mean([np.load(d / "model.npz")["bias"] for d in SEEDS], axis=0)
    t32 = np.mean(seed_tables, axis=0)                       # ensemble averaged in fp32
    abs_mean = np.mean([np.abs(t) for t in seed_tables], axis=0)
    t16 = model.table.astype(np.float32)                      # released fp16 table

    texts = ev.eval_set("test")["text"].tolist()
    ids = encode(model.tokenizer, texts, model.max_tokens)
    l16 = np.stack([t16[i].mean(axis=0) for i in ids]) + model.bias
    l32 = np.stack([t32[i].mean(axis=0) for i in ids]) + seed_bias
    stored = np.load(ev.prediction_path("statickd", "test"))["logits"]

    ens_diff = np.abs(l16 - l32).max(axis=1)
    # per-article, per-class bound on the shift of logit c: u * mean over tokens of mean_k |T_k[t, c]| (+ ensemble rounding)
    bound_c = np.stack([U * abs_mean[i].mean(axis=0) for i in ids]) + ens_diff[:, None]
    top = l16.argmax(axis=1)
    rows = np.arange(len(top))
    gap = l16[rows, top][:, None] - l16                       # how far each class is below the top class
    can_flip = (gap < bound_c[rows, top][:, None] + bound_c) & (gap > 0)
    bound = bound_c.max(axis=1)
    res = {
        "n_docs": len(texts),
        "stored_equals_fp16_recomputed_argmax": int((stored.argmax(1) == l16.argmax(1)).sum()),
        "ensemble_rounding_argmax_changes": int((l16.argmax(1) != l32.argmax(1)).sum()),
        "ensemble_rounding_max_abs_logit_diff": float(ens_diff.max()),
        "per_seed_bound_max": float(bound.max()),
        "per_seed_bound_median": float(np.median(bound)),
        "docs_whose_argmax_could_change": int(can_flip.any(axis=1).sum()),
        "unit_roundoff": U,
    }
    OUT.write_text(json.dumps(res, indent=1), encoding="utf8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
