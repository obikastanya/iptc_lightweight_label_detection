"""Steps 1-3 setup: stratified sample, answer key (written, never printed), blind input, A batches.

Usage: python scripts/label_audit/step1_sample.py --scratch <dir outside the repo>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import AUDIT_ORIGINS, OUT, STRATA, TEST_PATH, make_batches, truncate, write_batch_files, write_jsonl

SEED = 20261118


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    args = ap.parse_args()
    df = pd.read_json(TEST_PATH, lines=True, dtype={"domain": str, "url": str})
    pop = df[df["label_origin"].isin(AUDIT_ORIGINS)]
    rng = np.random.default_rng(SEED)
    parts = []
    for src, n in STRATA.items():
        s = pop[pop["source"] == src].sort_values("doc_id").reset_index(drop=True)
        idx = rng.choice(len(s), size=n, replace=False)
        parts.append(s.iloc[np.sort(idx)])
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    sample["item_id"] = [f"A{i:03d}" for i in range(1, len(sample) + 1)]

    OUT.mkdir(parents=True, exist_ok=True)
    sample[["item_id", "doc_id", "source", "lang", "label", "source_label", "url"]].to_csv(
        OUT / "sample_key.csv", index=False, encoding="utf-8")

    blind = [{"item_id": r.item_id, "lang": r.lang, "headline": r.headline or "",
              "text": truncate(r.text, r.lang)} for r in sample.itertuples()]
    write_jsonl(OUT / "blind_input.jsonl", blind)
    batches = make_batches(blind)
    paths = write_batch_files("A", batches, Path(args.scratch))
    for p, b in zip(paths, batches):
        print(p.name, len(b), b[0]["item_id"], b[-1]["item_id"], sum(len(r["text"]) for r in b))
    print("sampled", len(sample), "| per source:", sample["source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
