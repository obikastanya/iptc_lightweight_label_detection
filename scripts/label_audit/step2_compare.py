"""Steps 4-6 setup: compare A with the section label, build blind adjudication batches and the
spot-check sample for Annotator B. Prints only aggregate counts.

Usage: python step2_compare.py --scratch <dir>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT, make_batches, read_jsonl, write_batch_files, write_jsonl

SEED_ADJ, SEED_SPOT, N_SPOT, MIN_SPOT = 20261119, 20261120, 60, 5


def spot_allocation(counts: pd.Series) -> dict:
    total = counts.sum()
    raw = counts / total * N_SPOT
    alloc = np.floor(raw).astype(int)
    rem = (raw - alloc).sort_values(ascending=False)
    for src in rem.index[: N_SPOT - alloc.sum()]:
        alloc[src] += 1
    for src in alloc.index:
        if counts[src] >= MIN_SPOT and alloc[src] < MIN_SPOT:
            need = MIN_SPOT - alloc[src]
            alloc[src] += need
            alloc[alloc.idxmax()] -= need
    return alloc.to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    args = ap.parse_args()
    key = pd.read_csv(OUT / "sample_key.csv", dtype=str)
    a = pd.DataFrame(read_jsonl(OUT / "llm_labels_A.jsonl"))
    blind = {r["item_id"]: r for r in read_jsonl(OUT / "blind_input.jsonl")}
    df = key.merge(a, on="item_id", how="left").sort_values("item_id").reset_index(drop=True)
    df["agree"] = (df["primary"] == df["label"]) & (~df["unusable"].astype(bool))
    df["lenient"] = (df["primary"] == df["label"]) | (df["secondary"] == df["label"])
    print(df.groupby("source").agg(n=("item_id", "size"), agree=("agree", "sum"),
                                   lenient=("lenient", "sum"), unusable=("unusable", "sum")))

    # adjudication: section label vs A.primary, random order, origin hidden
    rng = np.random.default_rng(SEED_ADJ)
    dis = df[~df["agree"]]
    adj_key, adj_items = [], []
    for r in dis.itertuples():
        flip = bool(rng.integers(0, 2))
        if r.primary == r.label:
            o1, o2, g1, g2 = r.label, "", "both", ""
        elif flip:
            o1, o2, g1, g2 = r.primary, r.label, "llm_A", "section"
        else:
            o1, o2, g1, g2 = r.label, r.primary, "section", "llm_A"
        adj_key.append({"item_id": r.item_id, "option_1": o1, "option_2": o2,
                        "option_1_origin": g1, "option_2_origin": g2})
        adj_items.append({**blind[r.item_id], "option_1": o1, "option_2": o2})
    pd.DataFrame(adj_key).to_csv(OUT / "adjudication_key.csv", index=False, encoding="utf-8")
    write_jsonl(OUT / "adjudication_input.jsonl", [{"item_id": x["item_id"]} for x in adj_items])
    extra = (lambda x: {"OPTION_1": x["option_1"], "OPTION_2": x["option_2"] or "(none)"})
    batches = make_batches(adj_items)
    for p, b in zip(write_batch_files("ADJ", batches, Path(args.scratch), extra), batches):
        print(p.name, len(b))

    # spot-check of agreed items
    agreed = df[df["agree"]]
    alloc = spot_allocation(agreed["source"].value_counts())
    rng = np.random.default_rng(SEED_SPOT)
    picks = []
    for src in sorted(alloc):
        pool = agreed[agreed["source"] == src].sort_values("item_id")
        idx = rng.choice(len(pool), size=min(alloc[src], len(pool)), replace=False)
        picks.extend(pool.iloc[np.sort(idx)]["item_id"].tolist())
    picks = sorted(picks)
    write_jsonl(OUT / "spotcheck_input.jsonl", [{"item_id": i} for i in picks])
    batches = make_batches([blind[i] for i in picks])
    for p, b in zip(write_batch_files("B", batches, Path(args.scratch)), batches):
        print(p.name, len(b))
    print("disagree:", len(dis), "| single-option:", sum(k["option_2"] == "" for k in adj_key),
          "| spot allocation:", alloc)


if __name__ == "__main__":
    main()
