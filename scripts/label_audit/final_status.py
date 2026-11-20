"""Final editorial-label accuracy after the post-hoc recheck and the author's decisions on the three
items the recheck flagged (A027, A032, A222). Writes results/label_audit/final_status.csv and final_estimate.json.

    env/Scripts/python.exe scripts/label_audit/final_status.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402
from analyze import bootstrap, build_items, estimate  # noqa: E402
from common import AUDIT_ORIGINS, OUT, TEST_PATH  # noqa: E402

AUTHOR_FINAL = {"A027": "both_acceptable", "A032": "correct", "A222": "both_acceptable"}  # decided 2026-11-19

full = pd.read_json(TEST_PATH, lines=True, dtype={"domain": str, "url": str})
analyze.POP = full[full["label_origin"].isin(AUDIT_ORIGINS)]["source"].value_counts().to_dict()
df = build_items().set_index("item_id")
rc = pd.read_csv(OUT / "recheck_disagreements.csv", encoding="utf-8-sig").set_index("item_id")
df["final"] = df["status"]
df.loc[rc.index, "final"] = rc["recheck_status"]
for k, v in AUTHOR_FINAL.items():
    df.loc[k, "final"] = v
df = df.reset_index()
res = {}
for col in ("status", "final"):
    e = estimate(df, col); e.update(bootstrap(df, col)); res[col] = e
df[["item_id", "source", "status", "final"]].to_csv(OUT / "final_status.csv", index=False)
(OUT / "final_estimate.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf8")
print(df.loc[df.status.isin(["wrong", "both_acceptable", "correct"]) & df.final.notna(), "final"].value_counts().to_dict())
for k, v in res.items():
    print(k, f"strict {v['acc_strict']:.4f} {v['acc_strict_ci']}  lenient {v['acc_lenient']:.4f} {v['acc_lenient_ci']}")
    print("  strata", {s: (round(x['acc_strict'], 3), round(x['acc_lenient'], 3)) for s, x in v["strata"].items()})

# model check against the final verified labels: correct -> editorial label; both -> editorial label plus
# the label it was compared with; wrong -> unchanged verified label
def final_set(r):
    v = list(r["verified"]) if isinstance(r["verified"], list) else []
    if r["final"] == "correct":
        return [r["label"]]
    if r["final"] == "both_acceptable":
        return sorted(set(v) | {r["label"]})
    return v
df["verified"] = df.apply(final_set, axis=1)
df["verified_prov"] = df["verified"]
mc = analyze.model_check(df)
(OUT / "final_model_check.json").write_text(json.dumps(mc, indent=1, default=float), encoding="utf8")
for m, v in mc.items():
    print(f"{m:28s} vs section {v['acc_vs_section']:.3f}  vs final verified {v['acc_vs_verified']:.3f}  n {v['n_verified']}")
