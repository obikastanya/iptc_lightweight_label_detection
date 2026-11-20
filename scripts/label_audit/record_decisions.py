import sys, json, pandas as pd
v = pd.read_csv("verify_human.csv", encoding="utf-8-sig", dtype=str)
for item, dec, other, note in json.loads(sys.argv[1]):
    i = v.index[v.item_id == item]; assert len(i) == 1, item
    v.loc[i, "decision"] = dec; v.loc[i, "other_label"] = other or ""; v.loc[i, "note"] = note or ""
v.to_csv("verify_human.csv", index=False, encoding="utf-8-sig")
print("filled", v.decision.fillna("").ne("").sum(), "of", len(v))
