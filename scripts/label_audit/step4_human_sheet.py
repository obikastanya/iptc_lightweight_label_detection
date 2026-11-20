"""Step 7: write verify_human.csv (UTF-8 with BOM) and iptc_labels_reference.csv.

Usage: python step4_human_sheet.py --scratch <dir>
Does not overwrite verify_human.csv if it already holds human decisions (use --force).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common import BATCH_DIR, NO_SPACE_LANGS, OUT, definitions, read_jsonl

SEED_HUMAN = 20261121


def excerpt(text: str, lang: str) -> str:
    return text[:1000] if lang in NO_SPACE_LANGS else " ".join(text.split()[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    target = OUT / "verify_human.csv"
    if target.exists() and not args.force:
        old = pd.read_csv(target, dtype=str, encoding="utf-8-sig").fillna("")
        if (old["decision"] != "").any():
            raise SystemExit("verify_human.csv already has human decisions; refusing to overwrite")

    tr = {}
    for p in sorted((Path(args.scratch) / "out").glob("TR_*.jsonl")):
        shutil.copy(p, BATCH_DIR / ("raw_" + p.name))
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("item_id") and r.get("text_en"):
                    tr[r["item_id"]] = r
    need = [r["item_id"] for r in read_jsonl(OUT / "translation_input.jsonl")]
    missing = [i for i in need if i not in tr]
    if missing:
        raise SystemExit(f"translations missing for {missing}")

    route = pd.read_csv(OUT / "human_routing.csv", dtype=str)
    akey = pd.read_csv(OUT / "adjudication_key.csv", dtype=str).fillna("").set_index("item_id")
    blind = {r["item_id"]: r for r in read_jsonl(OUT / "blind_input.jsonl")}
    rng = np.random.default_rng(SEED_HUMAN)
    rows, hkey = [], []
    for r in route.itertuples():
        b = blind[r.item_id]
        if r.item_id in tr:
            h, t = tr[r.item_id].get("headline_en", ""), tr[r.item_id]["text_en"]
        else:
            h, t = b["headline"], excerpt(b["text"], b["lang"])
        oa = ob = ""
        if r.task == "choose_between":
            k = akey.loc[r.item_id]
            cand = {k.option_1_origin: k.option_1, k.option_2_origin: k.option_2}
            order = ["section", "llm_A"] if rng.integers(0, 2) == 0 else ["llm_A", "section"]
            oa, ob = cand[order[0]], cand[order[1]]
            hkey.append({"item_id": r.item_id, "option_A_origin": order[0], "option_B_origin": order[1]})
        rows.append({"item_id": r.item_id, "task": r.task, "lang": b["lang"], "headline_en": h, "text_en": t,
                     "option_A": oa, "option_B": ob, "decision": "", "other_label": "", "note": ""})
    pd.DataFrame(rows).to_csv(target, index=False, encoding="utf-8-sig")
    pd.DataFrame(hkey).to_csv(OUT / "human_options_key.csv", index=False, encoding="utf-8")
    pd.DataFrame([{"label": k, "official_definition": v} for k, v in definitions().items()]).to_csv(
        OUT / "iptc_labels_reference.csv", index=False, encoding="utf-8-sig")
    print("verify_human rows:", len(rows), pd.Series([x["task"] for x in rows]).value_counts().to_dict())


if __name__ == "__main__":
    main()
