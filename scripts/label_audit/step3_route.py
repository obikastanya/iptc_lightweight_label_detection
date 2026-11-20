"""Steps 5-7: validate adjudication output, decide which items go to the human, and build blind
translation batches for routed non-en/id items. Prints only aggregate counts.

Usage: python step3_route.py --scratch <dir>
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from common import BATCH_DIR, CONF, NO_SPACE_LANGS, OUT, make_batches, norm_label, read_jsonl, write_batch_files, write_jsonl

DECISIONS = ("option_1", "option_2", "both_acceptable", "neither", "unusable")


def load_adjudication(scratch: Path, ids: list[str], single: set[str]):
    rows = []
    for p in sorted((scratch / "out").glob("ADJ_*.jsonl")):
        shutil.copy(p, BATCH_DIR / ("raw_" + p.name))
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    good = {}
    for r in rows:
        iid, d = r.get("item_id"), str(r.get("decision", "")).strip()
        c = str(r.get("confidence", "")).lower()
        if iid not in ids or iid in good or d not in DECISIONS or c not in CONF:
            continue
        if iid in single and d in ("option_2", "both_acceptable"):
            continue
        cl = norm_label(r.get("correct_label")) if d == "neither" else None
        if d == "neither" and cl in (None, "__invalid__"):
            continue
        good[iid] = {"item_id": iid, "decision": d, "correct_label": cl, "confidence": c,
                     "rationale": str(r.get("rationale", ""))}
    return good, [i for i in ids if i not in good]


def excerpt(text: str, lang: str) -> str:
    return text[:1000] if lang in NO_SPACE_LANGS else " ".join(text.split()[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    args = ap.parse_args()
    scratch = Path(args.scratch)
    akey = pd.read_csv(OUT / "adjudication_key.csv", dtype=str).fillna("")
    ids = akey["item_id"].tolist()
    single = set(akey.loc[akey["option_2"] == "", "item_id"])
    good, missing = load_adjudication(scratch, ids, single)
    print(f"adjudication valid {len(good)}/{len(ids)}; re-run: {missing}")
    if missing:
        return
    write_jsonl(OUT / "adjudication.jsonl", [good[i] for i in ids])

    adj = pd.DataFrame([good[i] for i in ids])
    print("decision x confidence:\n", pd.crosstab(adj["decision"], adj["confidence"]))
    routed_adj = adj[(adj["confidence"] != "high") | adj["decision"].isin(["neither", "unusable"])]["item_id"]

    key = pd.read_csv(OUT / "sample_key.csv", dtype=str).set_index("item_id")
    b = pd.DataFrame(read_jsonl(OUT / "llm_labels_B.jsonl")).set_index("item_id")
    agreed = key.loc[b.index, "label"]
    confirm = (b["primary"] == agreed) | (b["secondary"] == agreed)
    routed_spot = confirm[~confirm].index
    print(f"spot-check: {int(confirm.sum())}/{len(confirm)} confirmed by B")

    route = [{"item_id": i, "reason": "adjudication",
              "task": "label_from_scratch" if i in single else "choose_between"} for i in routed_adj]
    route += [{"item_id": i, "reason": "spotcheck", "task": "label_from_scratch"} for i in routed_spot]
    route = sorted(route, key=lambda r: r["item_id"])
    pd.DataFrame(route).to_csv(OUT / "human_routing.csv", index=False, encoding="utf-8")
    print("routed to human:", len(route), "| adjudication:", len(routed_adj), "| spotcheck:", len(routed_spot))

    blind = {r["item_id"]: r for r in read_jsonl(OUT / "blind_input.jsonl")}
    tr = [{"item_id": r["item_id"], "lang": blind[r["item_id"]]["lang"],
           "headline": blind[r["item_id"]]["headline"],
           "text": excerpt(blind[r["item_id"]]["text"], blind[r["item_id"]]["lang"])}
          for r in route if blind[r["item_id"]]["lang"] not in ("en", "id")]
    write_jsonl(OUT / "translation_input.jsonl", [{"item_id": t["item_id"]} for t in tr])
    if tr:
        batches = make_batches(tr, max_items=50, max_chars=40_000)
        for p, bb in zip(write_batch_files("TR", batches, scratch), batches):
            print(p.name, len(bb))
    print("to translate:", len(tr))


if __name__ == "__main__":
    main()
