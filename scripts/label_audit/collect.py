"""Validate raw annotator outputs (<prefix>_NN.jsonl in the scratch out dir) and merge them.

Usage: python collect.py --scratch <dir> --prefix A --ids-from blind_input.jsonl --out llm_labels_A.jsonl
Also copies the raw outputs into results/label_audit/batches/. Prints only counts and the ids to re-run.
"""
import argparse
import json
import shutil
from pathlib import Path

from common import BATCH_DIR, OUT, read_jsonl, validate_labels, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--ids-from", required=True, help="jsonl in results/label_audit with item_id per line")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ids = [r["item_id"] for r in read_jsonl(OUT / args.ids_from)]
    rows = []
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted((Path(args.scratch) / "out").glob(f"{args.prefix}_*.jsonl")):
        shutil.copy(p, BATCH_DIR / ("raw_" + p.name))
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print("bad json line in", p.name)
    good, missing = validate_labels(rows, ids)
    write_jsonl(OUT / args.out, [good[i] for i in ids if i in good])
    print(f"valid {len(good)}/{len(ids)}; re-run: {missing}")


if __name__ == "__main__":
    main()
