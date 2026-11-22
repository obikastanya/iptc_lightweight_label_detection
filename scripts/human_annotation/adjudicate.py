"""Adjudication of the human annotation (PROTOCOL.md, section Adjudication).

Decision 2026-11-23: for the four items with different main topics the two annotators agreed on Obi's
answers. The script writes these decisions into human_annotation/adjudikasi/adjudikasi.xlsx, builds the
final labels (main topic: the shared answer or the adjudicated one; correct topics: the adjudicated list for
the four items and the union of both annotators' lists elsewhere), and adds the adjudicated estimates to
human_annotation/results.json.

Run after analyze.py:  env/Scripts/python.exe scripts/human_annotation/adjudicate.py
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from analyze import TOP, weighted_rate  # noqa: E402

H = ROOT / "human_annotation"


def main():
    df = pd.read_csv(H / "_key/merged_labels.csv").set_index("item_id")
    for c in ("obi_all", "anan_all"):
        df[c] = df[c].apply(ast.literal_eval)
    dis = df.obi_main != df.anan_main
    df["final_main"] = np.where(dis, df.obi_main, df.obi_main)
    df["final_all"] = [sorted(o) if d else sorted(set(o) | set(a)) for o, a, d in zip(df.obi_all, df.anan_all, dis)]
    df["final_strict"] = df["label"] == df["final_main"]
    df["final_lenient"] = [lab in s for lab, s in zip(df["label"], df["final_all"])]

    num = {t: i + 1 for i, t in enumerate(TOP)}
    wb = load_workbook(H / "adjudikasi/adjudikasi.xlsx")
    ws = wb["Adjudikasi"]
    for row in ws.iter_rows(min_row=2):
        iid = row[0].value
        if iid in df.index:
            row[5].value = f"{num[df.at[iid, 'final_main']]:02d} - {df.at[iid, 'final_main']}"
            row[6].value = "; ".join(str(num[t]) for t in df.at[iid, "final_all"])
    wb.save(H / "adjudikasi/adjudikasi.xlsx")
    (H / "adjudikasi/ADJUDICATION.md").write_text(
        "# Adjudication\n\n2026-11-23: the two annotators discussed the four items with different main topics "
        f"({', '.join(df.index[dis])}) and agreed on Obi's answers (main topic and list of correct topics).\n",
        encoding="utf8")

    res = json.loads((H / "results.json").read_text(encoding="utf8"))
    res["adjudicated"] = {"items_adjudicated": list(df.index[dis])}
    for col in ("final_strict", "final_lenient"):
        est, ci = weighted_rate(df, col)
        res["adjudicated"][col] = {"weighted": est, "ci95": ci, "unweighted": float(df[col].mean()),
                                    "by_source": {s: float(df.loc[df.source == s, col].mean()) for s in sorted(df.source.unique())}}
    (H / "results.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    df[["source", "lang", "label", "final_main", "final_all"]].to_csv(H / "_key/final_labels.csv")
    print(json.dumps(res["adjudicated"], indent=1))


if __name__ == "__main__":
    main()
