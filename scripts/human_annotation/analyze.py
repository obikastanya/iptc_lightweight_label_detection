"""Analysis of the independent human annotation (plan in human_annotation/PROTOCOL.md).

1. Records the submitted workbooks: copies them to human_annotation/submitted/ with SHA-256 hashes.
2. Agreement between the two annotators on the main topic (Cohen's kappa with 95% bootstrap CI,
   Krippendorff's alpha, percent agreement), computed before any discussion.
3. Correct rate of the editorial labels per annotator, strict (= main topic) and lenient (in the set of all
   correct topics), weighted by each source's share of the 8,650 editorial labels, with 95% intervals.
4. Agreement of each annotator with the blind LLM labels.
5. Writes the adjudication workbook with the disagreements (no editorial or LLM labels shown).

Run from the project root:  env/Scripts/python.exe scripts/human_annotation/analyze.py
"""
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "human_annotation"
POP = {"ccnews_jsonld": 6050, "l3cube_indic": 2361, "bangla_plos": 139, "indonews": 100}
N_BOOT = 2000
rng = np.random.default_rng(0)


def topics():
    ref = pd.read_csv(ROOT / "results/label_audit/iptc_labels_reference.csv", encoding="utf-8-sig")
    return list(ref.label)


TOP = topics()


def read_sheet(path):
    ws = load_workbook(path, data_only=True)["Anotasi"]
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if not r[1]:
            continue
        main = TOP[int(str(r[3]).split(" - ")[0]) - 1]
        nums = [int(x) for x in str(r[4] or "").replace(",", ";").split(";") if x.strip()]
        allc = {TOP[n - 1] for n in nums} | {main}
        rows.append({"item_id": r[1], "main": main, "all": sorted(allc), "main_in_list": main in {TOP[n - 1] for n in nums}})
    return pd.DataFrame(rows).set_index("item_id")


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    cats = np.union1d(a, b)
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def alpha_nominal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    vals = np.concatenate([a, b])
    n = len(vals)
    cats, counts = np.unique(vals, return_counts=True)
    do = (a != b).mean()
    de = 1 - ((counts * (counts - 1)).sum() / (n * (n - 1)))
    return 1 - do / de


def boot(fn, *cols):
    n = len(cols[0])
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        vals.append(fn(*[np.asarray(c)[i] for c in cols]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def weighted_rate(df, col, strata="source"):
    w = {s: POP[s] / sum(POP.values()) for s in POP}
    est = sum(w[s] * df.loc[df[strata] == s, col].mean() for s in POP)
    vals = []
    for _ in range(N_BOOT):
        parts = []
        for s in POP:
            g = df.loc[df[strata] == s, col].to_numpy()
            parts.append(w[s] * g[rng.integers(0, len(g), len(g))].mean())
        vals.append(sum(parts))
    return float(est), [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def record_submission():
    sub = H / "submitted"
    sub.mkdir(exist_ok=True)
    lines = ["# Submitted annotation workbooks", "", "| File | Last modified | SHA-256 |", "|---|---|---|"]
    for who in ("obi", "anan"):
        src = H / who / f"annotasi_{who}.xlsx"
        dst = sub / src.name
        shutil.copy2(src, dst)
        digest = hashlib.sha256(dst.read_bytes()).hexdigest()
        mtime = datetime.fromtimestamp(src.stat().st_mtime).isoformat(timespec="seconds")
        lines.append(f"| {src.name} | {mtime} | `{digest}` |")
    lines += ["", f"Recorded {datetime.now().isoformat(timespec='seconds')} before the annotators compared their answers."]
    (sub / "SUBMISSION.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    record_submission()
    key = pd.read_csv(H / "_key/sample_key.csv").set_index("item_id")
    ai = {json.loads(l)["item_id"]: json.loads(l) for l in open(H / "_work/ai_labels.jsonl", encoding="utf8")}
    o, a = read_sheet(H / "submitted/annotasi_obi.xlsx"), read_sheet(H / "submitted/annotasi_anan.xlsx")
    df = key.join(o.add_prefix("obi_")).join(a.add_prefix("anan_"))
    df["ai_main"] = [ai[i]["main_topic"] for i in df.index]
    df["ai_all"] = [sorted(ai[i]["all_correct"]) for i in df.index]
    for who in ("obi", "anan"):
        df[f"{who}_strict"] = df["label"] == df[f"{who}_main"]
        df[f"{who}_lenient"] = [lab in s for lab, s in zip(df["label"], df[f"{who}_all"])]
    df["ai_strict"] = df["label"] == df["ai_main"]
    df["ai_lenient"] = [lab in s for lab, s in zip(df["label"], df["ai_all"])]
    df["both_strict"] = df.obi_strict & df.anan_strict
    df["any_lenient"] = df.obi_lenient | df.anan_lenient

    res = {"n": len(df)}
    om, am = df.obi_main.to_numpy(), df.anan_main.to_numpy()
    res["human_human"] = {"percent_agreement": float((om == am).mean()), "kappa": float(kappa(om, am)),
                          "kappa_ci95": boot(kappa, om, am), "krippendorff_alpha": float(alpha_nominal(om, am)),
                          "lenient_overlap": float(np.mean([bool(set(x) & set(y)) for x, y in zip(df.obi_all, df.anan_all)]))}
    for who in ("obi", "anan"):
        res[f"{who}_vs_ai"] = {"percent_agreement": float((df[f"{who}_main"] == df.ai_main).mean()),
                                "kappa": float(kappa(df[f"{who}_main"], df.ai_main)),
                                "kappa_ci95": boot(kappa, df[f"{who}_main"].to_numpy(), df.ai_main.to_numpy())}
        res[f"{who}_vs_editorial_kappa"] = float(kappa(df[f"{who}_main"], df.label))
    res["ai_vs_editorial_kappa"] = float(kappa(df.ai_main, df.label))
    rates = {}
    for col in ("obi_strict", "obi_lenient", "anan_strict", "anan_lenient", "both_strict", "any_lenient", "ai_strict", "ai_lenient"):
        est, ci = weighted_rate(df, col)
        rates[col] = {"weighted": est, "ci95": ci, "unweighted": float(df[col].mean()),
                      "by_source": {s: float(df.loc[df.source == s, col].mean()) for s in POP}}
    res["editorial_correct_rate"] = rates
    res["main_not_in_list"] = {"obi": int((~df.obi_main_in_list).sum()), "anan": int((~df.anan_main_in_list).sum())}
    res["extension_rule_triggered"] = res["human_human"]["kappa"] < 0.65
    (H / "results.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    df.to_csv(H / "_key/merged_labels.csv")

    # adjudication workbook: disagreements on the main topic, no editorial or LLM label shown
    dis = df[df.obi_main != df.anan_main]
    inp = {json.loads(l)["item_id"]: json.loads(l) for l in open(H / "_work/blind_input.jsonl", encoding="utf8")}
    tr = {}
    for f in sorted((H / "_work").glob("translations_*.jsonl")):
        for l in open(f, encoding="utf8"):
            if l.strip():
                d = json.loads(l)
                tr[d["item_id"]] = d
    wb = Workbook()
    ws = wb.active
    ws.title = "Adjudikasi"
    ws.append(["Item ID", "Jawaban Obi (utama)", "Jawaban Obi (semua)", "Jawaban Anan (utama)", "Jawaban Anan (semua)",
               "Topik utama final (isi)", "Semua topik benar final (isi: nomor)", "Judul (Inggris)", "Teks (Inggris)"])
    num = {t: i + 1 for i, t in enumerate(TOP)}
    for i, r in dis.iterrows():
        ws.append([i, f"{num[r.obi_main]:02d} - {r.obi_main}", "; ".join(str(num[t]) for t in r.obi_all),
                   f"{num[r.anan_main]:02d} - {r.anan_main}", "; ".join(str(num[t]) for t in r.anan_all), None, None,
                   tr[i]["title_en"], tr[i]["text_en"]])
    for c, w in zip("ABCDEFGHI", (8, 30, 14, 30, 14, 34, 18, 28, 80)):
        ws.column_dimensions[c].width = w
    for cell in ws[1]:
        cell.font, cell.fill = Font(bold=True), PatternFill("solid", fgColor="DDEBF7")
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[row[0].row].height = 140
    ts = wb.create_sheet("Topik IPTC")
    ts.append(["No", "Topik"])
    for t in TOP:
        ts.append([num[t], f"{num[t]:02d} - {t}"])
    dv = DataValidation(type="list", formula1=f"='Topik IPTC'!$B$2:$B${len(TOP) + 1}", allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"F2:F{len(dis) + 1}")
    ws.freeze_panes = "H2"
    for sheet in wb.worksheets:
        sheet.sheet_view.tabSelected = sheet.title == "Adjudikasi"
    (H / "adjudikasi").mkdir(exist_ok=True)
    wb.save(H / "adjudikasi" / "adjudikasi.xlsx")
    print(json.dumps(res, indent=1))
    print("disagreements", len(dis))


if __name__ == "__main__":
    main()
