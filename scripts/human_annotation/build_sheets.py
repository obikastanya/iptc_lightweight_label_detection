"""Build the independent human-annotation sheets (protocol: human_annotation/PROTOCOL.md).

Step 1 (default): draw a new stratified sample of editorial-label test articles that the label audit did
not use, write one blind annotation workbook per annotator (own random order), a shared calibration
workbook, the blind input for the LLM labels, and the answer key (editorial labels, kept out of the
annotators' folders).
Step 2 (--ai): turn the LLM labels (human_annotation/_work/ai_labels.jsonl) into
human_annotation/ai_reference/ai_labels.xlsx.

Run from the project root:
    env/Scripts/python.exe scripts/human_annotation/build_sheets.py
    env/Scripts/python.exe scripts/human_annotation/build_sheets.py --ai
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from statickd import evaluation as ev  # noqa: E402

OUT = ROOT / "human_annotation"
SEED = 20261123
STRATA = {"ccnews_jsonld": 67, "l3cube_indic": 23, "bangla_plos": 5, "indonews": 5}   # audit proportions, 100 articles
CALIB = {"ccnews_jsonld": 6, "l3cube_indic": 2, "bangla_plos": 1, "indonews": 1}
LANG = dict(en="English", es="Spanish", ru="Russian", tr="Turkish", it="Italian", pt="Portuguese", fr="French",
            hi="Hindi", de="German", vi="Vietnamese", el="Greek", pl="Polish", zh="Chinese", ar="Arabic",
            sw="Swahili", id="Indonesian", uk="Ukrainian", fa="Persian", ko="Korean", bg="Bulgarian", ro="Romanian",
            th="Thai", nl="Dutch", ca="Catalan", cs="Czech", mr="Marathi", ta="Tamil", or_="Odia", te="Telugu",
            kn="Kannada", so="Somali", am="Amharic", ml="Malayalam", gu="Gujarati", pa="Punjabi", ha="Hausa",
            ig="Igbo", lg="Luganda", rn="Kirundi", bn="Bengali", ln="Lingala", pcm="Nigerian Pidgin", sn="Shona",
            ti="Tigrinya", xh="isiXhosa", yo="Yoruba", om="Oromo")
LANG["or"] = LANG.pop("or_")
HEAD = PatternFill("solid", fgColor="DDEBF7")
WRAP = Alignment(wrap_text=True, vertical="top")


def topics():
    ref = pd.read_csv(ROOT / "results/label_audit/iptc_labels_reference.csv", encoding="utf-8-sig")
    return [(i + 1, r.label, r.official_definition) for i, r in enumerate(ref.itertuples())]


def topic_sheet(wb, tops):
    ws = wb.create_sheet("Topik IPTC")
    ws.append(["No", "Topik (pilihan di kolom Topik utama)", "Definisi resmi IPTC"])
    for no, lab, d in tops:
        ws.append([no, f"{no:02d} - {lab}", d])
    for c, w in zip("ABC", (5, 48, 110)):
        ws.column_dimensions[c].width = w
    for cell in ws[1]:
        cell.font, cell.fill = Font(bold=True), HEAD
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = WRAP
    return ws


GUIDE = [
    "Petunjuk anotasi (baca sebelum mulai)",
    "",
    "1. Baca judul dan teks artikel. Kolom 'Judul (Inggris)' dan 'Teks (Inggris)' berisi terjemahan mesin sebagai bantuan membaca; teks asli tetap menjadi acuan. Bila terjemahan meragukan, cek teks asli atau Google Translate. Jangan meminta ChatGPT, Claude, Gemini, atau LLM lain untuk menentukan topik.",
    "2. Kolom 'Topik utama': pilih SATU topik IPTC yang paling banyak dibahas artikel (daftar dan definisi ada di sheet 'Topik IPTC').",
    "3. Kolom 'Semua topik yang benar': tulis NOMOR semua topik yang juga benar untuk artikel ini, termasuk topik utama, dipisah titik koma. Contoh: 3; 12. Kalau hanya satu topik yang benar, tulis nomor topik utama saja.",
    "4. Bila teks rusak atau terpotong, tentukan topik sebisanya dari judul dan bagian teks yang terbaca.",
    "",
    "Aturan independensi:",
    "- Kerjakan sendiri. Jangan berdiskusi dengan annotator lain sampai keduanya selesai dan menyerahkan file.",
    "- Jangan membuka folder ai_reference atau _key, dan jangan mencari URL atau sumber artikel.",
    "- Urutan artikel sengaja diacak dan berbeda untuk setiap annotator.",
    "- Setelah selesai, simpan file dan serahkan apa adanya. Jangan mengubah jawaban setelah melihat jawaban orang lain.",
]


def guide_sheet(wb):
    ws = wb.create_sheet("Petunjuk", 0)
    for line in GUIDE:
        ws.append([line])
    ws.column_dimensions["A"].width = 140
    ws["A1"].font = Font(bold=True, size=13)
    for row in ws.iter_rows():
        row[0].alignment = Alignment(wrap_text=True, vertical="top")


def annotation_book(rows, path, tops, title):
    wb = Workbook()
    ws = wb.active
    ws.title = "Anotasi"
    cols = ["No", "Item ID", "Bahasa", "Topik utama (isi)", "Semua topik yang benar (isi: nomor, pisahkan dengan ;)",
            "Judul (Inggris)", "Teks (Inggris)", "Judul (asli)", "Teks asli (maks. 512 kata pertama)"]
    ws.append(cols)
    for i, r in enumerate(rows, 1):
        ws.append([i, r["item_id"], f"{LANG.get(r['lang'], r['lang'])} ({r['lang']})", None, None,
                   r.get("title_en", ""), r.get("text_en", ""), r["title"], r["text"]])
    widths = (5, 8, 14, 34, 20, 28, 75, 28, 55)
    for c, w in zip("ABCDEFGHI", widths):
        ws.column_dimensions[c].width = w
    for cell in ws[1]:
        cell.font, cell.fill, cell.alignment = Font(bold=True), HEAD, WRAP
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = WRAP
        ws.row_dimensions[row[0].row].height = 160
    answer = PatternFill("solid", fgColor="FFF2CC")
    for row in ws.iter_rows(min_row=2, min_col=4, max_col=5):
        for cell in row:
            cell.fill = answer
    ws.freeze_panes = "F2"
    topic_sheet(wb, tops)
    guide_sheet(wb)
    n = len(rows) + 1
    dv = DataValidation(type="list", formula1=f"='Topik IPTC'!$B$2:$B${len(tops) + 1}", allow_blank=True)
    dv.error, dv.errorTitle = "Pilih salah satu topik dari daftar.", "Topik tidak valid"
    ws.add_data_validation(dv)
    dv.add(f"D2:D{n}")
    wb.properties.title = title
    for sheet in wb.worksheets:   # exactly one selected tab, otherwise Excel opens the sheets as a group
        sheet.sheet_view.tabSelected = sheet.title == "Anotasi"
    wb.active = wb.sheetnames.index("Anotasi")
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def step1():
    test = ev.eval_set("test")
    audit = set(pd.read_csv(ROOT / "results/label_audit/sample_key.csv").doc_id)
    pop = test[(test.label_origin != "human_annotation") & ~test.doc_id.isin(audit)]
    rng = np.random.default_rng(SEED)
    picked, calib = [], []
    for src, k in STRATA.items():
        g = pop[pop.source == src]
        idx = rng.permutation(len(g))
        picked.append(g.iloc[idx[:k]])
        calib.append(g.iloc[idx[k:k + CALIB[src]]])
    sample = pd.concat(picked).reset_index(drop=True)
    cal = pd.concat(calib).reset_index(drop=True)
    sample["item_id"] = [f"H{i:03d}" for i in range(1, len(sample) + 1)]
    cal["item_id"] = [f"C{i:02d}" for i in range(1, len(cal) + 1)]
    for df in (sample, cal):
        df["title"] = df["headline"].fillna("")
    tops = topics()
    tr = {}
    for f in sorted((OUT / "_work").glob("translations_*.jsonl")):
        for line in open(f, encoding="utf8"):
            if line.strip():
                d = json.loads(line)
                tr[d["item_id"]] = d
    for df in (sample, cal):
        df["title_en"] = [tr.get(i, {}).get("title_en", "") for i in df["item_id"]]
        df["text_en"] = [tr.get(i, {}).get("text_en", "") for i in df["item_id"]]
    missing = [i for i in list(sample.item_id) + list(cal.item_id) if i not in tr]
    print("translations found", len(tr), "missing", missing[:10], len(missing))
    recs = sample[["item_id", "lang", "title", "text", "title_en", "text_en"]].to_dict("records")
    for name, seed in (("obi", SEED + 1), ("anan", SEED + 2)):
        order = np.random.default_rng(seed).permutation(len(recs))
        annotation_book([recs[i] for i in order], OUT / name / f"annotasi_{name}.xlsx", tops, f"Anotasi {name}")
    annotation_book(cal[["item_id", "lang", "title", "text", "title_en", "text_en"]].to_dict("records"), OUT / "kalibrasi" / "kalibrasi.xlsx", tops, "Kalibrasi")
    key = OUT / "_key"
    key.mkdir(parents=True, exist_ok=True)
    sample[["item_id", "doc_id", "source", "lang", "label", "source_label"]].to_csv(key / "sample_key.csv", index=False)
    cal[["item_id", "doc_id", "source", "lang", "label", "source_label"]].to_csv(key / "calibration_key.csv", index=False)
    work = OUT / "_work"
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "blind_input.jsonl", "w", encoding="utf8") as f:
        for r in recs:
            f.write(json.dumps({"item_id": r["item_id"], "lang": r["lang"], "title": r["title"], "text": r["text"]}, ensure_ascii=False) + "\n")
    print("sample", sample.source.value_counts().to_dict(), "calibration", len(cal), "languages", sample.lang.nunique())


def step2():
    tops = {lab: no for no, lab, _ in topics()}
    rows = [json.loads(l) for l in open(OUT / "_work/ai_labels.jsonl", encoding="utf8")]
    inp = {json.loads(l)["item_id"]: json.loads(l) for l in open(OUT / "_work/blind_input.jsonl", encoding="utf8")}
    wb = Workbook()
    ws = wb.active
    ws.title = "Jawaban AI"
    ws.append(["Item ID", "Bahasa", "Judul", "Topik utama (AI)", "Semua topik yang benar (AI, nomor)", "Keyakinan AI", "Alasan singkat AI"])
    for r in sorted(rows, key=lambda x: x["item_id"]):
        main = r["main_topic"]
        allc = "; ".join(str(tops[t]) for t in r["all_correct"])
        lang = inp[r["item_id"]]["lang"]
        ws.append([r["item_id"], f"{LANG.get(lang, lang)} ({lang})", inp[r["item_id"]]["title"], f"{tops[main]:02d} - {main}", allc, r["confidence"], r["rationale"]])
    for c, w in zip("ABCDEFG", (9, 16, 45, 40, 22, 12, 80)):
        ws.column_dimensions[c].width = w
    for cell in ws[1]:
        cell.font, cell.fill, cell.alignment = Font(bold=True), HEAD, WRAP
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = WRAP
    path = OUT / "ai_reference" / "ai_labels.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    print("written", path, len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", action="store_true")
    args = ap.parse_args()
    step2() if args.ai else step1()
