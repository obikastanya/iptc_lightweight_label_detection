"""Export the rubric -> IPTC mapping table as a review sheet for the authors (decision D2).

One row per mapped section name, with the IPTC node, the top-level label, how many test-set articles
carry that name, and up to three example headlines. The reviewer fills the last three columns.
Language-specific overrides, ambiguous names that are deliberately NOT mapped, and the names that reject
an article (astrology) follow the mapped names (status column), so the whole rule set is in one file.

Output: research_final/mapping_review.xlsx with three sheets:
  Petunjuk     how to review, an example row, and a live count of reviewed rows
  Dipetakan    the 814 mapped names, most used in the test set first
  Aturan lain  language-specific rules, ambiguous names, rejecting names

Usage (from the project root):
    python scripts/export_mapping_review.py [--force]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from statickd.data import load_unified_test  # noqa: E402
from statickd.testset import iptc_map  # noqa: E402

OUT = ROOT / "research_final" / "mapping_review.xlsx"
# English glosses (reading aid only; written by Claude, see the _note in the file)
GLOSS = json.loads((ROOT / "src/statickd/testset/iptc_reference/section_names_en.json").read_text(encoding="utf8"))
EN_COL = "nama_rubrik_en (terjemahan Claude)"
REVIEW_COLUMNS = {"setuju (ya/tidak)": "", "label_seharusnya": "", "catatan": ""}
CHOICES = ["ya", "tidak", "tidak bisa dinilai"]
WIDTHS = {"status": 22, "perlu_verifikasi": 11, "label_iptc": 26, "node_iptc": 34, "nama_rubrik": 24, EN_COL: 26, "bahasa_khusus": 9,
          "artikel_di_set_uji": 11, "contoh_judul": 80, "setuju (ya/tidak)": 14, "label_seharusnya": 26, "catatan": 40}


def write_sheet(ws, df: pd.DataFrame, labels: list[str]) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    head_font, body_font = Font(name="Arial", bold=True, color="FFFFFF"), Font(name="Arial", size=10)
    head_fill = PatternFill("solid", fgColor="44546A")
    input_fill = PatternFill("solid", fgColor="FFF2CC")   # cells the reviewer fills in
    ws.append(list(df.columns))
    for values in df.itertuples(index=False):
        ws.append(list(values))
    n = len(df) + 1
    for j, col in enumerate(df.columns, start=1):
        letter = ws.cell(row=1, column=j).column_letter
        ws.column_dimensions[letter].width = WIDTHS.get(col, 14)
        ws.cell(row=1, column=j).font, ws.cell(row=1, column=j).fill = head_font, head_fill
        ws.cell(row=1, column=j).alignment = Alignment(wrap_text=True, vertical="center")
        for i in range(2, n + 1):
            c = ws.cell(row=i, column=j)
            c.font = body_font
            c.alignment = Alignment(wrap_text=col in ("contoh_judul", "node_iptc", "catatan"), vertical="top")
            if col in REVIEW_COLUMNS:
                c.fill = input_fill
    col = {c: ws.cell(row=1, column=j).column_letter for j, c in enumerate(df.columns, start=1)}
    ws.freeze_panes = f"{ws.cell(row=1, column=list(df.columns).index(EN_COL) + 2).column_letter}2"
    ws.auto_filter.ref = ws.dimensions
    agree = DataValidation(type="list", formula1='"' + ",".join(CHOICES) + '"', allow_blank=True)
    label = DataValidation(type="list", formula1="=Petunjuk!$N$2:$N$18", allow_blank=True)
    ws.add_data_validation(agree)
    ws.add_data_validation(label)
    agree.add(f"{col['setuju (ya/tidak)']}2:{col['setuju (ya/tidak)']}{n}")
    label.add(f"{col['label_seharusnya']}2:{col['label_seharusnya']}{n}")


def write_workbook(mapped: pd.DataFrame, other: pd.DataFrame) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Petunjuk"
    bold, normal = Font(name="Arial", bold=True, size=11), Font(name="Arial", size=10)
    input_fill = PatternFill("solid", fgColor="FFF2CC")
    last = len(mapped) + 1
    L = {c: get_column_letter(j) for j, c in enumerate(mapped.columns, start=1)}   # column letters of Dipetakan
    rng = lambda c: f"Dipetakan!{L[c]}2:{L[c]}{last}"
    lines = [
        ("Tinjauan tabel pemetaan nama rubrik -> IPTC Media Topic", bold),
        ("Sumber tabel: src/statickd/testset/iptc_map.py. File ini dibuat ulang oleh scripts/export_mapping_review.py.", normal),
        ("", normal),
        ("Yang diisi: hanya tiga kolom berwarna kuning di sheet Dipetakan dan Aturan lain.", bold),
        ("Baris dengan perlu_verifikasi = tidak sudah diisi otomatis 'ya': nama rubrik sama persis dengan node/label IPTC,", normal),
        ("  atau hanya bentuk jamak bahasa Inggrisnya (mis. sports -> sport). Alasannya ada di kolom catatan; boleh diubah.", normal),
        ("  Saring kolom perlu_verifikasi = ya untuk melihat baris yang perlu Anda nilai.", normal),
        ("Kolom nama_rubrik_en: terjemahan bahasa Inggris oleh Claude, hanya sebagai alat bantu baca (bukan penerjemah", normal),
        ("  independen). Nama yang sudah berbahasa Inggris ditampilkan apa adanya. Bila ragu, lihat contoh_judul.", normal),
        ("  setuju (ya/tidak): ya = nama rubrik memang termasuk label_iptc; tidak = salah; tidak bisa dinilai = bahasa tidak dikuasai.", normal),
        ("  label_seharusnya: isi hanya bila 'tidak'. Pilih dari daftar 17 label (kolom N), atau kosongkan bila nama itu", normal),
        ("    sebaiknya tidak dipetakan sama sekali (ambigu).", normal),
        ("  catatan: alasan singkat, terutama untuk 'tidak'.", normal),
        ("", normal),
        ("Urutan sheet Dipetakan: baris yang perlu diverifikasi dulu, lalu menurut artikel_di_set_uji (terbanyak dulu).", normal),
        ("Nama dengan 0 artikel tidak memengaruhi set uji saat ini, tetapi tetap bagian dari aturan pemetaan.", normal),
        ("", normal),
        ("Contoh baris yang sudah diisi (hanya contoh, tidak ada di data):", bold),
    ]
    for i, (text, font) in enumerate(lines, start=1):
        ws.cell(row=i, column=1, value=text).font = font
    example = {"status": "dipetakan", "perlu_verifikasi": "ya", "label_iptc": "sport", "node_iptc": "sport",
               "nama_rubrik": "olahraga", EN_COL: "sport", "bahasa_khusus": "", "artikel_di_set_uji": 120,
               "contoh_judul": "[id] Timnas Indonesia lolos ke putaran ketiga ...", "setuju (ya/tidak)": "tidak",
               "label_seharusnya": "crime, law and justice",
               "catatan": "contoh saja: bila rubrik ini ternyata berisi berita kriminal"}
    row_ex = len(lines) + 1
    for j, colname in enumerate(mapped.columns, start=1):
        c = ws.cell(row=row_ex, column=j, value=example[colname])
        c.font = normal
        if colname in REVIEW_COLUMNS:
            c.fill = input_fill
    ws.cell(row=row_ex + 2, column=1, value="Progres (dihitung otomatis)").font = bold
    agree = rng("setuju (ya/tidak)")
    progress = [("Baris Dipetakan", f"=COUNTA({rng('nama_rubrik')})"),
                ("Perlu diverifikasi", f'=COUNTIF({rng("perlu_verifikasi")},"ya")'),
                ("Diisi otomatis 'ya'", f'=COUNTIF({rng("perlu_verifikasi")},"tidak")'),
                ("Sudah ditinjau (termasuk otomatis)", f"=COUNTA({agree})"),
                ("Ditandai 'tidak'", f'=COUNTIF({agree},"tidak")'),
                ("Artikel set uji pada baris 'tidak'", f'=SUMIF({agree},"tidak",{rng("artikel_di_set_uji")})')]
    for i, (name, formula) in enumerate(progress, start=row_ex + 3):
        ws.cell(row=i, column=1, value=name).font = normal
        ws.cell(row=i, column=2, value=formula).font = normal
    ws.cell(row=1, column=14, value="17 label IPTC").font = bold
    for i, lab in enumerate(iptc_map.LABELS, start=2):
        ws.cell(row=i, column=14, value=lab).font = normal
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["N"].width = 40
    for c in ws["A"]:
        c.alignment = Alignment(wrap_text=False)
    write_sheet(wb.create_sheet("Dipetakan"), mapped, iptc_map.LABELS)
    write_sheet(wb.create_sheet("Aturan lain"), other, iptc_map.LABELS)
    wb.calculation.fullCalcOnLoad = True   # Excel computes the progress formulas when the file is opened
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)


def matched_terms(source: str, source_label: str, lang: str) -> set[tuple[str, str]]:
    """Mapping terms an article was labelled with, as (term, lang) pairs (lang = "" for the general table).

    Mirrors iptc_map.map_section, iptc_map.label_article (JSON-LD sources store the list of section and
    breadcrumb names) and unified.iptc_label (L3Cube Hindi names end in -news-hindi)."""
    names = json.loads(source_label) if source_label.startswith("[") else [source_label]
    found = set()
    for name in names:
        s = iptc_map.normalise(str(name))
        if source == "l3cube_indic":
            s = re.sub(r"-news(-hindi)?$", "", s)
        if not s or len(s) > 40:
            continue
        if lang in iptc_map.LANG_TERMS and s in iptc_map.LANG_TERMS[lang]:
            found.add((s, lang))
        elif s in iptc_map._LOOKUP:
            found.add((s, ""))
        else:
            stripped = re.sub(iptc_map._GENERIC, " ", s).strip()
            if stripped and stripped != s and stripped in iptc_map._LOOKUP:
                found.add((stripped, ""))
    return found


def auto_reason(term: str, node: str, label: str) -> str | None:
    """Rows that need no human check: the section name IS the IPTC node or label name (or its English plural)."""
    t = term.strip().lower()
    for target in (node.strip().lower(), label.strip().lower()):
        if t == target:
            return "otomatis: nama rubrik sama persis dengan node/label IPTC"
        if t == target + "s":
            return "otomatis: bentuk jamak bahasa Inggris dari node/label IPTC"
    return None


def headline(r) -> str:
    title = str(r.get("headline") or "") or str(r["text"])[:120]
    return f"[{r['lang']}] " + " ".join(title.split())[:120]


def main() -> None:
    # the reviewed workbook is the record of the authors' review (2026-10-17, see
    # iptc_reference/mapping_review_record.json): never overwrite it by accident
    if OUT.exists() and "--force" not in sys.argv:
        done = pd.read_excel(OUT, sheet_name="Dipetakan")["setuju (ya/tidak)"].notna().sum()
        if done > 0:
            sys.exit(f"{OUT} already holds {done} reviewed rows; use --force to overwrite it")
    test = load_unified_test()
    test = test[test["source"] != "mnds"]   # MN-DS is annotated directly with IPTC topics
    examples, counts = defaultdict(list), defaultdict(int)
    unmatched = []
    for _, r in test.iterrows():
        keys = matched_terms(r["source"], str(r["source_label"]), r["lang"])
        if not keys:
            unmatched.append(r["source_label"])
        for key in keys:
            counts[key] += 1
            if len(examples[key]) < 3:
                examples[key].append(headline(r))

    def row(status, label, node, term, lang=""):
        key = (term, lang)
        auto = auto_reason(term, node, label) if status == "dipetakan" else None
        review = {**REVIEW_COLUMNS, "setuju (ya/tidak)": "ya", "catatan": auto} if auto else REVIEW_COLUMNS
        return {"status": status, "perlu_verifikasi": "tidak" if auto else "ya", "label_iptc": label, "node_iptc": node, "nama_rubrik": term, EN_COL: GLOSS.get(term, term),
                "bahasa_khusus": lang,
                "artikel_di_set_uji": counts.get(key, 0), "contoh_judul": " || ".join(examples.get(key, [])),
                **review}

    rows = [row("dipetakan", label, node, term)
            for label, groups in iptc_map.MAPPING.items() for node, terms in groups.items() for term in sorted(terms)]
    rows += [row("aturan bahasa khusus" if label else "ditolak untuk bahasa ini", label or "", "", term, lang)
             for lang, terms in iptc_map.LANG_TERMS.items() for term, label in terms.items()]
    rows += [row("ambigu, tidak dipetakan", "", why, term) for term, why in iptc_map.AMBIGUOUS.items()]
    rows += [row("artikel ditolak (astrologi/horoskop)", "", "", term) for term in sorted(iptc_map.REJECT_TERMS)]
    df = pd.DataFrame(rows)
    mapped = df[df.status == "dipetakan"].sort_values(["perlu_verifikasi", "artikel_di_set_uji", "label_iptc", "nama_rubrik"],
                                                      ascending=[False, False, True, True])
    write_workbook(mapped, df[df.status != "dipetakan"])
    print(f"auto-approved: {(mapped.perlu_verifikasi == 'tidak').sum()} names; to verify: {(mapped.perlu_verifikasi == 'ya').sum()}")
    print(f"wrote {OUT}: {len(mapped)} mapped names ({(mapped.artikel_di_set_uji > 0).sum()} used in the test set), "
          f"{len(df) - len(mapped)} other rules; test articles with a matched name: {len(test) - len(unmatched)}/{len(test)}")
    if unmatched:
        print("unmatched source labels (first 10):", pd.Series(unmatched).value_counts().head(10).to_dict())


if __name__ == "__main__":
    main()
