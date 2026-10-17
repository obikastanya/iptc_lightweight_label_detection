from cells import code, md

CELLS = [
md(r"""
# StaticKD: *Knowledge Distillation* Pengklasifikasi Topik Berita IPTC Multibahasa ke Tabel Logit Token Statis untuk Inferensi Ringan di CPU

**Notebook implementasi dan hasil penelitian.** Setiap angka, tabel, dan gambar di jurnal
(`jurnal_final/main.tex`) dihasilkan oleh notebook ini. Bagian 11 menulis angka-angka tersebut ke
`results/final/paper_generated/`, lalu `scripts/build_paper.py` menyusun satu file `main.tex` dengan angka itu,
sehingga tidak ada angka yang diketik manual.

| | |
|---|---|
| Penulis | Obi Kastanya, Ananta Dwi Prayoga Alwy |
| Teacher | `classla/multilingual-IPTC-news-topic-classifier` (Kuzman & Ljubešić, IEEE Access 2025) |
| Metode usulan | StaticKD: knowledge distillation ke tabel 17 logit per token |
| Kode | paket `src/statickd/` (diimpor oleh notebook ini) dan skrip `scripts/` |
"""),
md(r"""
## Daftar isi

0. [Cara memakai notebook](#0)
1. [Setup lingkungan](#1)
2. [Tugas: taksonomi IPTC Media Topic](#2)
3. [Data](#3): data distilasi, set *development*, set uji gabungan 47 bahasa
4. [Teacher](#4)
5. [Metode StaticKD langkah demi langkah](#5)
6. [Training StaticKD dan baseline](#6)
7. [Protokol evaluasi dan uji statistik](#7)
8. [Hasil akurasi (RQ2)](#8)
9. [Ablasi, temperature, dan ensemble (RQ3)](#9)
10. [Efisiensi CPU (RQ1) dan analisis generalisasi (RQ4)](#10)
11. [Ekspor angka, tabel, dan gambar ke jurnal](#11)
12. [Keterbatasan dan catatan reproduktibilitas](#12)

**Pertanyaan penelitian**

- **RQ1 (biaya).** Berapa biaya inferensi CPU teacher, dan berapa besar StaticKD menurunkannya?
- **RQ2 (akurasi).** Seberapa besar akurasi teacher yang dipertahankan StaticKD pada label yang independen
  dari teacher, dibanding baseline berbiaya serupa (fastText, TF-IDF + SVM) dan berbiaya lebih tinggi
  (MiniLM-L6, DistilmBERT)?
- **RQ3 (komponen).** Komponen distilasi apa yang menentukan hasil pada student berkapasitas sangat rendah?
- **RQ4 (generalisasi).** Di bahasa, label, dan sumber label mana StaticKD lemah, dan mengapa?
"""),
md(r"""
<a id="0"></a>
## 0. Cara memakai notebook

Langkah berat (pengunduhan data, pelabelan teacher, training, benchmark) dijalankan oleh skrip di
`scripts/` dan hasilnya disimpan di `data/`, `models/`, dan `results/final/`. Saklar `RUN_*` di sel berikut
menentukan apakah suatu langkah dihitung ulang dari nol (`True`) atau hasil tersimpan yang dipakai
(`False`, bawaan). Dengan semua saklar `False`, notebook selesai dalam beberapa menit dan menghasilkan
ulang seluruh tabel dan gambar dari prediksi yang tersimpan.

| Saklar | Langkah | Perkiraan waktu (RTX 4050 + Core Ultra 7) |
|---|---|---|
| `RUN_CORPUS` | unduh korpus distilasi CC-News 2019–2021 (69 bahasa) | ±1 jam |
| `RUN_TESTSET` | unduh WARC CC-NEWS 2024–2026 (±34 GB), ekstraksi JSON-LD, dataset publik, bangun set uji | ±6 jam |
| `RUN_TEACHER_LABELS` | pelabelan 300 ribu dokumen oleh teacher (GPU) | ±1,5 jam |
| `RUN_TRAINING` | 72 run StaticKD (ablasi, seed) | ±3 jam |
| `RUN_BASELINES` | fastText, TF-IDF + SVM | ±1 jam |
| `RUN_TRANSFORMERS` | MiniLM-L6 + KD dan DistilmBERT + KD (GPU) | ±3 jam |
| `RUN_EVAL_TEACHER` | prediksi teacher di set evaluasi (GPU dan CPU ONNX) | ±1,5 jam |
| `RUN_BENCHMARK` | benchmark CPU semua model (mesin harus idle) | ±1 jam |
"""),
code(r"""
RUN_CORPUS = False
RUN_TESTSET = False
RUN_TEACHER_LABELS = False
RUN_TRAINING = False
RUN_BASELINES = False
RUN_TRANSFORMERS = False
RUN_EVAL_TEACHER = False
RUN_BENCHMARK = False
N_BOOT = 1000   # bootstrap resamples for confidence intervals (2000 for paired tests)
"""),
md(r"""
<a id="1"></a>
## 1. Setup lingkungan

Notebook mengimpor paket `statickd` dari `src/`. Fungsi `run_script` menjalankan skrip di proses terpisah
(dengan `PYTHONPATH=src`) dan menampilkan output-nya, sehingga langkah berat tidak memenuhi memori kernel.
"""),
code(r"""
import json, os, subprocess, sys, time, platform, warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "statickd").exists())  # project root
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from statickd import config, data, evaluation as ev, analysis as an, viz
from statickd.config import LABELS, SHORT_LABELS, NUM_LABELS, FINAL_DIR
from statickd.report import Numbers, fmt, fmt_p, write_table

viz.apply_style()
pd.set_option("display.max_columns", 40, "display.width", 200, "display.float_format", "{:.4f}".format)
NUM = Numbers()   # every number that appears in the paper is registered here (Section 11)


def run_script(script: str, *args: str) -> None:
    # Run scripts/<script> (or `-m module`) in a fresh process and stream its output.
    cmd = [sys.executable, "-u", *(["-m", script] if not script.endswith(".py") else [str(ROOT / "scripts" / script)]), *args]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    with subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            print(line, end="")
    if proc.returncode:
        raise RuntimeError(f"{script} failed ({proc.returncode})")


def show(df, caption=None, digits=4):
    if caption:
        display(Markdown(f"**{caption}**"))
    display(df.style.format(precision=digits, thousands=".", decimal=",") if isinstance(df, pd.DataFrame) else df)
"""),
code(r"""
import importlib.metadata as md_
import torch
info = {
    "python": platform.python_version(), "os": platform.platform(),
    "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    **{pkg: md_.version(pkg) for pkg in ["numpy", "pandas", "scikit-learn", "scipy", "torch", "transformers",
                                          "tokenizers", "onnxruntime", "fasttext-wheel", "datasketch", "trafilatura"]},
}
FINAL_DIR.mkdir(parents=True, exist_ok=True)
(FINAL_DIR / "environment.json").write_text(json.dumps(info, indent=1))
pd.Series(info, name="versi").to_frame()
"""),
md(r"""
<a id="2"></a>
## 2. Tugas: taksonomi IPTC Media Topic

IPTC Media Topic adalah taksonomi berita dari International Press Telecommunications Council. Level
teratasnya punya 17 topik. Tugasnya adalah klasifikasi *single-label*: setiap artikel (judul + isi, 512
kata pertama) diberi satu dari 17 label tersebut, sama seperti Kuzman & Ljubešić (2025). Urutan label di
bawah dipakai oleh semua model dan tabel.
"""),
code(r"""
iptc = pd.DataFrame(json.loads((ROOT / "src/statickd/testset/iptc_reference/mediatopic_en-GB.json").read_text(encoding="utf8")))
active = iptc[~iptc["retired"]]
from statickd.testset.iptc_map import IPTC_TOP
label_table = pd.DataFrame({"label": LABELS, "singkatan": SHORT_LABELS,
                            "subtopik aktif": [int((active.top_name == IPTC_TOP[l]).sum()) - 1 for l in LABELS]})
NUM.add("iptc.nodes", len(active), 0)
NUM.add("iptc.version_nodes", len(iptc), 0)
show(label_table, "17 label IPTC level atas dan jumlah subtopiknya di pohon IPTC")
"""),
md(r"""
<a id="3"></a>
## 3. Data

Penelitian ini memakai empat jenis data dengan peran yang berbeda. Pemisahan peran ini penting agar
evaluasi akhir tidak bocor.

| Set | Isi | Label | Peran |
|---|---|---|---|
| Data distilasi | EMMediaTopic *train* (4 bahasa) + CC-News 2019–2021 (69 bahasa) | logit teacher | training student |
| *Development* | EMMediaTopic *dev* (1.000 dok, 4 bahasa) | GPT-4o | pemilihan resep |
| *Fidelity* | CC-News *held-out* (12.554 dok dari situs yang tidak pernah dilihat) | prediksi teacher | kemiripan dengan teacher |
| **Uji akhir** | Set uji gabungan (12.744 dok, 47 bahasa, 2024–2026 + dataset publik) | manusia / penerbit | semua klaim akurasi |

### 3.1 EMMediaTopic 1.0

EMMediaTopic (Kuzman & Ljubešić, 2025; CLARIN.SI `11356/1991`) berisi 21.000 artikel berbahasa Katalan,
Yunani, Kroasia, dan Slovenia yang dilabeli GPT-4o. *Teacher* di-*fine-tune* pada 15.000 artikel dari split
*train*. Split *dev* (1.000 artikel) tidak pernah dipakai melatih teacher maupun student, sehingga menjadi
set *development* yang adil untuk keduanya.
"""),
code(r"""
emm = data.load_emmediatopic()
emm_stats = emm.groupby(["split", "lang"]).size().unstack(fill_value=0)
emm_stats["total"] = emm_stats.sum(axis=1)
dev = emm[emm.split == "dev"]
NUM.add("emm.train", int((emm.split == "train").sum()), 0)
NUM.add("emm.dev", len(dev), 0)
NUM.add("emm.dev.words", float(dev.text.str.split().str.len().median()), 0)
show(emm_stats, "EMMediaTopic: dokumen per split dan bahasa")
"""),
md(r"""
### 3.2 Korpus distilasi CC-News multibahasa

Teacher dapat melabeli berita dalam bahasa apa pun tanpa biaya anotasi. Karena itu korpus distilasi
diambil dari CC-News 2019–2021 (dataset `CloverSearch/cc-news-mutlilingual`, per bahasa) dengan aturan
berikut:

1. Dokumen hanya dipakai jika bahasa di metadata sama dengan hasil fastText-LID.
2. Teks = judul + isi, minimal 75 kata (200 karakter untuk zh/ja/th/my), dipotong ke 512 kata.
3. Maksimal 150 dokumen per situs, supaya satu situs tidak mendominasi.
4. Split *held-out* dibuat **per domain**: seluruh dokumen dari sebagian situs disisihkan (maksimal 20% per
   bahasa), sehingga *held-out* mengukur generalisasi ke situs berita yang belum pernah dilihat.
5. Dokumen tambahan (maksimal 5.000 per bahasa, maksimal 400 per situs) diambil dari situs *train*, tidak
   pernah dari domain *held-out*.
"""),
code(r"""
if RUN_CORPUS:
    run_script("statickd.corpus.ccnews", "--per-lang", "3000", "--heldout", "300")
    run_script("statickd.corpus.expand", "--extra", "5000")

pool = data.load_distillation_pool()
heldout = data.load_ccnews_heldout()
pool_langs = pool["lang"].value_counts()
corpus_summary = pd.DataFrame({
    "dokumen": [int((pool.source == s).sum()) for s in ("emmediatopic", "ccnews", "ccnews_extra")] + [len(pool), len(heldout)],
    "bahasa": [pool[pool.source == s].lang.nunique() for s in ("emmediatopic", "ccnews", "ccnews_extra")] + [pool.lang.nunique(), heldout.lang.nunique()],
    "situs": [pool[pool.source == s].domain.nunique() for s in ("emmediatopic", "ccnews", "ccnews_extra")] + [pool.domain.nunique(), heldout.domain.nunique()],
}, index=["EMMediaTopic train", "CC-News train", "CC-News tambahan", "total data distilasi", "CC-News held-out"])
overlap_sites = set(pool.domain) & set(heldout.domain)
strict = ev.eval_view("heldout")
print(f"situs held-out yang juga muncul di data distilasi (di bahasa lain, mis. bbc.com): {len(overlap_sites)} "
      f"-> {len(heldout) - len(strict)} dokumen dikeluarkan; held-out yang dinilai: {len(strict)} dokumen, "
      f"{strict.lang.nunique()} bahasa, {strict.domain.nunique()} situs")
NUM.add("data.heldout.strict", len(strict), 0); NUM.add("data.heldout.strict.langs", strict.lang.nunique(), 0)
NUM.add("data.heldout.strict.sites", strict.domain.nunique(), 0)
for key, row in zip(["emm", "cc", "ccextra", "pool", "heldout"], corpus_summary.itertuples()):
    NUM.add(f"data.{key}.docs", row.dokumen, 0); NUM.add(f"data.{key}.langs", row.bahasa, 0)
NUM.add("data.pool.sites", int(pool[pool.source != "emmediatopic"].domain.nunique()), 0)
NUM.add("data.pool.words", float(pool.text.str.split().str.len().median()), 0)
show(corpus_summary, "Korpus distilasi dan held-out")
"""),
code(r"""
fig, ax = plt.subplots(figsize=(11, 3.0))
counts = pool[pool.source != "emmediatopic"].lang.value_counts()
ax.bar(counts.index, counts.values, color=viz.COLORS["statickd"], width=0.8)
ax.set_yscale("log"); ax.set_ylabel("documents (log scale)"); ax.set_xlim(-0.6, len(counts) - 0.4)
ax.tick_params(axis="x", rotation=90, labelsize=6.5)
ax.set_title("CC-News distillation documents per language")
viz.show_and_save(fig, "fig_distil_per_language")
"""),
md(r"""
### 3.3 Set uji gabungan 47 bahasa

Test set manusia Kuzman & Ljubešić tidak publik dan hanya mencakup 4 bahasa. Karena itu kami membangun set
uji multibahasa dengan **label yang independen dari teacher**. Semua sumber diproses dengan pipeline yang
sama.

**(a) CC-NEWS 2024–2026 dengan rubrik JSON-LD.** Satu file WARC per bulan (Januari 2024 – Agustus 2026,
32 file, ±34 GB) diunduh dari Common Crawl. Dari setiap halaman berita diambil blok JSON-LD schema.org
(`NewsArticle` dan turunannya), berisi `articleSection`, `BreadcrumbList`, `inLanguage`,
`datePublished`, `isAccessibleForFree`, dan `articleBody`. Situs yang ada di data distilasi atau evaluasi
dikeluarkan.

**(b) Dataset publik berlabel.**
- MasakhaNEWS: 16 bahasa Afrika + en/fr, anotasi manusia.
- MN-DS: bahasa Inggris, label IPTC dari manusia.
- L3Cube-IndicNews: 10 bahasa India, kategori penerbit.
- Berita Bangla (PLOS ONE), IndoNews (Liputan6), berita Amharik, dan berita Swahili: kategori penerbit.

**(c) Pemetaan ke IPTC.** Nama rubrik atau kategori dipetakan ke node pohon IPTC Media Topic resmi
(`mediatopic_en-GB`: 1.393 node, 1.085 di antaranya aktif). Aturannya konservatif: sebuah nama hanya dipetakan jika
**seluruh isi yang lazim di rubrik itu masuk ke satu topik level atas**. Nama ambigu tidak pernah
dipetakan, misalnya:
- *tech*: bisa *science and technology* atau industri TI di *economy*;
- *entertainment*: bisa *arts* atau *lifestyle*;
- *terrorism*: bisa *conflict* atau *crime*.

Validasi otomatis memastikan setiap node tujuan ada di pohon IPTC, tidak *retired*, dan leluhur level-1-nya
sama dengan label tujuan.

**(d) Filter kualitas dan kebocoran.**
- Terbit ≥ 2024-01-01 (khusus CC-NEWS).
- Tidak berbayar.
- Bahasa yang dideklarasikan cocok dengan fastText lid.176 (jika bahasanya dikenal).
- Panjang ≥ 75 kata.
- Deduplikasi URL, teks identik, dan *near-duplicate* (MinHash, Jaccard ≥ 0,7) lintas semua sumber.
- *Overlap* 8-gram kata terhadap ~300 ribu dokumen latih/evaluasi < 30%.

**(e) Sampling.**
- Maksimal 50 artikel per sel (bahasa × label).
- Sumber berlabel manusia diprioritaskan, lalu kategori penerbit, lalu rubrik JSON-LD.
- Bahasa dengan < 30 artikel dikeluarkan.
- **Tidak ada batas per situs.** Kebanyakan sel hanya berasal dari sedikit situs, sehingga batas per
  situs hanya membuang data tanpa menambah keragaman. Ketergantungan antarartikel dari situs yang sama
  ditangani saat evaluasi dengan *site-cluster bootstrap* (Bagian 7).
"""),
code(r"""
if RUN_TESTSET:
    run_script("statickd.testset.ccnews_warc", "--start", "2024-01", "--end", "2026-08", "--files-per-month", "1")
    run_script("statickd.testset.jsonld", "--workers", "10")
    run_script("statickd.testset.silver", "--per-cell", "50", "--per-domain", "5", "--min-lang-docs", "100")
    run_script("statickd.testset.external")
    run_script("statickd.testset.unified", "--per-cell", "50", "--min-lang-docs", "30")

from statickd.testset import iptc_map
mapping_all = pd.read_csv(ROOT / "src/statickd/testset/iptc_reference/mapping_table.csv")
mapping = mapping_all[~mapping_all.label.str.startswith("(")]   # rows "(not mapped: ambiguous)" list rejected names
NUM.add("map.nodes", len(mapping), 0)
NUM.add("map.names", int(mapping.n_terms.sum()), 0)
NUM.add("map.ambiguous", len(iptc_map.AMBIGUOUS), 0)
print(f"{mapping.n_terms.sum()} nama rubrik/kategori dipetakan ke {len(mapping)} node IPTC; {len(iptc_map.AMBIGUOUS)} nama ambigu sengaja tidak dipetakan")
# author review of the mapping table (scripts/export_mapping_review.py -> research_final/mapping_review.xlsx)
review = json.loads((ROOT / "src/statickd/testset/iptc_reference/mapping_review_record.json").read_text(encoding="utf8"))
assert review["mapped_names"]["total"] == int(mapping.n_terms.sum()), "review record does not match the mapping table"
NUM.add("map.review.total", review["mapped_names"]["total"] + review["other_rules"]["total"], 0)
NUM.add("map.review.other", review["other_rules"]["total"], 0)
NUM.add("map.review.rejected", review["mapped_names"]["rejected"] + review["other_rules"]["rejected"], 0)
print(f"tinjauan penulis ({review['reviewer']}, {review['date']}): {review['result']}")
display(mapping.sample(10, random_state=1)[["label", "iptc_node_path", "n_terms", "section_names"]])
display(pd.Series(iptc_map.AMBIGUOUS, name="alasan tidak dipetakan").head(12).to_frame())
"""),
code(r"""
base = data.TESTSET_DIR
silver_funnel = pd.DataFrame(json.loads((base / "silver_funnel.json").read_text()), columns=["tahap", "artikel"])
unified_funnel = pd.DataFrame(json.loads((base / "unified_funnel.json").read_text()), columns=["tahap", "artikel"])
show(silver_funnel, "Funnel (a): CC-NEWS 2024–2026 JSON-LD", digits=0)
show(unified_funnel, "Funnel (b)–(e): penggabungan dengan dataset publik", digits=0)
sf = dict(silver_funnel.values); uf = dict(unified_funnel.values)
NUM.add("ts.html_articles", int(silver_funnel.artikel.iloc[0]), 0)
NUM.add("ts.mapped", int([v for k, v in sf.items() if k.startswith("1.")][0]), 0)
NUM.add("ts.silver_pool", int([v for k, v in sf.items() if k.startswith("4.")][0]), 0)
NUM.add("ts.ext_loaded", int(unified_funnel.artikel.iloc[0]), 0)
NUM.add("ts.ext_mapped", int(unified_funnel.artikel.iloc[1]), 0)
NUM.add("ts.pool", int(unified_funnel.artikel.iloc[-2]), 0)
NUM.add("ts.contam_removed", int(unified_funnel.artikel.iloc[-3] - unified_funnel.artikel.iloc[-2]), 0)
"""),
code(r"""
test = ev.eval_set("test")
lang_info = an.language_info(test, pool_langs)
by_source = (test.groupby(["source", "label_origin"])
             .agg(artikel=("doc_id", "size"), bahasa=("lang", "nunique"), situs=("cluster", "nunique"))
             .reset_index())
by_origin = test.groupby("label_origin").size()
NUM.add("ts.docs", len(test), 0); NUM.add("ts.langs", test.lang.nunique(), 0)
NUM.add("ts.labels", test.label.nunique(), 0); NUM.add("ts.sites", test.cluster.nunique(), 0)
for o, n in by_origin.items():
    NUM.add(f"ts.origin.{o}", int(n), 0); NUM.add(f"ts.origin.{o}.pct", n / len(test), 1, percent=True)
NUM.add("ts.words", float(test.text.str.split().str.len().median()), 0)
NUM.add("ts.lang_labels.median", float(lang_info.n_labels.median()), 0)
SOURCE_TEX = {"ccnews_jsonld": "CC-NEWS 2024--2026", "masakhanews": "MasakhaNEWS~\cite{adelani2023masakhanews}",
              "mnds": "MN-DS~\cite{petukhova2023mnds}", "l3cube_indic": "L3Cube-IndicNews~\cite{mirashi2023indicnews}",
              "bangla_plos": "Bangla news~\cite{hossain2025banglanews}", "indonews": "IndoNews",
              "amharic_news": "Amharic news~\cite{amharicnews2024}", "swahili_news": "Swahili news~\cite{swahilinews2020}"}
ORIGIN_TEX = {o: an.en(o) for o in ("human_annotation", "publisher_category", "publisher_section_jsonld")}
rows_tex = [f"{SOURCE_TEX[r.source]} & {ORIGIN_TEX[r.label_origin]} & {fmt(int(r.artikel), 0)} & {fmt(int(r.bahasa), 0)} & {fmt(int(r.situs), 0)} \\\\"
            for r in by_source.sort_values("artikel", ascending=False).itertuples()]
rows_tex += ["\midrule", f"Total & & {fmt(len(test), 0)} & {fmt(test.lang.nunique(), 0)} & {fmt(test.cluster.nunique(), 0)} \\\\"]
write_table("testset", rows_tex)
show(by_source, "Set uji gabungan per sumber", digits=0)
"""),
code(r"""
region_table = (lang_info.reset_index().groupby("region")
                .agg(bahasa=("lang", "size"), artikel=("n_test", "sum"), daftar=("lang", lambda s: ", ".join(sorted(s))))
                .sort_values("artikel", ascending=False))
tier_table = lang_info.groupby("diversity_tier").agg(bahasa=("n_test", "size"), artikel=("n_test", "sum"))
distil_table = lang_info.reset_index().groupby("distil_tier").agg(bahasa=("lang", "size"), artikel=("n_test", "sum"),
                                                                  daftar=("lang", lambda s: ", ".join(sorted(s))))
for tier, row in distil_table.iterrows():
    NUM.add(f"ts.distil.{tier.split()[0]}.langs", int(row.bahasa), 0)
for tier, row in tier_table.iterrows():
    NUM.add(f"ts.div.{tier.split()[0]}.langs", int(row.bahasa), 0)
show(region_table, "Bahasa per wilayah", digits=0)
show(distil_table, "Bahasa uji menurut jumlah data distilasi di bahasa itu", digits=0)
show(tier_table, "Tingkat keragaman situs per bahasa (jumlah situs efektif = 1/Σ share²)", digits=0)
"""),
code(r"""
ct = pd.crosstab(test.lang, test.label).reindex(columns=LABELS, fill_value=0)
ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(7.2, 8.8))
im = ax.imshow(ct.values, cmap="Blues", aspect="auto", vmin=0, vmax=50)
ax.set_xticks(range(NUM_LABELS)); ax.set_xticklabels(SHORT_LABELS, rotation=60, ha="right")
ax.set_yticks(range(len(ct))); ax.set_yticklabels([f"{l} ({an.en(test[test.lang == l].label_origin.mode()[0])[:9]})" for l in ct.index], fontsize=6.5)
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.4, label="articles")
ax.set_title("Unified test set: articles per language × label")
viz.show_and_save(fig, "fig_testset_heatmap")
label_counts = test.label.value_counts().reindex(LABELS)
NUM.add("ts.label.max", int(label_counts.max()), 0); NUM.add("ts.label.min", int(label_counts.min()), 0)
NUM.add("ts.label.min.name", SHORT_LABELS[LABELS.index(label_counts.idxmin())])
"""),
]
