# StaticKD: *Knowledge Distillation* Pengklasifikasi Topik Berita IPTC Multibahasa ke Tabel Logit Token Statis untuk Inferensi Ringan di CPU

Riset ini mendistilasi pengklasifikasi IPTC Media Topic milik Kuzman & Ljubešić (2025), XLM-RoBERTa-large
dengan 560 juta parameter, menjadi **StaticKD**. Setelah *training*, StaticKD hanya berupa satu tabel berisi
17 logit per token. Inferensinya cukup tokenisasi, rata-rata baris tabel, dan penambahan bias, hanya dengan
`numpy` + `tokenizers`, dan mampu memproses ratusan artikel per detik per *core* CPU.

Seluruh angka, tabel, dan gambar jurnal dihasilkan oleh satu notebook:
**[notebooks/StaticKD_IPTC_final.ipynb](notebooks/StaticKD_IPTC_final.ipynb)** (penjelasan berbahasa
Indonesia). Ringkasan hasil ada di Bagian 10.6 notebook dan di
[results/final/paper_numbers.json](results/final/paper_numbers.json).

---

## Peta cepat: mana yang final?

| Yang dicari | Lokasi |
|---|---|
| **Paper final (bahasa Inggris)** | [`paper_final/main.tex`](paper_final/main.tex) dan [`paper_final/main.pdf`](paper_final/main.pdf). Edit teks **langsung di file ini**. |
| Terjemahan bahasa Indonesia (untuk dibaca) | [`paper_final_id/paper_id_final.pdf`](paper_final_id/) dan ringkasan presentasi `ringkasan_presentasi_StaticKD.docx` |
| **Kode final** | [`src/statickd/`](src/statickd/) (paket Python) dan [`scripts/`](scripts/) |
| **Notebook final** | [`notebooks/StaticKD_IPTC_final.ipynb`](notebooks/StaticKD_IPTC_final.ipynb) (dibangkitkan dari `src/notebook_build/`) |
| Hasil yang dilaporkan paper | [`results/final/`](results/final/) (angka, gambar, benchmark, prediksi) |
| Model final | `models/final/statickd/` dan `models/final/statickd_compact/` |
| Lembar kerja penulisan, panduan gaya, review pemetaan | [`paper_writing/`](paper_writing/) |
| Desain penelitian | [`docs/RANCANGAN_PENELITIAN.md`](docs/RANCANGAN_PENELITIAN.md) |
| Arsip tahap eksplorasi (v1/v2), **tidak dipakai lagi** | [`_archive/`](_archive/), `models/_archive_v1/`, `results/_archive_v1/` |

**Peringatan.** `scripts/build_paper.py` menyusun `paper_final/main.tex` dari sumber modular `src/paper/`.
Sumber itu sudah ketinggalan dari revisi teks manual. Jangan jalankan `build_paper.py --force`, karena semua
revisi di `paper_final/main.tex` akan tertimpa.

## Struktur proyek

| Path | Isi |
|---|---|
| `paper_final/` | **Naskah final IEEE Access**: `main.tex`, `main.pdf`, `references.bib`, `figures/` |
| `paper_final_id/` | Terjemahan bahasa Indonesia dan ringkasan presentasi (bukan naskah resmi) |
| `paper_writing/` | Lembar kerja parafrase, panduan gaya (`ai_text_revision_guide*.md`), `change_log.md`, `mapping_review.xlsx`, template IEEE Access (tidak di-push) |
| `src/statickd/` | **Kode final**: data, set uji, *teacher*, StaticKD, *baseline*, evaluasi, *benchmark*, ekspor angka |
| `src/notebook_build/` | Generator notebook final (`part1.py` … `part5.py`) |
| `src/paper/` | Sumber modular awal paper. Sudah ketinggalan, hanya untuk `build_paper.py` |
| `scripts/` | Titik masuk langkah berat (training, evaluasi, benchmark, pelabelan *teacher*) dan pemeriksa paper (`check_paper.py`, `check_style.py`) |
| `notebooks/StaticKD_IPTC_final.ipynb` | Notebook penelitian final, langkah demi langkah |
| `data/raw/`, `data/processed/` | Dataset mentah dan olahan, termasuk set uji gabungan (`data/processed/testset_jsonld/unified_test.jsonl`). Tidak di-push |
| `models/final/`, `models/students/`, `models/baselines/`, `models/lid/` | Model final, 72 run StaticKD, *baseline*, LID fastText. Bobot tidak di-push |
| `results/final/` | Semua yang dilaporkan paper; `predictions/` tidak di-push |
| `docs/` | Rancangan penelitian dan checklist jurnal |
| `presentation/` | Video presentasi (tidak di-push) |
| `materi_kuliah/`, `reference/`, `sample/` | Bahan kuliah, paper referensi, contoh tugas (dua terakhir tidak di-push) |
| `dist/` | Paket rilis hasil `scripts/build_dist.py` (bisa dibangun ulang) |
| `_archive/` | Tahap eksplorasi: `code_v1/` (kode lama), `jurnal_latex_v1/` (naskah lama), `notebooks_v1/` (notebook Colab lama), `proposal_materials/`, `root_docs_v1/` (laporan dan draf lama), `gdrive_upload_v1/` (zip Drive lama, tidak di-push) |
| `models/_archive_v1/`, `results/_archive_v1/` | Model dan hasil tahap eksplorasi |
| `env/` | Virtual environment (tidak di-push) |

Data, model, dan prediksi untuk menjalankan ulang proyek dibagikan terpisah sebagai bundle zip
(lihat `README_BUNDLE.md` di dalam bundle).

## Setup

```bash
python -m venv env
env\Scripts\activate                 # Linux/macOS: source env/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # atau /whl/cpu
pip install -r requirements.txt
```

Kebutuhan yang dipasang terpisah:
- Python 3.12.
- PyTorch 2.6 (untuk *training* dan pelabelan *teacher*).
- GPU NVIDIA (opsional, hanya mempercepat).
- LaTeX (MiKTeX/TeX Live) untuk mengompilasi jurnal.
- `models/lid/lid.176.bin` (fastText LID) untuk konstruksi set uji.
- EMMediaTopic 1.0 dari CLARIN.SI (`http://hdl.handle.net/11356/1991`) di `data/raw/EMMediaTopic-1.0.jsonl`.

## Menjalankan

| Tujuan | Perintah |
|---|---|
| Membuat ulang semua tabel/gambar dari hasil tersimpan | jalankan notebook dengan semua saklar `RUN_* = False` |
| Melatih ulang 72 run StaticKD | `python scripts/train_statickd.py` (bisa paralel: `--worker 0 --num-workers 2`) |
| *Baseline* | `python scripts/train_baselines.py` lalu `python scripts/train_transformers.py` |
| Prediksi *teacher* | `python scripts/evaluate_model.py --name teacher --spec teacher-gpu` |
| *Benchmark* CPU (mesin idle) | notebook Bagian 10.1 dengan `RUN_BENCHMARK = True` |
| Konstruksi set uji | `python -m statickd.testset.ccnews_warc`, `…jsonld`, `…silver`, `…external`, `…unified` (dengan `PYTHONPATH=src`) |
| Paket rilis | `python scripts/build_dist.py` |
| Kompilasi jurnal | `cd paper_final && pdflatex main && bibtex main && pdflatex main && pdflatex main` |

## Memakai model

```python
import sys; sys.path.insert(0, "src")
from statickd import StaticKDModel

model = StaticKDModel.load("models/final/statickd")
model.predict_labels(["Pemerintah menaikkan harga BBM mulai pekan depan ...",
                      "Timnas Indonesia menang 2-0 atas Vietnam ..."])
```

Input sebaiknya berupa judul + isi artikel (maksimal 512 kata pertama). Varian `statickd_compact` lebih
hemat RAM, tetapi hanya layak untuk bahasa yang tercakup data distilasi (lihat jurnal, Bagian IV-E).

## Masalah yang diketahui

- **fastText dengan NumPy 2.** `fasttext-wheel` 0.9.2 gagal di `model.predict()` pada NumPy 2. Gunakan API
  level-rendah `model.f.predict(teks, k, 0.0, "strict")`, seperti di `src/statickd/baselines.py`.
- **Windows dan heredoc.** Heredoc di Git Bash mengubah `\\` dan escape seperti `\v`/`\t`. Jangan menyunting
  `.tex` lewat `sed`/heredoc; pakai editor.
- **VRAM 6 GB.** Maksimal dua *training* StaticKD bersamaan. Transformer dijalankan sendiri.
- **`transformers` 5.x** memuat sebagian *checkpoint* dalam fp16. Kode *training* transformer memaksa
  `dtype=torch.float32` karena AMP membutuhkan bobot induk fp32.
