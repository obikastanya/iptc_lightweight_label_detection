# StaticKD: Klasifikasi Topik Berita IPTC Multibahasa Secepat *Language Identification*

Riset ini mendistilasi pengklasifikasi IPTC Media Topic milik Kuzman & Ljubešić (2025), XLM-RoBERTa-large
dengan 560 juta parameter, menjadi **StaticKD**. Setelah *training*, StaticKD hanya berupa satu tabel berisi
17 logit per token. Inferensinya cukup tokenisasi, rata-rata baris tabel, dan penambahan bias, hanya dengan
`numpy` + `tokenizers`, dan mampu memproses ratusan artikel per detik per *core* CPU.

Seluruh angka, tabel, dan gambar jurnal dihasilkan oleh satu notebook:
**[notebooks/StaticKD_IPTC_final.ipynb](notebooks/StaticKD_IPTC_final.ipynb)** (penjelasan berbahasa
Indonesia). Ringkasan hasil ada di Bagian 10.6 notebook dan di
[results/final/paper_numbers.json](results/final/paper_numbers.json).

---

## Struktur proyek

| Path | Isi |
|---|---|
| `src/statickd/` | **Kode final** (paket Python): data, set uji, *teacher*, StaticKD, *baseline*, evaluasi, *benchmark*, ekspor ke LaTeX |
| `scripts/` | Titik masuk langkah berat: `train_statickd.py`, `train_baselines.py`, `train_transformers.py`, `evaluate_model.py`, `label_with_teacher.py`, `validity_silver.py`, `build_dist.py` |
| `notebooks/StaticKD_IPTC_final.ipynb` | Notebook penelitian final, langkah demi langkah |
| `jurnal_latex/` | Naskah IEEE (`main.tex`, `sections/`); `generated/` ditulis oleh notebook |
| `docs/RANCANGAN_PENELITIAN.md` | Topik, latar belakang, RQ, batasan, *novelty*, kontribusi, protokol |
| `docs/CHECKLIST_JURNAL.md` | Checklist kelengkapan informasi jurnal ↔ notebook |
| `results/final/` | Prediksi semua model (`predictions/*.npz`), angka paper, *benchmark* CPU, versi pustaka |
| `models/students/` | 72 run StaticKD (ablasi dan *seed*) |
| `models/final/` | Model rilis: `statickd/` (*ensemble* 3 *seed*) dan `statickd_compact/` |
| `models/baselines/` | fastText, TF-IDF + SVM, MiniLM-L6 + KD, DistilmBERT + KD |
| `dist/github/`, `dist/huggingface/` | Paket rilis (dibuat oleh `scripts/build_dist.py`) |
| `data/raw/`, `data/processed/` | Dataset mentah dan olahan, termasuk set uji gabungan (`data/processed/testset_jsonld/unified_test.jsonl`) |
| `code/`, `LAPORAN_RISET.md`, `jurnal.md`, `notebooks/IPTC_StaticKD_Colab*.ipynb` | **Arsip tahap eksplorasi** (versi v1/v2 sebelum riset difinalkan). Tidak dipakai oleh notebook final |
| `env/` | Virtual environment (tidak di-push) |

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
| Kompilasi jurnal | `cd jurnal_latex && pdflatex main && bibtex main && pdflatex main && pdflatex main` |

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
