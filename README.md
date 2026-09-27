# StaticKD: Klasifikasi Topik Berita IPTC Multibahasa yang Ringan untuk CPU

Riset ini mendistilasi pengklasifikasi IPTC Media Topic milik Kuzman & Ljubešić (2025) menjadi model ringan untuk CPU. Model teacher-nya adalah XLM-RoBERTa-large (560 juta parameter), dan hasil distilasinya disebut **StaticKD**: embedding token statis multibahasa yang dirata-rata lalu dipetakan secara linear ke 17 topik. Setelah dilatih, model dilipat menjadi tabel berisi 17 skor per token (34 MB). Inferensinya hanya butuh `numpy` dan `tokenizers`.

| Model | Macro-F1 dev (4 bahasa) | Macro-F1 16 bahasa lain | Kecepatan (1 core CPU) | Ukuran |
|---|---|---|---|---|
| Teacher XLM-R-large (ONNX int8) | 0,828 | 0,836 | 0,62 artikel/detik | 537 MB |
| MiniLM-L6 + KD | 0,763 | 0,749 | 12,5 artikel/detik | 408 MB |
| **StaticKD v2** | **0,761** | **0,745** | **±450–600 artikel/detik** | **34 MB** |
| fastText + KD | 0,728 | 0,656 | 639 artikel/detik | 18 MB |

Laporan lengkap ada di [LAPORAN_RISET.md](LAPORAN_RISET.md). Draf artikel jurnal format IEEE ada di [jurnal_latex/](jurnal_latex/).

---

## Struktur folder

| Folder / file | Isi | Di GitHub? |
|---|---|---|
| `code/` | Kode riset: data, teacher, student, evaluasi, eksperimen ([code/README.md](code/README.md)) | ya |
| `notebooks/` | Notebook Google Colab `IPTC_StaticKD_Colab.ipynb` beserta versi preview hasilnya | ya (bundle `.zip` via Google Drive) |
| `results/` | Hasil akurasi (JSON), benchmark CPU, gambar, dan tabel paper | ya |
| `jurnal_latex/` | Naskah jurnal LaTeX (IEEEtran), referensi, dan gambar | ya |
| `LAPORAN_RISET.md`, `jurnal.md` | Laporan riset dan draf jurnal versi Markdown | ya |
| `archieve/` | Bahan penyusunan proposal | ya |
| `data/` | Dataset mentah dan hasil olahan (±37 GB) | **tidak**, via Google Drive |
| `models/` | Bobot model (±15 GB). Hanya `meta.json` / `config.json` yang di-push. | **tidak**, via Google Drive |
| `env/` | Virtual environment Python | **tidak**, install ulang |
| `reference/`, `sample/` | Paper referensi berhak cipta dan contoh tugas | **tidak** |
| `scratch/` | File kerja sementara | **tidak** |

---

## 1. Yang perlu diinstall terpisah

Hal-hal berikut tidak bisa dipasang lewat `pip install -r requirements.txt`:

| Kebutuhan | Wajib? | Catatan |
|---|---|---|
| **Python 3.12** | wajib | Riset memakai 3.12.4. |
| **Git** | wajib | Untuk clone repositori. |
| **PyTorch 2.6** | wajib untuk training/teacher | Dipasang sesuai perangkat (lihat langkah 2). Inferensi StaticKD tidak membutuhkannya. |
| **Driver NVIDIA + CUDA 12.x** | opsional | Hanya untuk mempercepat pelabelan teacher dan training. Riset memakai RTX 4050 6 GB. Benchmark tetap CPU-only. |
| **Distribusi LaTeX** (MiKTeX / TeX Live) | opsional | Untuk mengompilasi `jurnal_latex/`. Bisa diganti Overleaf. |
| **Model fastText LID** `lid.176.bin` | opsional | Deteksi bahasa untuk pipeline test set CC-NEWS 2024–2026. Unduh dari `https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin` ke `models/lid/`. |
| **Dataset EMMediaTopic 1.0** | wajib untuk training/evaluasi | Unduh dari CLARIN.SI (`http://hdl.handle.net/11356/1991`) lalu simpan sebagai `data/raw/EMMediaTopic-1.0.jsonl`. Bisa juga diambil dari Google Drive (langkah 3). |
| **`ANTHROPIC_API_KEY`** | opsional | Hanya untuk membuat ulang anotasi LLM test set di notebook. |

Model dari Hugging Face diunduh otomatis saat pertama kali dipakai ke `models/hf_cache/`, karena `code/config.py` mengatur `HF_HOME`. Model yang diunduh: teacher `classla/multilingual-IPTC-news-topic-classifier`, versi ONNX-nya, `minishlab/potion-multilingual-128M`, dan `nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large`. Folder ini **tidak perlu** di-upload ke Google Drive.

## 2. Setup environment

```bash
git clone <URL-repo-ini>
cd iptc_label_class

python -m venv env
# Windows:  env\Scripts\activate
# Linux/Mac: source env/bin/activate

# PyTorch: pilih salah satu
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # GPU NVIDIA
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu     # CPU saja

pip install -r requirements.txt
```

Semua perintah di `code/` dijalankan dari dalam folder `code/`, misalnya `cd code && python evaluation/evaluate_accuracy.py ...`.

## 3. File besar dari Google Drive

Tautan Google Drive: **[TODO: isi tautan]**

Unduh folder yang dibutuhkan, lalu letakkan **persis** di path berikut (relatif terhadap root proyek):

| Path | Ukuran | Isi | Dibutuhkan untuk |
|---|---|---|---|
| `models/static/static_v2_table/` | 34 MB | **Model rekomendasi (StaticKD v2)** | memakai model |
| `models/static/static_v2_table_compact10/` | 23 MB | v2 dengan tokenizer dipangkas (RAM ±281 MB) | memakai model di server kecil |
| `models/static/static_final_table*/` | 23–34 MB | Model v1 (angka utama di laporan) | reproduksi tabel |
| `notebooks/iptc_statickd_bundle.zip` | 37 MB | Hasil + model final untuk notebook Colab (mode REPORT) | notebook Colab |
| `data/raw/EMMediaTopic-1.0.jsonl` | kecil | Dataset EMMediaTopic | training, evaluasi |
| `data/processed/` | 2,8 GB | Logit teacher untuk 286 ribu dokumen, test set LLM, token cache | training ulang student tanpa melabel ulang |
| `models/fasttext/`, `models/transformer/`, `models/static/` (lainnya) | ±10 GB | Baseline dan varian ablasi | reproduksi ablasi & benchmark |
| `models/static_base/` | 261 MB | Embedding statis hasil distilasi dari teacher (ablasi D) | ablasi |
| `models/lid/lid.176.bin` | 126 MB | fastText LID | pipeline test set |
| `data/raw/ccnews_recent/` | 32 GB | 32 file WARC mentah CC-NEWS 2024-01 s.d. 2026-08 dan `manifest.json` | pembuatan test set manusia |

Tiga skenario umum:
- **Hanya memakai model:** cukup `models/static/static_v2_table/`.
- **Mereproduksi tabel dan gambar:** cukup `notebooks/iptc_statickd_bundle.zip`, lalu jalankan notebook dalam mode REPORT.
- **Melatih ulang student:** `data/raw/EMMediaTopic-1.0.jsonl` ditambah `data/processed/`.

`data/raw/ccnews_recent/warc/` tidak perlu diunduh dari Drive. Folder ini bisa diunduh ulang langsung dari Common Crawl dengan `python data/download_ccnews_warc.py`. File yang dipilih deterministik, jadi hasilnya sama persis dengan `manifest.json`.

## 4. Memakai model

```python
# jalankan dari folder code/
from students.static_student import StaticPredictor
from config import LABELS

model = StaticPredictor("../models/static/static_v2_table")
texts = ["Pemerintah menaikkan harga BBM mulai pekan depan ...",
         "Timnas Indonesia menang 2-0 atas Vietnam ..."]
print([LABELS[i] for i in model.predict(texts)])
```

Input sebaiknya berupa judul + isi artikel (maksimal 512 kata pertama), sama seperti data latih.

## 5. Reproduksi riset

- **Pipeline lengkap** (pengumpulan data, pelabelan teacher, training, evaluasi, benchmark): lihat [code/README.md](code/README.md).
- **Eksplorasi akurasi** (bigram, temperatur, ensemble, bobot token, v2): `code/experiments/` dan opsi baru di `code/students/static_student.py` (`--temperature`, `--bigram-bits`, `--token-weight`, `--conf-weight`, `--gpt-weight`, `--max-tokens`). Hasilnya ada di `results/exploration/`.
- **Notebook Colab:** buka `notebooks/IPTC_StaticKD_Colab.ipynb` di Google Colab dan unggah `iptc_statickd_bundle.zip` bila diminta. Mode `REPORT` hanya membuat ulang tabel dan gambar, sedangkan mode `FULL` menjalankan seluruh pipeline.
- **Benchmark CPU:** `code/evaluation/run_cpu_benchmarks.py`. CPU riset (Intel Core Ultra 7 155H) bersifat *hybrid*, jadi proses dipin ke P-core (`BENCH_PCORES`, default `0,10,12,14,16,18`). Sesuaikan untuk CPU lain.

## 6. Mengompilasi jurnal

```bash
cd jurnal_latex
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Cara lain: unggah isi `jurnal_latex/` (tanpa `template_ieee/`) ke Overleaf. Template IEEE asli dapat diunduh dari IEEE Author Center (*IEEE Transactions LaTeX2e templates*).

## 7. Masalah yang diketahui

- **fastText dengan NumPy 2.** `fasttext-wheel` 0.9.2 memanggil `np.array(..., copy=False)` di `model.predict()`. Pada NumPy ≥ 2 pemanggilan ini bisa memunculkan `ValueError`. Ada dua solusi:
  - memanggil API level-rendah, yaitu `model.f.predict(teks, k, 0.0, "strict")`, seperti yang dipakai di notebook;
  - memakai environment terpisah dengan `numpy<2` untuk skrip fastText.
- **Windows dan heredoc.** Saat menulis skrip melalui heredoc di Git Bash, `\\` bisa berubah menjadi `\`. Tulis file `.tex`/`.py` langsung dengan editor.
- **VRAM GPU 6 GB.** Jangan menjalankan lebih dari dua training student secara bersamaan.

## Sitasi

Draf artikel: O. Kastanya dan A. D. P. Alwy, "StaticKD: Klasifikasi Topik Berita IPTC Multibahasa yang Ringan untuk CPU melalui Distilasi Pengetahuan ke Model Static-Embedding," draf, 2026.

Model teacher dan dataset EMMediaTopic berasal dari: T. Kuzman dan N. Ljubešić, "LLM teacher-student framework for text classification with no manually annotated data: A case study in IPTC news topic classification," *IEEE Access*, vol. 13, 2025.
