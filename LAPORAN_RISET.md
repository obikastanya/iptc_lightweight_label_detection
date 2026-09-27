# Laporan Riset: Klasifikasi Topik Berita IPTC Multibahasa yang Ringan untuk Pipeline Real-time

*Disusun otomatis oleh Claude pada 26 September 2026. Semua angka di laporan ini berasal dari eksperimen yang benar-benar dijalankan; kode ada di [code/](code/), hasil mentah di [results/](results/).*

---

## 0. Ringkasan (TL;DR)

**Masalah.** Model IPTC terbaik yang tersedia (Kuzman & Ljubešić, 2025; `classla/multilingual-IPTC-news-topic-classifier`, XLM-RoBERTa-large, 560 juta parameter) akurat, tetapi terlalu berat untuk pipeline real-time di server kecil tanpa GPU: di CPU butuh **0,9 detik (ONNX int8) hingga 2,6 detik (PyTorch) per artikel** pada 1 core, throughput 0,2–0,6 artikel/detik, dan RAM 1,4–3,0 GB.

**Metode yang diusulkan — *StaticKD*:** distilasi pengetahuan dari model Kuzman (teacher) ke **model static-embedding multibahasa** (arsitektur secepat fastText):

1. Teacher melabeli ~286 ribu artikel berita tak berlabel dari **70 bahasa** (CC-News), disimpan sebagai *soft label* (17 logit).
2. Student = tokenizer subword → lookup embedding statis multibahasa pra-latih (model2vec *potion-multilingual-128M*, embedding sudah selaras lintas bahasa) → rata-rata → lapisan linear → 17 kelas. Dilatih end-to-end dengan KL-divergence ke distribusi teacher (T=2) + *token dropout*.
3. Karena kepala (head) linear, seluruh model dilipat menjadi **satu tabel berisi 17 angka per token** (float16). Inferensi = tokenisasi + rata-rata baris tabel. Hanya butuh `numpy` + `tokenizers`.

**Hasil utama (CPU saja, 1 core P-core):**

| Model | Macro-F1 dev (4 bahasa) | Macro-F1 16 bahasa lain | Latensi/artikel (p50) | Throughput 1 core | Throughput 4 core | RAM model | Ukuran file |
|---|---|---|---|---|---|---|---|
| Teacher XLM-R-large (Kuzman), PyTorch fp32 | 0,820 | 0,830 | 2.600 ms | 0,22 dok/s | 0,59 dok/s | 2.961 MB | 2.136 MB |
| Teacher, ONNX int8 (versi CPU paling optimal) | 0,828 | 0,836 | 913 ms | 0,62 dok/s | 2,1 dok/s | 1.383 MB | 537 MB |
| MiniLM-L6 + KD (transformer kecil, ONNX) | 0,763 | 0,749 | 71 ms | 12 dok/s | 21 dok/s | 1.242 MB | 408 MB |
| **StaticKD (usulan)** | **0,751** | **0,740** | **1,9 ms** | **447 dok/s** | **1.683 dok/s** | **505 MB** | **34 MB** |
| StaticKD-compact (tokenizer dipangkas) | 0,751 | 0,730 | 1,7 ms | 567 dok/s | 2.191 dok/s | 281 MB | 23 MB |
| fastText + distilasi teacher (quantized) | 0,728 | 0,656 | 1,4 ms | 639 dok/s | 1.537 dok/s | 308 MB | 18 MB |
| fastText dilatih seperti Kuzman (label GPT-4o, 4 bahasa) | 0,681 | 0,131 | — | — | — | — | 816 MB |

- StaticKD **~720× lebih cepat** dari teacher ONNX int8 dan **~2.000× lebih cepat** dari teacher PyTorch (throughput 1 core), **16–63× lebih kecil**, dan mempertahankan **~91% macro-F1 teacher** (0,751/0,828 di dev; 0,740/0,836 di 16 bahasa yang tidak ada di data latih teacher). Dengan 447 dok/detik per core, **1 juta artikel/hari (~12 dok/detik) cukup ditangani oleh ~3% dari satu core.**
- Dibanding fastText (model "gaya language-ID" yang Anda jadikan contoh), StaticKD sama cepatnya tetapi **+8,4 poin macro-F1 di bahasa-bahasa lain** dan jauh lebih stabil untuk bahasa dengan data sedikit dan aksara non-Latin.
- **Mode cascade** (opsional): StaticKD menangani artikel dengan confidence ≥ 0,8 (~66% trafik) dan sisanya dikirim ke teacher → micro-F1 dev 0,855 vs 0,858 teacher penuh, dengan hanya ~34% beban komputasi teacher.

**Rekomendasi:** gunakan StaticKD sebagai klasifikator utama pipeline real-time (varian *compact* bila RAM terbatas). Opsional, pasang cascade ke teacher untuk artikel low-confidence bila akurasi maksimum diperlukan dan ada kapasitas batch/offline. Proposal awal (KD + LoRA) **tidak** menyelesaikan masalah kecepatan inferensi (lihat §6).

---

## 1. Review dan laporan performa model Kuzman & Ljubešić (2025)

### 1.1 Profil model
| Aspek | Nilai |
|---|---|
| Model | `classla/multilingual-IPTC-news-topic-classifier` |
| Arsitektur | XLM-RoBERTa-large (24 layer, hidden 1024), 559,9 juta parameter |
| Ukuran file | 2,24 GB (safetensors fp32); versi ONNX int8 dari onnx-community 537 MB |
| Data latih | 15.000 artikel (hr, sl, ca, el) dari EMMediaTopic, dilabeli GPT-4o |
| Input | teks artikel, 512 kata pertama (tokenisasi dipotong di 512 token) |
| Output | 17 label top-level IPTC Media Topic |
| Skor yang dilaporkan paper | macro-F1 0,746 / micro-F1 0,734 pada test set manusia (1.129 dok, **tidak publik**) |

### 1.2 Protokol evaluasi di riset ini
Test set manusia Kuzman (IPTC-test) hanya tersedia atas permintaan ke penulis, sehingga saya membangun tiga set evaluasi:

| Set | Isi | Label | Kegunaan |
|---|---|---|---|
| **dev** | 1.000 dok EMMediaTopic (hr/sl/ca/el), split `dev` | GPT-4o (label yang sama dengan data latih Kuzman) | Tidak pernah dipakai melatih teacher → pembanding adil teacher vs student |
| **llm_test** | 320 dok CC-News, **16 bahasa di luar data latih teacher** (en, de, fr, es, pt, it, ru, pl, tr, ar, zh, ja, id, vi, hi, sw), situs berita yang tidak dipakai saat training | Dianotasi LLM (Claude) dengan **prompt dan deskripsi label persis dari paper Kuzman (Lampiran B)**, tanpa melihat prediksi model mana pun | Menguji generalisasi lintas bahasa |
| **ccnews** | 13.549 dok CC-News, 70 bahasa, dari domain berita yang tidak ada di data latih | Prediksi teacher | Mengukur *fidelity* distilasi (seberapa mirip student dengan teacher) per bahasa |

Semua skor dihitung dengan kode evaluasi yang sama dengan repo benchmark Kuzman (macro/micro-F1 sklearn). Interval kepercayaan 95% dihitung dengan bootstrap (1.000 resampling).

### 1.3 Akurasi teacher
| Set | Macro-F1 | Micro-F1 | CI95 macro |
|---|---|---|---|
| dev (GPT-4o) | 0,820 | 0,858 | 0,779–0,849 |
| llm_test (16 bahasa) | 0,830 | 0,831 | 0,769–0,872 |
| ONNX int8 — dev | 0,828 | 0,860 | |
| ONNX int8 — llm_test | 0,836 | 0,838 | |

Catatan:
- Skor dev lebih tinggi dari 0,746 di paper karena labelnya GPT-4o (label yang ditiru teacher), bukan label manusia. Kesepakatan GPT-4o dengan manusia sendiri hanya ~0,73 (paper).
- Kemampuan lintas bahasa teacher sangat baik: pada 16 bahasa yang tidak pernah dilihat saat fine-tuning, akurasinya tetap 0,83. Ini juga memvalidasi llm_test: kesepakatan anotator LLM dan teacher (83%) setara dengan kesepakatan manusia–GPT-4o di paper.
- Per bahasa (llm_test, akurasi): en 0,95, es 1,00, it 0,95, sw 1,00, de/fr/id/pt/vi 0,85, ar/ja 0,80, pl/tr/zh 0,75, hi/ru 0,65.
- Label tersulit (dev): *lifestyle and leisure* (F1 0,59), *society* (0,67), *conflict* (0,73), *human interest* (0,75). Kebingungan utama: human interest ↔ arts/culture, society/economy → politics, konsisten dengan temuan paper.

### 1.4 Efisiensi teacher di CPU (inti masalah)
Diukur di Intel Core Ultra 7 155H, proses dipin ke P-core (mensimulasikan server kecil dengan core seragam), GPU dimatikan, 1 dokumen per panggilan untuk latensi, batch 8 untuk throughput. Sampel: artikel dev (median 220 kata).

| Konfigurasi teacher | Core | Latensi p50 | Latensi p95 | Throughput | RAM model | Waktu load |
|---|---|---|---|---|---|---|
| PyTorch fp32 (cara pakai di model card) | 1 | 2.600 ms | 4.415 ms | 0,22 dok/s | 2.957 MB | 19 s |
| PyTorch fp32 | 4 | 1.083 ms | 1.710 ms | 0,59 dok/s | 2.961 MB | 11 s |
| ONNX int8 (onnx-community) | 1 | 913 ms | 1.942 ms | 0,62 dok/s | 1.383 MB | 14 s |
| ONNX int8 | 4 | 329 ms | 527 ms | 2,08 dok/s | 1.387 MB | 9 s |

Akurasi versi ONNX int8 tidak turun (dev 0,828/0,860; llm_test 0,836/0,838), jadi baris ONNX int8 adalah kondisi terbaik teacher di CPU.

**Kesimpulan review:** model Kuzman akurat dan kuat lintas bahasa, tetapi di CPU satu artikel butuh ~1–2,6 detik per core. Untuk 1 juta artikel/hari (~12 artikel/detik) dibutuhkan ~20 core (ONNX int8) sampai ~55 core (PyTorch) khusus hanya untuk klasifikasi, dengan RAM 1,4–3 GB per proses. Ini yang membuatnya tidak praktis untuk pipeline real-time di server kecil.

---

## 2. Metode yang diusulkan: StaticKD

```
          (offline, sekali saja, GPU opsional)                         (produksi, CPU)
 CC-News 70 bahasa ─┐
 EMMediaTopic ──────┼─► Teacher XLM-R-large ─► soft label (17 logit) ─┐
                    │                                                  ▼
                    └──────────────────────────► Student: tokenizer ► embedding statis ► mean ► linear
                                                  (init: potion-multilingual, selaras lintas bahasa)
                                                  loss = KL(student/T || teacher/T)·T², T=2, token dropout 0,2
                                                                                        │ lipat W ke tabel
                                                                                        ▼
                                                  Model produksi: tabel 500k token × 17 logit (fp16, 17 MB)
                                                  prediksi = argmax( rata-rata baris token + bias )
```

**Mengapa ini bekerja:**
1. **Distilasi dengan korpus multibahasa besar.** Kuzman hanya punya 15 ribu label GPT-4o di 4 bahasa. Teacher dapat melabeli data sebanyak apa pun secara gratis. Student belajar meniru teacher di 70 bahasa, bukan hanya 4.
2. **Embedding statis yang sudah selaras lintas bahasa.** potion-multilingual (didistilasi dari BGE-M3) memberi kata bermakna sama di bahasa berbeda vektor yang berdekatan. Karena itu student bekerja juga untuk bahasa yang datanya sedikit atau tidak ada (contoh: bahasa Jawa diklasifikasikan dengan benar walau tidak ada di data latih).
3. **Topik berita sebagian besar ditentukan oleh kosakata.** Klasifikasi topik (beda dengan sentimen atau sarkasme) sangat bisa ditangani model bag-of-words. Konteks penuh transformer hanya menambah ~1–2 poin (§4.4).
4. **Pelipatan tabel.** mean(E[ids])·Wᵀ = mean((E·Wᵀ)[ids]), sehingga inferensi hanya berupa penjumlahan 17 angka per token, tanpa perkalian matriks.

---

## 3. Data

| Korpus | Jumlah | Bahasa | Label |
|---|---|---|---|
| EMMediaTopic train | 20.000 | hr, sl, ca, el | GPT-4o + logit teacher |
| CC-News train | 109.842 | 70 | logit teacher |
| CC-News train tambahan | 156.157 | 70 | logit teacher |
| **Total data distilasi** | **285.999** | 70 | |
| CC-News held-out (domain tidak terlihat) | 13.549 | 70 | logit teacher |

CC-News diambil dari arsip 2019–2021, maksimal 150–400 dokumen per situs, dan split held-out dibuat **per domain** (situs held-out tidak pernah muncul di training). Preprocessing sama dengan Kuzman: judul + isi, dipotong ke 512 kata pertama, minimal 75 kata. Pelabelan teacher butuh ~51 dok/detik di RTX 4050 (fp16). Ini langkah offline satu kali.

---

## 4. Hasil eksperimen

### 4.1 Tabel utama (akurasi)
| Model | dev macro | dev micro | llm_test macro | llm_test micro | kesepakatan dgn teacher (ccnews, 70 bhs) |
|---|---|---|---|---|---|
| Teacher XLM-R-large fp32 | 0,820 | 0,858 | 0,830 | 0,831 | 1,000 |
| Teacher ONNX int8 | 0,828 | 0,860 | 0,836 | 0,838 | — |
| MiniLM-L6 + KD (ONNX fp32, 256 token) | 0,763 | 0,801 | 0,749 | 0,766 | 0,814 |
| MiniLM-L6 + KD (ONNX int8) | 0,633 | 0,725 | 0,610 | 0,669 | — |
| **StaticKD (final, 286k dok)** | **0,751** | **0,793** | **0,740** | **0,744** | **0,800** |
| fastText + KD, subword, quantized | 0,728 | 0,764 | 0,656 | 0,669 | 0,698 |
| fastText + KD, subword | 0,721 | 0,767 | 0,642 | 0,675 | 0,701 |
| fastText + KD, kata | 0,717 | 0,764 | 0,619 | 0,666 | 0,711 |
| fastText, label GPT-4o saja (setting Kuzman) | 0,681 | 0,734 | 0,131 | 0,225 | — |
| (referensi paper) TF-IDF + SVC, test manusia | 0,421 | 0,423 | — | — | — |

CI95 bootstrap macro-F1: StaticKD dev 0,708–0,783, llm_test 0,659–0,787; MiniLM dev 0,718–0,792; teacher dev 0,779–0,849, llm_test 0,769–0,872. Selisih StaticKD vs MiniLM **tidak signifikan**; selisih StaticKD vs fastText di llm_test (0,740 vs 0,656) dan vs teacher (0,740 vs 0,830) bermakna.

### 4.2 Ablasi: apa yang membuat StaticKD berhasil
| # | Variasi | dev macro | llm_test macro | Temuan |
|---|---|---|---|---|
| A | fastText, label GPT-4o 4 bahasa | 0,681 | 0,131 | Tanpa data multibahasa, model gagal total di bahasa lain |
| B | fastText, label teacher 70 bahasa | 0,721 | 0,642 | **Distilasi multibahasa adalah kunci** (+51 poin lintas bahasa) |
| C | Static (potion), KD hanya EMMediaTopic 4 bhs | 0,718 | 0,629 | Embedding selaras lintas bahasa saja sudah memberi transfer (0,13 → 0,63) |
| D | Static (embedding didistilasi dari teacher), KD 130k | 0,726 | 0,642 | Embedding dari teacher kalah dari potion untuk lintas bahasa |
| E | Static (potion), KD 130k | 0,732 | 0,723 | Kombinasi B + C: terbaik |
| F | E + token dropout 0,2 | 0,746 | 0,737 | Regularisasi +1,4 poin (seed lain: 0,739 / 0,732) |
| G | E + head MLP 512 | 0,739 | 0,736 | Tidak lebih baik dari F, dan tidak bisa dilipat → ditolak |
| H | F + data 286k (final) | 0,751 | 0,740 | Data ×2 → +0,5 poin: kapasitas model sudah jenuh |
| I | F dipangkas vocab (min count 5) | 0,746 | 0,737 | Pemangkasan tidak merusak akurasi |
| J | F dilipat ke tabel fp16 | 0,746 | 0,737 | **Identik**; ukuran 128 MB → 34 MB |
| K | H + tokenizer dipangkas (min count 10) | 0,751 | 0,730 | RAM 505 → 281 MB, file 23 MB, −1 poin lintas bahasa |
| L | H + tokenizer dipangkas (min count 50) | 0,749 | 0,714 | RAM 190 MB, file 15 MB, −2,6 poin lintas bahasa |

**Kurva data (static potion + KD):** 20k dok (4 bhs) 0,718 / 0,629 → 130k (70 bhs) 0,746 / 0,737 → 286k 0,751 / 0,740.

### 4.3 Per bahasa
- Kesepakatan StaticKD final dengan teacher pada 70 bahasa (ccnews held-out): ≥0,80 untuk 33 bahasa, termasuk es 0,92, bg 0,89, ru 0,87, fa 0,87, en 0,86, ar 0,86, id 0,85, de 0,84, el 0,84, sl 0,84, zh 0,82, hi 0,80, sw 0,80, vi 0,80. 0,70–0,79 untuk 24 bahasa (mis. ja 0,79, pt 0,79, pl 0,78, tr 0,74, hr 0,73). Rendah hanya untuk bahasa yang datanya sangat sedikit (ur 0,47, ms 0,47, am 0,40, ne 0,25, ps 0,22).
- Dibanding fastText+KD, StaticKD jauh lebih stabil di bahasa dengan data sedikit atau aksara non-Latin: be 0,79 vs 0,06; pa 0,81 vs 0,19; zh 0,80 vs 0,41; ka 0,61 vs 0,30.
- llm_test per bahasa (akurasi StaticKD final; teacher dalam kurung): es 1,00 (1,00), en 0,90 (0,95), id 0,85 (0,85), sw 0,85 (1,00), ar/de/pl/pt 0,80 (0,80/0,85/0,75/0,85), fr/it/ja 0,75 (0,85/0,95/0,80), vi/zh 0,70 (0,85/0,75), ru 0,65 (0,65), hi 0,55 (0,65), **tr 0,25 (0,75)**. Hanya 20 dok per bahasa, jadi variansnya besar. Turki adalah kelemahan paling jelas (lihat §7).
- Label tersulit StaticKD (dev F1): *lifestyle and leisure* 0,45, *human interest* 0,61, *society* 0,62. Pola ini sama dengan teacher (0,59 / 0,75 / 0,67), hanya lebih tajam. Label termudah: sport 0,94, weather 0,89, health 0,88.

### 4.4 Efisiensi CPU (inti tujuan)
Setup: Intel Core Ultra 7 155H, GPU dimatikan, proses dipin ke P-core fisik (1 core = CPU 0; 4 core = CPU 0, 10, 12, 14). Latensi = 1 dokumen per panggilan (kasus streaming), throughput = batch 64 (student) atau 8 (transformer). Setiap konfigurasi dijalankan di proses baru.

| Model | Core | Latensi p50 | Latensi p95 | Throughput | RAM model* | Ukuran file | Dependensi inferensi |
|---|---|---|---|---|---|---|---|
| Teacher PyTorch fp32 | 1 | 2.600 ms | 4.415 ms | 0,22 dok/s | 2.957 MB | 2.136 MB | torch, transformers |
| | 4 | 1.083 ms | 1.710 ms | 0,59 dok/s | | | |
| Teacher ONNX int8 | 1 | 913 ms | 1.942 ms | 0,62 dok/s | 1.383 MB | 537 MB | onnxruntime, transformers |
| | 4 | 329 ms | 527 ms | 2,08 dok/s | | | |
| MiniLM-L6 + KD (ONNX fp32) | 1 | 70,6 ms | 106,7 ms | 12,5 dok/s | 1.242 MB | 408 MB | onnxruntime, transformers |
| | 4 | 43,3 ms | 60,8 ms | 20,6 dok/s | | | |
| **StaticKD** | 1 | **1,94 ms** | 6,31 ms | **447 dok/s** | 505 MB | 34 MB | numpy, tokenizers |
| | 4 | 1,70 ms | 3,61 ms | 1.683 dok/s | | | |
| StaticKD-compact | 1 | 1,68 ms | 5,01 ms | 567 dok/s | 281 MB | 23 MB | numpy, tokenizers |
| | 4 | 1,58 ms | 3,47 ms | 2.191 dok/s | | | |
| fastText + KD (quantized) | 1 | 1,36 ms | 3,28 ms | 639 dok/s | 308 MB | 18 MB | fasttext, tokenizers |
| | 4 | 1,52 ms | 3,50 ms | 1.537 dok/s | | | |

\*RAM model = kenaikan RSS setelah model dimuat (di luar interpreter Python ± 250 MB). Pada StaticKD, **~490 MB dari 505 MB adalah struktur internal tokenizer Unigram 500k-token**; tabel modelnya sendiri hanya 19 MB. Karena itu varian compact (vocab tokenizer 335k) memangkas RAM ke 281 MB. Hal yang sama berlaku untuk fastText + tokenizer XLM-R (308 MB).

Catatan metodologi: tanpa pinning, Windows memindahkan thread ke E-core (2,5× lebih lambat) dan core low-power (3,7× lebih lambat), sehingga angka multi-thread tidak stabil (bahkan lebih lambat dari 1 thread). Pinning ke P-core wajib untuk benchmark yang bisa direproduksi.

### 4.5 Confidence dan cascade
Probabilitas softmax StaticKD terkalibrasi dengan baik:

| Ambang confidence | Porsi artikel ditangani student | Akurasi pada porsi itu (dev) | Cascade micro-F1 dev | Cascade macro-F1 dev | Cascade macro-F1 llm_test |
|---|---|---|---|---|---|
| 0 (student saja) | 100% | 0,793 | 0,793 | 0,751 | 0,740 |
| 0,6 | 82% | 0,871 | 0,838 | 0,800 | 0,795 |
| 0,7 | 73% | 0,904 | 0,850 | 0,814 | 0,808 |
| 0,8 | 66% | 0,925 | 0,855 | 0,816 | 0,834 |
| 0,9 | 56% | 0,941 | 0,856 | 0,818 | 0,850 |
| teacher saja | 0% | — | 0,858 | 0,820 | 0,830 |

Dengan ambang 0,8, akurasi setara teacher dapat dicapai dengan hanya ~34% panggilan ke teacher, sekitar 3× lebih hemat dari teacher saja. Namun biaya rata-rata tetap didominasi teacher (~0,34 × 913 ms ≈ 310 ms/dok per core), sehingga cascade cocok untuk jalur semi-real-time/batch, bukan pengganti StaticKD di jalur real-time. Label berconfidence rendah juga bisa diperlakukan sebagai kandidat multi-label, sesuai rencana future work Kuzman.

### 4.6 Eksperimen lanjutan: bigram hashing dan ensemble seed
Dua cara menaikkan akurasi yang tetap berupa lookup tabel (resep final, 286k dokumen, 6 epoch):
- **Bigram hashing**: setiap pasangan token berurutan di-hash (Fibonacci hashing uint64) ke tabel 2^b × 17 yang dilatih langsung dalam bentuk terlipat (init nol, SparseAdam). Rata-rata bigram ditambahkan ke logit unigram. Opsi `--bigram-bits`, `--lr-bigram` di `students/static_student.py`.
- **Ensemble seed**: rata-rata tabel dari 3 seed (`ensemble_tables`). Hasilnya identik dengan rata-rata logit karena modelnya linear, jadi biaya inferensinya nol.

| Varian | dev macro | llm_test macro | Kesepakatan 70 bhs | File | Throughput 1c* |
|---|---|---|---|---|---|
| Baseline seed 0 / 1 / 2 | 0,751 / 0,748 / 0,750 | 0,740 / 0,763 / 0,732 | 0,800 / 0,799 / 0,802 | 34 MB | 600 dok/s |
| Baseline rata-rata ± SD | 0,750 ± 0,002 | 0,745 ± 0,016 | 0,800 ± 0,002 | | |
| Ensemble 3 seed | 0,753 | 0,722 | 0,803 | 34 MB | 600 dok/s |
| Bigram 2^21, lr 1e-2 | 0,738 | 0,722 | 0,804 | 102 MB | 578 dok/s |
| Bigram 2^20, lr 2e-3 | 0,742 | 0,716 | 0,803 | 68 MB | – |

*Diukur ulang pada sesi yang sama (P-core 0). Angka absolutnya lebih tinggi dari Tabel 4.4 karena kondisi mesin berbeda, jadi hanya perbandingan relatifnya yang bermakna.

Paired bootstrap terhadap seed 0 (`evaluation/compare_variants.py`, `results/variant_comparison.csv`):
- Bigram 2^21: kesepakatan +0,004 (p = 0,02), dev −0,013 (p = 0,058), llm_test −0,018 (p = 0,25).
- Ensemble: kesepakatan +0,003 (p = 0,01), dev +0,002 (p = 0,72).

**Temuan:**
1. **Bigram tidak menaikkan akurasi.** Bigram sedikit menaikkan kemiripan dengan teacher di domain CC-News, tetapi menurunkan akurasi terhadap label independen (dev GPT-4o dan test LLM). Dua konfigurasi memberi pola yang sama, dan skor dev/test keduanya berada di bawah rentang tiga seed baseline. Dugaan penyebabnya: bigram dilatih dari nol tanpa inisialisasi lintas bahasa, sehingga lebih banyak menghafal pola khas data distilasi daripada belajar sinyal topik yang tergeneralisasi. Biayanya: file 2–3× lebih besar, RAM +68 MB, latensi p50 +25%.
2. **Ensemble seed juga praktis tidak berpengaruh**: +0,002–0,003, dalam batas noise.
3. **Variasi antar-seed di llm_test (320 dokumen) sekitar 0,016 SD** (rentang 0,031), jauh lebih besar dari di dev (0,002). Selisih < 0,03 di llm_test tidak bisa diinterpretasikan tanpa beberapa seed.
4. Bersama hasil sebelumnya (data ×2, kepala MLP, MiniLM-L6 dengan attention), ini memperkuat kesimpulan bahwa **student kecil tertahan di sekitar 0,75 macro-F1 dev** dengan resep distilasi ini.

### 4.7 Eksplorasi akurasi dengan syarat model tetap ringan → StaticKD v2
Syaratnya: format deployment tetap tabel token + bias, dan biaya inferensi tetap O(jumlah token). Protokolnya: dev dipakai untuk seleksi, kesepakatan 70 bahasa sebagai sinyal pendukung, dan LLM16 sebagai konfirmasi. Kandidat akhir diuji dengan 3 seed. Kode ada di `experiments/explore_inference.py` dan `experiments/run_queue.py`, dengan opsi baru `--temperature`, `--conf-weight`, `--gpt-weight`, `--label-smoothing`, `--max-tokens`, dan `--token-weight {idf,learned}` di `static_student.py`. Hasil lengkapnya di `results/exploration/`.

**Tahap 1, inferensi saja (model v1 tidak diubah):**

| Variasi | dev | LLM16 | Kesepakatan |
|---|---|---|---|
| Baseline (1.024 token) | 0,751 | 0,740 | 0,800 |
| Potong input 128 / 256 / 384 / 512 / 768 token | 0,688 / 0,725 / 0,761 / 0,753 / 0,750 | 0,693 / 0,690 / 0,721 / 0,713 / 0,725 | 0,753 / 0,784 / 0,800 / 0,803 / 0,800 |
| Bobot posisi (16–64 token pertama ×1,5–3) | 0,732–0,758 | 0,733–0,751 | 0,801–0,805 |
| Pooling berbobot idf^0,5 / idf^1 | 0,749 / 0,745 | 0,738 / 0,743 | 0,807 / 0,807 |
| Kalibrasi bias per kelas (5-fold CV di dev) | **0,721** | 0,747 | 0,787 |

Tidak ada yang konsisten lebih baik. Kalibrasi bias overfit, dan memotong input merugikan dokumen multibahasa.

**Tahap 2, training ulang (1 seed kecuali disebut lain):**

| Variasi | dev | LLM16 | Kesepakatan |
|---|---|---|---|
| τ = 0,5 | 0,747 | 0,743 | 0,807 |
| **τ = 1 (3 seed, rata-rata)** | **0,755 ± 0,002** | 0,748 ± 0,015 | **0,808 ± 0,000** |
| τ = 2 (v1, 3 seed, rata-rata) | 0,750 ± 0,002 | 0,745 ± 0,016 | 0,800 ± 0,001 |
| τ = 4 | 0,724 | 0,704 | 0,792 |
| + CE label GPT-4o (bobot 1) | 0,736 | 0,720 | 0,797 |
| + bobot keyakinan teacher (γ = 1) | 0,755 | 0,735 | 0,799 |
| Maks. 512 token (training + inferensi) | 0,758 | 0,706 | 0,803 |
| Pooling berbobot idf^0,5 (dilatih) | 0,748 | 0,737 | 0,801 |
| Bobot token dipelajari (init idf^0,5) | 0,742 | 0,715 | 0,801 |

**Tahap 3, kandidat akhir:**

| Model | dev | LLM16 | Kesepakatan | File | RAM | p50 (1 core) |
|---|---|---|---|---|---|---|
| v1 (τ = 2, seed 0) | 0,751 | 0,740 | 0,800 | 34 MB | 504 MB | 1,30 ms |
| **v2 = τ = 1, ensemble 3 seed** (`models/static/static_v2_table`) | **0,761** | **0,745** | **0,811** | 34 MB | 504 MB | 1,31 ms |
| v1 compact10 | 0,751 | 0,730 | 0,789 | 23 MB | ±281 MB | – |
| **v2 compact10** (`static_v2_table_compact10`) | 0,754 | 0,748 | 0,801 | 23 MB | ±281 MB | – |

Uji statistik:
- Pada level seed (Welch, 3 vs 3 seed), τ = 1 lebih baik dari τ = 2 untuk dev (+0,005; p = 0,014) dan kesepakatan (+0,008; p = 0,007). Untuk LLM16 tidak berbeda (p = 0,82).
- Paired bootstrap v2 vs v1: kesepakatan +0,011 (p < 0,001), dev +0,010 (p = 0,43; 1.000 dokumen terlalu sedikit untuk selisih sebesar ini), LLM16 +0,005 (n.s.).
- v2 compact10 vs v1 compact10: kesepakatan +0,012 (p < 0,001), LLM16 +0,017 (n.s.).

**Kesimpulan eksplorasi:**
1. Satu-satunya perbaikan yang konsisten adalah **suhu distilasi τ = 1**. Soft label yang lebih tajam lebih cocok untuk student berkapasitas rendah: τ tinggi memaksa student meniru distribusi ekor yang tidak mampu direpresentasikannya.
2. **Ensemble seed** memberi tambahan kecil tanpa biaya inferensi, karena model linear sehingga tabel bisa dirata-rata.
3. **v2 menggantikan v1 sebagai model rekomendasi.** Arsitektur, ukuran, dan kecepatannya identik, dengan akurasi +0,5–1,1 poin.
4. Kenaikannya kecil. Batas atas sekitar 0,75–0,76 macro-F1 dev untuk student bag-of-tokens tetap berlaku. Semua ide lain (bigram, bobot token, bobot keyakinan, label GPT-4o, kalibrasi bias, pemotongan input) tidak membantu atau malah merugikan.

---

## 5. Seberapa "terbukti" solusinya?

| Kriteria kebutuhan | Target | Hasil StaticKD |
|---|---|---|
| Cepat seperti fastText LID | latensi ms, ratusan dok/s per core | 1,9 ms/dok, 447 dok/s per core (compact: 567); setara fastText (639 dok/s) |
| Ringan | < 100 MB, tanpa GPU | 34 MB file (compact 23 MB), RAM 281–505 MB, tanpa PyTorch; hanya numpy + tokenizers |
| Semua bahasa | multibahasa seperti fastText LID | Dilatih 70 bahasa; embedding mendukung 101 bahasa; teruji di 16 bahasa yang tidak ada di data teacher |
| Akurat | mendekati model Kuzman | 91% macro-F1 teacher (dev & 16 bahasa); cascade menutup gap sepenuhnya |
| Input sama dengan Kuzman | teks artikel (512 kata) | Ya |

---

## 6. Catatan terhadap proposal awal (KD + LoRA)

- **LoRA mengurangi parameter yang *dilatih*, bukan biaya *inferensi*.** Setelah adapter di-merge, model LoRA sama besar dan sama lambatnya dengan backbone. Jika backbone-nya XLM-R, masalah latensi CPU tidak berubah.
- **KD ke transformer kecil** (setara "student" di proposal) sudah diuji di sini: mMiniLMv2-L6 + KD mencapai 0,763 / 0,749 macro-F1 (hanya +1,2/+0,9 poin di atas StaticKD, tidak signifikan). Biayanya 71 ms/dok dan 12 dok/s per core, ~36× lebih lambat dan 2,5× lebih boros RAM dibanding StaticKD, dan kuantisasi int8 dinamis merusak akurasinya (0,633). Jadi untuk tujuan *real-time di CPU*, transformer kecil adalah trade-off yang kurang menguntungkan.
- Research gap proposal ("model IPTC tidak melaporkan efisiensi") tetap relevan dan dijawab langsung oleh §1.4 dan §4.4.
- Kontribusi yang bisa diklaim untuk paper: (1) laporan efisiensi pertama model IPTC Kuzman di CPU; (2) metode distilasi teacher → static-embedding multibahasa untuk klasifikasi topik berita, dengan ablasi yang menunjukkan kontribusi tiap komponen; (3) evaluasi lintas bahasa pada 16 + 70 bahasa; (4) model 34 MB dengan throughput ratusan dok/detik per core.

---

## 7. Keterbatasan dan ancaman validitas

1. **Test set manusia Kuzman tidak diakses.** Dev berlabel GPT-4o dan llm_test berlabel LLM (satu anotator, 320 dok, 20 per bahasa → CI lebar). *Langkah wajib berikutnya:* minta IPTC-test ke taja.kuzman@ijs.si dan laporkan skor StaticKD di sana (repo benchmark mereka menerima submisi).
2. llm_test dianotasi oleh Claude, yaitu model dari keluarga yang sama dengan asisten yang menjalankan riset ini. Anotasi dibuat buta terhadap prediksi model mana pun, dengan prompt Kuzman. Tetap disarankan validasi manusia pada sampel.
3. Batas atas student adalah teacher. Kesalahan dan bias teacher (misalnya pada *lifestyle*, *society*) ikut tertiru.
4. Bahasa dengan data sangat sedikit (ps, ne, am, ur, ms) masih lemah. Solusinya menambah data berita untuk bahasa tersebut lalu melabelinya dengan teacher. **Bahasa Turki** anjlok di llm_test (0,25 vs teacher 0,75) walau kesepakatan di ccnews 0,74. Kemungkinan akibat sedikitnya domain berita Turki di data latih (1.465 dok dari beberapa situs saja); perlu diselidiki. **Bahasa Indonesia sudah baik** (llm_test 0,85 = teacher; kesepakatan 0,85), tetapi datanya didominasi beberapa portal (okezone, tribunnews, metrotvnews).
5. Benchmark CPU dilakukan di laptop (Core Ultra 7 155H, dipin ke P-core). Angka absolut di server bisa berbeda, tetapi rasio antar model tetap informatif.

---

## 8. Langkah selanjutnya yang disarankan
1. Evaluasi di IPTC-test manusia (minta ke penulis).
2. Tambah data untuk bahasa lemah, terutama bahasa target industri Anda (misalnya Indonesia/Melayu), lalu distilasi ulang (murah: hanya perlu GPU untuk pelabelan).
3. RAM masih didominasi tokenizer. Varian compact sudah menurunkannya ke 281 MB. Langkah berikutnya: tokenizer word-lookup sederhana (80% vocab potion adalah kata utuh) atau distilasi ulang dengan vocab ~100k.
4. Kembangkan ke level-2 IPTC secara hierarkis, dengan resep distilasi yang sama.

---

## Lampiran A. Lokasi artefak
| Artefak | Lokasi |
|---|---|
| Model final (produksi) | `models/static/static_final_table/` (model.npz + tokenizer.json) |
| Model final, varian RAM rendah | `models/static/static_final_table_compact10/` |
| Semua hasil akurasi (per bahasa, per label, confusion matrix, prediksi) | `results/accuracy/*.json` |
| Benchmark CPU | `results/cpu_benchmark.jsonl` |
| Ringkasan | `results/summary.md` |
| Analisis confidence/cascade | `results/confidence_*.json` |
| Test set LLM (16 bahasa) | `data/processed/llm_test.jsonl` |
| Kode & cara reproduksi | `code/README.md` |

## Lampiran B. Cara pakai model final
```python
import sys; sys.path.insert(0, "code")
from students.static_student import StaticPredictor
from config import LABELS

model = StaticPredictor("models/static/static_final_table")
docs = ["Pemerintah resmi menaikkan harga BBM bersubsidi mulai Sabtu ..."]
print([LABELS[i] for i in model.predict(docs)])   # ['economy, business and finance']
```
