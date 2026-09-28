# Rancangan Penelitian (versi final)

Dokumen ini menetapkan kerangka penelitian sebelum hasil akhir dihitung. Notebook
(`notebooks/StaticKD_IPTC_final.ipynb`) dan jurnal (`jurnal_latex/main.tex`) mengikuti struktur ini.
Istilah teknis berbahasa Inggris sengaja tidak diterjemahkan.

## 1. Judul

**StaticKD: *Knowledge Distillation* Pengklasifikasi Topik Berita IPTC Multibahasa ke Tabel Logit Token Statis untuk Inferensi Ringan di CPU**

Judul pendek (*running head*): *StaticKD: Klasifikasi Topik Berita IPTC Multibahasa di CPU*

Perubahan dari proposal awal ("Knowledge Distillation & LoRA"): LoRA dikeluarkan dari topik karena LoRA
hanya menurunkan biaya *training*, bukan biaya inferensi. Setelah adapter digabungkan, model tetap sebesar
*backbone*-nya. Masalah yang ingin dipecahkan adalah biaya inferensi. Transformer kecil hasil KD tetap
diuji sebagai *baseline*.

## 2. Latar belakang (alur argumen)

1. **Kebutuhan.** *Pipeline* analitik berita memproses jutaan artikel per hari dalam banyak bahasa, dan
   perlu menyaring artikel berdasarkan topik terstandar sebelum analisis lanjut. IPTC Media Topic adalah
   taksonomi yang dipakai kantor berita besar (17 topik level atas).
2. **Solusi terbaik yang ada mahal.** Pengklasifikasi IPTC multibahasa terbuka dari Kuzman & Ljubešić
   (IEEE Access 2025) berbasis XLM-RoBERTa-large (560 juta parameter). Model ini akurat, tetapi di CPU
   butuh waktu dalam hitungan detik per artikel per core. Paper aslinya tidak melaporkan biaya inferensi.
3. **Alternatif murah belum memadai.**
   - fastText cepat tetapi tidak punya transfer lintas bahasa. Ia butuh data berlabel untuk setiap bahasa,
     padahal data berlabel IPTC hanya tersedia untuk 4 bahasa.
   - Transformer kecil (MiniLM, DistilmBERT) masih puluhan milidetik per artikel.
   - PEFT/LoRA tidak menurunkan biaya inferensi.
4. **Peluang.**
   - *Teacher* yang kuat lintas bahasa dapat melabeli berita tak berlabel dalam bahasa apa pun tanpa biaya
     anotasi.
   - *Static embedding* multibahasa yang selaras lintas bahasa (Model2Vec potion-multilingual) memberi
     representasi kata berbiaya lookup.
   - Klasifikasi topik berita sangat ditentukan oleh kosakata, sehingga model *bag-of-tokens* berpotensi
     cukup.
5. **Masalah evaluasi.** Test set manusia IPTC (Kuzman) tidak publik dan hanya mencakup 4 bahasa. Klaim
   "multibahasa" membutuhkan test set multibahasa dengan label yang independen dari *teacher*.

## 3. Rumusan masalah (*research questions*)

- **RQ1 (biaya).** Berapa biaya inferensi CPU pengklasifikasi IPTC *teacher* (latensi, *throughput*, RAM,
  ukuran), dan berapa besar biaya itu dapat diturunkan oleh StaticKD?
- **RQ2 (akurasi).** Seberapa besar akurasi *teacher* yang dipertahankan StaticKD pada label yang
  independen dari *teacher*, di bahasa yang dilihat maupun tidak dilihat saat distilasi, dibandingkan
  *baseline* berbiaya serupa (fastText, TF-IDF + SVM) dan berbiaya lebih tinggi (MiniLM-L6, DistilmBERT)?
- **RQ3 (komponen).** Komponen apa yang menentukan keberhasilan distilasi ke *student* berkapasitas sangat
  rendah? Yang diuji: *soft label*, data distilasi multibahasa, inisialisasi embedding yang selaras lintas
  bahasa, *fine-tuning* embedding, *token dropout*, *temperature*, dan *ensemble* tabel.
- **RQ4 (generalisasi).** Di bahasa, keluarga bahasa, label, dan sumber label mana StaticKD lemah, dan
  faktor apa yang menjelaskannya? Faktor yang ditelaah: jumlah data distilasi, keyakinan *teacher*, sistem
  tulisan, dan kehadiran bahasa di data distilasi.

## 4. Batasan masalah

1. Klasifikasi *single-label* ke 17 topik IPTC Media Topic level atas. Level 2 ke bawah dan *multi-label*
   di luar cakupan.
2. Input berupa judul + isi artikel, 512 kata pertama, sama dengan Kuzman & Ljubešić.
3. *Teacher* tetap, yaitu `classla/multilingual-IPTC-news-topic-classifier`, dan tidak dilatih ulang.
4. Efisiensi diukur hanya di CPU, pada satu mesin (Intel Core Ultra 7 155H, dipin ke P-core) dengan 1 dan
   4 *thread*. GPU hanya dipakai untuk pelabelan *teacher* dan *training*.
5. Label uji berasal dari anotasi manusia (MasakhaNEWS, MN-DS) dan kategori/rubrik penerbit (silver label
   yang dipetakan ke pohon IPTC). Test set manusia Kuzman yang privat tidak dipakai.
6. Bahasa uji terbatas pada 47 bahasa yang datanya tersedia. Bahasa lain tidak diklaim.

## 5. Kebaruan (*novelty*)

1. **Metode.** Distilasi *teacher* transformer multibahasa ke *student* yang, setelah *training*, berupa
   satu tabel 17 logit per token. *Student* diinisialisasi dari *static embedding* yang selaras lintas
   bahasa, dan kepalanya yang linear dilipat ke dalam tabel. Biaya inferensinya setara *language
   identification* (NumPy saja). Kami juga menunjukkan secara empiris dua sifat khusus *student*
   berkapasitas sangat rendah:
   - *temperature* rendah (τ = 1) lebih baik daripada τ ≥ 2 yang lazim;
   - *ensemble* beberapa *seed* dapat dilipat menjadi satu tabel tanpa biaya inferensi, karena modelnya
     linear.
2. **Evaluasi.** Test set IPTC multibahasa pertama dengan 12.744 artikel dari 47 bahasa, dibangun dari
   label yang independen dari *teacher*:
   - rubrik penerbit (JSON-LD schema.org CC-NEWS 2024–2026) dan kategori dataset publik, dipetakan ke
     pohon IPTC Media Topic resmi dengan aturan "hanya jika seluruh isi rubrik masuk satu topik level atas";
   - deduplikasi lintas sumber dan kontrol kontaminasi n-gram terhadap seluruh data latih;
   - *site-cluster bootstrap* untuk interval kepercayaan.
3. **Empiris.** Audit biaya CPU pertama untuk pengklasifikasi IPTC terbuka, beserta *frontier*
   akurasi–biaya dari tujuh keluarga model.

## 6. Kontribusi

1. StaticKD v2: model 34 MB yang hanya butuh `numpy` + `tokenizers`, dengan *throughput* ratusan artikel
   per detik per core. Tersedia juga varian *compact* 23 MB dengan RAM lebih rendah.
2. Analisis komponen (ablasi 5 *seed*) dan analisis *temperature*/*ensemble* untuk *student* berkapasitas
   sangat rendah.
3. Unified IPTC test set: manifest (URL/ID + label + asal label), skrip rekonstruksi, dan tabel pemetaan
   rubrik → IPTC yang tervalidasi terhadap pohon IPTC.
4. Benchmark CPU yang dapat direproduksi, termasuk protokol *pinning* untuk CPU *hybrid*.
5. Kode, notebook, dan model yang dirilis (GitHub + Hugging Face).

## 7. Protokol evaluasi (dibekukan sebelum pengujian akhir)

| Peran | Set | Label | Dipakai untuk |
|---|---|---|---|
| Validasi internal | 3% data distilasi | *teacher* | pemilihan *epoch* |
| *Development* | EMMediaTopic dev (1.000 dok, 4 bahasa) | GPT-4o | pemilihan resep (τ, *ensemble*) |
| *Fidelity* | CC-News *held-out* (12.554 dok, 63 bahasa, situs yang tidak muncul di data distilasi) | *teacher* | kemiripan dengan *teacher* |
| **Uji akhir** | Unified test (12.744 dok, 47 bahasa) | manusia / penerbit | semua klaim akurasi |

- Resep StaticKD v2 (τ = 1, *ensemble* 3 *seed*) ditetapkan dari eksperimen *development* **sebelum** set
  uji gabungan dibangun. Tidak ada hiperparameter yang dipilih berdasarkan set uji.
- Metrik utama adalah macro-F1 atas label yang hadir di referensi (sama dengan Kuzman). Metrik tambahan:
  akurasi, macro-F1 rata-rata per bahasa, dan rasio terhadap *teacher*.
- Interval kepercayaan dihitung dengan *site-cluster bootstrap* (1.000×). Perbandingan model memakai
  *paired cluster bootstrap* (2.000×) dengan koreksi Holm. Perbedaan akurasi diuji dengan McNemar eksak.
  Variasi antar-*seed* diuji dengan Welch.
- Analisis subkelompok:
  - asal label (manusia / kategori penerbit / rubrik JSON-LD);
  - wilayah bahasa;
  - tingkat keragaman situs;
  - bahasa ada atau tidak di data distilasi;
  - per label.
- Uji sensitivitas: subset dengan maksimal 5 artikel per situs per sel.

## 8. Rencana eksperimen

| Kelompok | Isi | Seed |
|---|---|---|
| Resep utama | StaticKD (τ = 1, potion, *token dropout* 0,2, 286 rb dok) | 10 |
| Ablasi | *hard label*; EMMediaTopic saja; 130 rb dok; init acak; embedding beku; tanpa *token dropout*; kepala MLP | 5 per varian |
| *Temperature* | τ ∈ {0,5; 1; 2; 4} | 5 (τ = 1: 10) |
| Hasil negatif | bigram *hashing*; *pooling* idf; bobot keyakinan; CE label GPT-4o | 3 |
| *Ensemble* | ukuran k ∈ {1, 2, 3, 5, 10} | kombinasi dari 10 *seed* |
| *Baseline* | fastText (label GPT-4o / *teacher*; kata / subword; terkuantisasi), TF-IDF + SVM, MiniLM-L6 KD, DistilmBERT KD, *teacher* PyTorch & ONNX int8 | 1 |
| Efisiensi | semua model *deployable*, 1 & 4 core, P-core terpin, satu sesi | – |
