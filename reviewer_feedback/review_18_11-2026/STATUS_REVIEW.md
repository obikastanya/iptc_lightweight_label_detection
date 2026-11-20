# Status Tindak Lanjut Review 16 November 2026

Sumber:
- `02_Review_StaticKD_IPTC_MINOR.pdf` — keputusan **Minor Revision** (selanjutnya: **Rm**)
- `02_StaticKD.pdf` — keputusan **Major Revision** (selanjutnya: **RM**)

Naskah yang direview masih versi dua penulis (sebelum commit `dd504a6`).

Status: `Belum` · `Berjalan` · `Menunggu keputusan` · `Sudah ada` (sudah tertangani di paper, cukup dijelaskan di surat jawaban) · `Selesai`

Terakhir diperbarui: 18 November 2026 (malam): T1, T2, T4, T5 diterapkan ke main.tex dan main_id.tex; E2–E4 selesai dan dimasukkan ke paper serta script notebook (notebook belum dibangun ulang).

## 1. Validitas label (eksperimen audit)

| ID | Temuan | Sumber | Status | Catatan |
|---|---|---|---|---|
| L1 | Label editorial (JSON-LD section, kategori penerbit) belum diverifikasi per artikel; agreement dengan teacher bukan bukti kebenaran label | RM mayor 1 | Selesai | Audit selesai. Prosedur pra-registrasi: ketat 79,3% [74,4–83,9], longgar 81,9% [77,1–86,3]. Pemeriksaan ulang pasca-buka-blind (19 Nov): 0 kesalahan pemetaan, 42 ketidakcocokan rubrik nyata, 3 galat LLM (selebritas, fesyen); setelah keputusan penulis atas A027/A032/A222: ketat 80,4% [75,7–84,8], longgar 86,5% [82,6–90,4]. Peringkat model tidak berubah. Semua penyimpangan dicatat di protocol.md. |
| L2 | Anotator kedua untuk κ antar-manusia | RM mayor 1 | Selesai | Keputusan 18 Nov: tidak ada anotator kedua; paper menyebut audit tanpa κ. |
| L3 | MasakhaNEWS ikut diaudit atau tidak | — | Selesai | Keputusan 18 Nov: MasakhaNEWS tidak diaudit. |
| L4 | Perbarui paper dengan hasil audit (metode, hasil, Limitations, Acknowledgment, kontribusi, supplementary, notebook, versi Indonesia) | — | Menunggu keputusan | Usulan 8 butir siap (19 Nov), menunggu persetujuan penulis. |

## 2. Perbaikan teks paper

| ID | Temuan | Sumber | Lokasi | Status | Catatan |
|---|---|---|---|---|---|
| T1 | "Teacher bounds the performance of the student from above" keliru; StaticKD mengalahkan teacher di 7 dari 47 bahasa | RM konsistensi 3 | main.tex:743 | Selesai | |
| T2 | 682× adalah rasio throughput (rasio latensi ±592×); beri label yang tepat | RM audit 3, Rm 4.3 | Abstrak, Results, Conclusion | Selesai | 682× tetap satu-satunya headline kecepatan. |
| T3 | Novelty jangan bertumpu pada folding (T = EWᵀ bukan prinsip baru); poros klaim = model ringan, memori, deployment sederhana vs Model2Vec | Rm 2, RM mayor 3 | Pendahuluan (kalimat novelty) | Sudah ada | Diverifikasi 18 Nov: folding sudah diatribusikan ke Singh dkk. dan klaim memori vs Model2Vec sudah ada; tidak ada perubahan teks, dijawab di surat jawaban. |
| T4 | Nyatakan estimand: interval bersyarat pada kumpulan artikel dan situs ini; 47 bahasa bukan 47 replikasi independen | RM mayor 2 | Protokol / Limitations | Selesai | |
| T5 | Definisi bootstrap: label kosong per resample, cara p bootstrap, keluarga koreksi untuk TOST | RM mayor 4 | Protokol evaluasi | Selesai | Diterapkan 18 Nov (EN+ID): macro-F1 per resample atas label yang muncul, definisi p bootstrap dua sisi, dan p TOST dinyatakan tidak dikoreksi (keputusan penulis; Holm akan menggugurkan ekuivalensi Model2Vec p 0,050 dan hard label p ±0,12). |
| T6 | Beri nama eksplisit estimand macro-F1 (global 0,545 vs rerata per bahasa 0,622) | RM konsistensi 1 | Caption/header tabel hasil utama | Sudah ada | Diverifikasi 18 Nov: tabel utama sudah memisahkan Macro-F1 dan Macro-F1/language; tidak ada perubahan teks. |
| T7 | Cascade: throughput dari model biaya; threshold dipilih di test set; jangan kesankan 0,602 lebih baik dari 0,600 | RM audit 7, mayor 5 | Results (cascade) | Selesai | Diterapkan 18 Nov (EN+ID): θ = 0,8 dipilih di development set (0,822 vs teacher 0,823); cascade setara dengan teacher dalam ±0,01 (TOST p<0,001); pengukuran end-to-end 2,18 doc/s vs rumus 2,20. |
| T8 | Ekuivalensi dengan Model2Vec bersyarat pada test set ini | RM audit 5 | Results | Sudah ada | Diverifikasi 18 Nov: klaim selalu disebut test macro-F1; estimand kini eksplisit lewat T4; tidak ada perubahan teks. |
| T9 | Uji validitas via teacher membandingkan sumber berbeda (domain dan kesulitan berubah) | RM konsistensi 5 | Label Validity Check | Belum | Usulan kalimat siap (dibahas 19 Nov); diterapkan bersama L4. |

## 3. Sudah tertangani di paper (untuk surat jawaban)

| ID | Temuan | Sumber | Status | Letak di paper |
|---|---|---|---|---|
| S1 | Jangan atribusikan manfaat ke dark knowledge | RM mayor 3 | Sudah ada | Diskusi: soft label tidak lebih baik dari hard label |
| S2 | Hard label dan embedding beku sebagai hasil negatif jangan disembunyikan | RM konsistensi 4 | Sudah ada | Results ablasi, Diskusi |
| S3 | ECE test 0,139 belum terkalibrasi baik | RM konsistensi 5 | Sudah ada | Results kalibrasi ("overconfident on the out-of-distribution test set") |
| S4 | Single-label vs multi-label | RM mayor 1 | Sudah ada | Limitations |
| S5 | Overlap satu situs antarsumber | RM konsistensi 2 | Sudah ada | Catatan tabel komposisi test set |
| S6 | Ukuran file bukan RAM (15× file, ±2× RAM) | RM audit 6 | Sudah ada | Abstrak, Results efisiensi |
| S7 | Kalimat kalibrasi ekspektasi di abstrak | Rm 4.2 | Sudah ada | Abstrak menyebut 91,1% dari skor teacher (0,598) |
| S8 | Leave-publisher-out dan sensitivitas dominasi situs | RM mayor 2 | Sudah ada | Leave-one-source-out dan subset maks. 5 artikel per situs |

## 4. Di luar teks paper (tindakan penulis)

| ID | Temuan | Sumber | Status | Catatan |
|---|---|---|---|---|
| A1 | Isi [TODO]: URL GitHub dan Zenodo DOI, URL Hugging Face, biografi tiga penulis | Rm 4.1, RM 6.6 | Belum | URL GitHub = repo rilis `obikastanya/static_kd_iptc_classifier`. |
| A2 | Repo rilis harus bisa diakses publik (reviewer mendapat HTTP 404) | RM 9 | Belum | |
| A3 | Sinkronkan folder `dist/` setelah paper final | — | Belum | Menunggu aba-aba penulis. Ada sisa dua file dari 11 November: komentar kepala `paper/main.tex` dan PDF duplikat `static_kd_iptc_classfier.pdf`. |
| A4 | Bangun ulang notebook, cek konsistensi angka di abstrak/tabel/diskusi, dan susun surat jawaban | Rm 4.3 | Belum | Sebelum sinkronisasi dist. |

## 5. Butuh eksperimen baru (belum diputuskan)

| ID | Temuan | Sumber | Status | Catatan |
|---|---|---|---|---|
| E1 | Sparse Distillation sebagai baseline pada data dan budget sama | RM mayor 3 | Berjalan | Keputusan 20 Nov: dijalankan sebagai Sparse-Distillation-style baseline pada data dan budget yang sama (tokenizer sama, embedding n-gram dari nol, head MLP, KL, dua ukuran tabel ±0,13B dan ±0,27B, tuning setara di dev set). Agen terpisah; hasil di `results/review_experiments/e1/`. |
| E2 | Cascade diukur end-to-end (overhead routing, batching), threshold dari development set | RM mayor 5 | Selesai | Hasil di `results/review_experiments/`; data benchmark disalin ke `results/final/cascade_*.jsonl`; cell notebook ditambahkan di `src/notebook_build/part5.py`. |
| E3 | Selisih logit maksimum dan perubahan argmax fp32 → fp16 | RM audit 4 | Selesai | Prediksi dari tabel fp16 tersimpan identik (0 label berbeda); paper: satu kalimat di bagian folding. |
| E4 | CI berpasangan untuk rasio retensi 91,1% | RM audit 2 | Selesai | Retensi test 91,1% [88,7–94,8%], dev 92,6% [88,4–96,3%]; ditambahkan di Results; cell notebook di `src/notebook_build/part4.py`. |
