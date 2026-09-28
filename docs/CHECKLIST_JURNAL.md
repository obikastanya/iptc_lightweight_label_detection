# Checklist Kelengkapan Jurnal (dipakai saat audit akhir)

Setiap butir dicek pada `jurnal_latex/main.tex` beserta `sections/*.tex`. Kolom "Bukti di notebook"
menunjuk ke bagian `notebooks/StaticKD_IPTC_final.ipynb` tempat angka atau artefak tersebut dihasilkan.

| # | Informasi penting | Lokasi di jurnal | Bukti di notebook |
|---|---|---|---|
| 1 | Masalah, motivasi biaya, dan celah penelitian (biaya CPU teacher tidak dilaporkan) | I | 10.1 |
| 2 | RQ1–RQ4 dan kontribusi yang terukur | I | – |
| 3 | Penelitian terkait 2022–2026 yang peer-reviewed, termasuk karya terdekat (Sparse Distillation) dan tabel posisi | II, Tabel related | – |
| 4 | Sumber data, ukuran, bahasa, lisensi, dan praproses (512 kata, ≥ 75 kata) | III-C, III-D | 3.1–3.3 |
| 5 | Split dan pencegahan kebocoran (domain-disjoint held-out, dedup, kontaminasi 8-gram) | III-C, III-D | 3.2–3.3 |
| 6 | Konstruksi set uji: sumber, aturan pemetaan IPTC, filter, sampling, funnel, komposisi | III-D, Tabel testset | 3.3 |
| 7 | Detail teacher dan pelabelan (fp16, GPU, logit penuh) | III-E | 4 |
| 8 | Arsitektur student, loss, optimizer, semua hiperparameter, seed, pemilihan epoch | III-F, Tabel hparams | 5, 6.1 |
| 9 | Pelipatan tabel dan ensemble (bukti kesetaraan numerik) | III-F | 5.5, 6.2 |
| 10 | Baseline dengan data yang sama dan hiperparameter lengkap | III-G, Tabel hparams | 6.3 |
| 11 | Protokol evaluasi: metrik, CI (site-cluster bootstrap), uji berpasangan, Holm, Welch | III-H | 7 |
| 12 | Protokol efisiensi: CPU saja, pinning, thread, sampel, perangkat | III-I | 10.1 |
| 13 | Hasil utama dengan CI, uji signifikansi, dan persentase terhadap teacher | IV-A | 8.1–8.2 |
| 14 | Analisis subkelompok: asal label, data distilasi, wilayah, keragaman situs | IV-B | 8.3 |
| 15 | Uji sensitivitas (batas per situs) dan validitas label silver | IV-B | 8.4–8.5 |
| 16 | Ablasi 5 seed, temperature, ukuran ensemble, hasil negatif | IV-C | 9 |
| 17 | Efisiensi: latensi, throughput, RAM, ukuran, dependensi, frontier akurasi–biaya | IV-D | 10.1 |
| 18 | Per bahasa, faktor penentu (Spearman, Kruskal), per label, confusion | IV-E | 10.2–10.3 |
| 19 | Kalibrasi (ECE) dan cascade | IV-F | 10.4 |
| 20 | Pembahasan: mengapa bag-of-tokens cukup, capacity gap/τ, fidelity vs akurasi, implikasi praktis | V | – |
| 21 | Keterbatasan / ancaman validitas | VI | 12 |
| 22 | Kesimpulan dan pekerjaan lanjutan | VII | – |
| 23 | Ketersediaan kode, model, dan data (manifest; hak cipta; lisensi) | akhir | 11 |
| 24 | Pernyataan penggunaan AI, lingkungan perangkat lunak, perangkat keras, waktu training | akhir, III-J | 1, 6.1 |
| 25 | Semua angka memakai `\val{}` atau tabel `generated/`, dan tidak ada "??" di PDF | seluruh naskah | 11 |
