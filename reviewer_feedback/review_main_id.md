# Review Naskah StaticKD ([main_id.pdf](file:///D:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.pdf))

Review ini memeriksa seluruh isi [main_id.pdf](file:///D:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.pdf) (beserta baris sumbernya di [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex)) dengan fokus pada dua aspek:
1. **Notes development, meta-definisi internal, dan artefak draf** yang tidak sepantasnya muncul (atau harus diubah cara penyajiannya) dalam makalah ilmiah.
2. **Paragraf yang tidak sinambung dan kalimat yang tidak saling terkait** atau tidak jelas tujuan argumennya.

> [!IMPORTANT]
> **Diagnosis Akar Masalah Penulisan Naskah**
> Secara keseluruhan, masalah alur pada naskah ini berasal dari empat pola berulang:
> 1. **Penghilangan kata transisi secara ekstrem (*staccato fact-dumping*)**: Demi menghindari gaya bahasa AI, hampir semua penanda hubungan sebab-akibat, pertentangan, dan tujuan (*sehingga*, *akibatnya*, *namun*, *sebaliknya*, *untuk mengatasi hal tersebut*) ikut terhapus. Hasilnya, kalimat-kalimat berdiri sendiri seperti daftar fakta lepas dan pembaca harus menebak sendiri tujuan kalimat tersebut.
> 2. **Penggabungan paksa dengan kata `"dan"`**: Banyak kalimat menggabungkan dua fakta yang berbeda kategori menggunakan kata `"dan"` (misalnya arsitektur komputasi digabung dengan jumlah bahasa pelatihan, atau panjang token digabung dengan spesifikasi laptop *benchmark*).
> 3. **Kebocoran buku log eksperimen (*lab-notebook leaks*)**: Naskah menceritakan urutan coba-coba kronologis saat pengembangan (mis. membatasi 150 dokumen/situs lalu di langkah berikutnya menambah 156 ribu dokumen dengan batas 400/situs; menyebut set pilot 320 artikel; menyebut "4 tambahan run selama pengembangan").
> 4. **Catatan glosarium tempelan (*"disebut X dalam makalah ini"*)**: Definisi istilah ditempelkan secara kaku di akhir paragraf seperti catatan internal penulis, bahkan sering muncul *setelah* istilah tersebut dipakai di halaman-halaman sebelumnya.

---

## Bagian 1: Temuan *Notes Development*, *Meta-Notes*, dan Artefak Draf

### 1.A. Placeholder `[TODO]` dan Artefak Draf yang Masih Tercetak

| Lokasi PDF | Baris Sumber | Kutipan di Naskah | Catatan Masalah & Perbaikan |
| :--- | :--- | :--- | :--- |
| Hal. 17, Ketersediaan Kode | [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L671) | `[TODO: URL GitHub dan DOI Zenodo]` dan `[TODO: URL Hugging Face]` | Placeholder draf masih tercetak warna merah di PDF. Isi dengan tautan repositori final atau penyamaran *double-blind* jika diperlukan. |
| Hal. 18–19, Biografi | [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L679-L685) | `OBI KASTANYA [TODO: biografi penulis]` & `ANANTA DWI PRAYOGA ALWY [TODO: biografi penulis]` | Masih berupa catatan TODO merah dan menyebabkan halaman 19 nyaris kosong (hanya 1 baris). |
| Hal. 1–19, Header & Metadata | [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L27-L38) | `VOLUME 11, 2023`, `OBI KASTANYA AND ANANTA...`, header bahasa Inggris, serta seluruh teks di dalam **Gambar 1–7** | Tahun template masih `2023` (padahal isi riset 2024–2026), kata hubung penulis masih `AND`, *running header* belum berbahasa Indonesia, dan Gambar 1 memuat catatan gaya percakapan `offline, once (a GPU only speeds it up)`. |

### 1.B. Catatan Dapur Pengembangan (*Chronological Lab-Notebook Leaks*)

1. **Kontradiksi Pemilihan Resep & Set Pilot 320 Artikel (Hal. 5, Bagian IV-A — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L208))**
   > *"Semua pengaturan parameter dan pilihan resep StaticKD ditentukan pada set ini sebelum test set dibangun. Satu set berisi 320 artikel berlabel LLM yang dijelaskan dalam materi suplemen juga memberi masukan bagi dua dari pilihan tersebut: temperature $\tau = 1$ dan ensemble tiga seed."*
   - **Masalah**: Kalimat kedua secara langsung meralat kalimat pertama (*"Semua pengaturan..."*) dan membocorkan catatan eksperimen eksploratif internal (set 320 artikel yang bahkan tidak tercantum di Tabel 2). Frasa *"sebelum test set dibangun"* juga bernada pembelaan kronologis dapur riset.
   - **Saran**: Nyatakan pembagian peran data secara bersih: pemilihan hiperparameter dilakukan pada *development set* tanpa menyentuh *test set*; jika set 320 artikel memang tidak esensial (karena $\tau=1$ dan *ensemble* juga sudah diuji di Tabel 7 dan Gambar S4), hapus dari naskah utama.

2. **Urutan Kronologis Penambahan Data yang Kontradiktif (Hal. 6, Bagian IV-B — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L239-L241))**
   > *"3) Pertahankan paling banyak 150 dokumen per situs untuk menghindari bias dari konten serupa dalam satu situs.*
   > *4) Sisihkan semua dokumen dari sebagian situs sebagai split held-out yang memuat paling banyak 20% dari tiap bahasa.*
   > *5) Tambahkan 156,157 dokumen dari situs pelatihan saja (paling banyak 5,000 per bahasa dan 400 per situs)."*
   - **Masalah**: Ini adalah riwayat skrip dua tahap selama pengembangan (awalnya membuat dataset 130 ribu dokumen dengan batas 150/situs, lalu di tahap berikutnya menambah 156 ribu dokumen dan menaikkan batas menjadi 400/situs). Bagi pembaca paper, sangat membingungkan mengapa langkah 3 membatasi maksimal 150 dokumen/situs "untuk menghindari bias", lalu dua baris kemudian langkah 5 menabraknya menjadi 400 dokumen/situs.
   - **Saran**: Tuliskan spesifikasi akhir penyusunan *split held-out* dan korpus pelatihan (batas 400 dokumen/situs dan 5,000/bahasa pada situs pelatihan) secara langsung, lalu cukup sebutkan versi subset 130 ribu dokumen ketika menjelaskan varian ablasi di Bagian V-D.

3. **Catatan Pembukuan Jumlah *Run* & Eksperimen Sisa Pengembangan (Hal. 5, 8–9, Bagian III-D, V-B, V-D — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L184), [L323](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L323), [L341](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L341))**
   > - *"Nilai $\tau = 1$ dipilih selama pengembangan dan hasil untuk nilai lain $\tau \in \{0.5, 2, 4\}$ dilaporkan pada Bagian VI-D."* (Hal. 5)
   > - *"Koreksi Holm untuk perbandingan ini mencakup 10 varian pada Bagian V-D dan empat tambahan dengan masing-masing tiga seed pada Tabel S6 materi suplemen."* (Hal. 8–9)
   > - *"Tabel 7 memuat seluruh 60 run resep akhir dan ablasinya. Empat tambahan pada resep yang diuji selama pengembangan dilaporkan dalam Tabel S6 materi suplemen."* (Hal. 9)
   - **Masalah**: Terbaca seperti buku log eksperimen (*experiment tracker ledger*: menghitung total `60 run` dan melaporkan sisa `empat tambahan dengan masing-masing tiga seed` yang dicoba sewaktu *development*).
   - **Saran**: Cukup jelaskan faktor ablasi yang diuji secara ilmiah di naskah utama; detail varian eksploratif di suplemen cukup dirujuk dalam satu kalimat ringkas tanpa gaya pencatatan log *run*.

4. **Catatan Rekonsiliasi Angka di Judul Utama Tabel 3 (Hal. 7 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L275))**
   > *"TABEL 3. Komposisi Test Set Terpadu menurut Sumber. Satu Situs (jpost.com) Muncul di CC-NEWS dan di MN-DS"*
   - **Masalah**: Kalimat kedua ditaruh langsung di **judul tabel** semata-mata sebagai catatan *sanity-check* mengapa penjumlahan kolom terakhir (`1,092 + 6 + 1 + 127 + 1 + 1 = 1,228`) berbeda satu angka dari baris Total (`1,227`).
   - **Saran**: Pindahkan keterangan `jpost.com` menjadi catatan kaki kecil (*table footnote*) di bawah tabel, bukan di judul tabel.

5. **Catatan *Debug Log* pada Daftar Filter (Hal. 7, Bagian IV-C — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L260))**
   > *"Filter ini membuang 63 artikel. Sebagian besar artikel yang dibuang berasal dari MasakhaNEWS yang juga bersumber dari BBC."*
   - **Masalah**: Butir 1–5 hanya mendeskripsikan aturan penyaringan, tetapi khusus butir 6 diselipi catatan hasil inspeksi log pipa data.

6. **Angka Internal Skrip Analisis Tanpa Konteks Pembaca (Hal. 8–9, Bagian V-B — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L319), [L325](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L325))**
   > - *"...menghitung ulang macro-F1 sembilan model dengan membuang tiap sumber..."* (Pembaca belum tahu "sembilan model" mana saja, karena Tabel 4 memiliki 13 baris).
   > - *"Nilai $p$ dari ketujuh uji ini dikoreksi Holm bersama-sama."* (Kalimat-kalimat sebelumnya tidak pernah merinci daftar tujuh uji; angka "ketujuh" merujuk pada baris Tabel S4 di skrip penulis).

7. **Catatan Retrospektif Pemilihan Resep (Hal. 15, Bagian VII-B — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L624))**
   > *"Resep akhir tetap memakai fine-tuning karena resep ini dipilih pada development set dalam bahasa pelatihan. Development set semacam itu tidak dapat mengungkap efek ini."*
   - **Masalah**: Bernada catatan *post-mortem* mengapa penulis terlanjur memilih *fine-tuning*. Sebaiknya ditulis sebagai analisis ilmiah mengenai *trade-off* antara adaptasi *in-distribution* dan pelestarian penyelarasan lintas bahasa (*cross-lingual alignment preservation*).

---

### 1.C. Meta-Definisi & Catatan Glosarium Kaku (*"disebut X dalam makalah ini"*)

Naskah berulang kali menempelkan kalimat pendefinisian istilah bergaya catatan internal di tengah atau akhir paragraf:

1. **Hal. 2, Bagian I ([main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L62))** *(Contoh yang Anda temukan)*:
   > *"Penerapan head linear pada setiap embedding token sekali sebelum model dipakai disebut folding dalam makalah ini."*
   - Muncul tiba-tiba di ujung paragraf tinjauan literatur. Seharusnya konsep pelipatan bobot linear (*weight folding*) diperkenalkan secara alami saat memaparkan ide utama metode atau langsung merujuk istilah pada literatur terkait [17].
2. **Hal. 3, Bagian II-B ([main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L92))**:
   > *"Logit adalah skor mentah yang dikeluarkan model untuk setiap kelas sebelum dikonversi menjadi probabilitas... Distribusi lengkap ini disebut soft label dan kelas teratasnya saja disebut hard label."*
   - Catatan glosarium dasar buku teks yang memutus alur tinjauan pustaka *Knowledge Distillation*. Pembaca jurnal ilmiah sudah memahami definisi dasar logit.
3. **Hal. 4, Bagian III-D ([main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L171))**:
   > *"...potion-multilingual-128M dari Model2Vec [16] (selanjutnya disebut potion)."*
   - Terkesan seperti penamaan variabel *shorthand* di kode/catatan. Lebih bersih menggunakan istilah deskriptif (*embedding Model2Vec* atau *embedding prapelatihan*) setelah penyebutan spesifikasi model di awal subbab.
4. **Hal. 6, Bagian IV-B ([main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L243))**:
   > *"Agreement adalah proporsi dokumen yang label prediksi teratasnya sama dengan label teratas teacher."*
   - Ditempelkan di ujung Bagian IV-B, padahal istilah *agreement* sudah dipakai berkali-kali sejak halaman 2, 3, 5, dan Tabel 2.
5. **Hal. 6, Bagian IV-C ([main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L249))**:
   > *"Kategori penerbit tersebut dan label rubrik sama-sama mencatat tempat redaksi menaruh artikel saat artikel itu terbit. Label semacam ini disebut label editorial dalam makalah ini."*
   - Pola yang sama persis dengan contoh *folding*, muncul terlambat setelah istilah *label editorial* dipakai di Bagian II-E, Gambar 1, dan Tabel 2. Selain itu, definisi ini **bertabrakan dengan Tabel 2** (baris `Test set terpadu` menulis kolom Label = `editorial` untuk seluruh 12,744 dokumen, padahal 32.1% di antaranya adalah anotasi manusia).
6. **Hal. 7, Bagian IV-C ([main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L263))**:
   > *"Jenis label suatu artikel (anotasi manusia, kategori penerbit, atau rubrik JSON-LD) disebut asal label artikel tersebut."*
   - Pola meta-definisi yang sama dan memutus paragraf statistik komposisi *test set*.
7. **Hal. 9, Bagian V-B ([main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L325), [L327](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L327))**:
   > - *"Cakupan prapelatihan menyatakan apakah suatu bahasa termasuk dalam 100 bahasa yang dipakai untuk prapelatihan XLM-R [18]."* (Terselip di tengah daftar uji statistik).
   > - *"Selisih akurasi adalah akurasi StaticKD dikurangi akurasi teacher. Selisih negatif berarti StaticKD kurang akurat."* (Penjelasan pengurangan aritmetika dasar ala komentar *notebook*).

---

### 1.D. Kebocoran Istilah Skrip, Kode Mentah, dan Catatan AI

- **Terjemahan harfiah variabel lingkungan skrip (Hal. 9, Bagian V-C — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L331))**:
  > *"Benchmark dijalankan hanya pada CPU dengan GPU disembunyikan."*
  - Frasa *"dengan GPU disembunyikan"* adalah terjemahan harfiah dari trik *command-line* `CUDA_VISIBLE_DEVICES=""`. Dalam makalah ilmiah cukup tulis *"dijalankan sepenuhnya pada CPU tanpa akselerasi GPU"*.
- **Rujukan langsung ke skrip alih-alih definisi metrik (Hal. 8, Bagian V-B — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L317))**:
  > *"Metrik ini mencakup label yang ada dalam referensi dan dihitung seperti pada kode evaluasi dari [6]."*
  - Sebaiknya tulis definisi matematisnya secara mandiri (rata-rata skor F1 tak tertimbang atas seluruh kelas yang memiliki sampel pada label referensi, mengikuti protokol evaluasi [6]).
- **Kode bahasa ISO mentah dari *DataFrame* (Hal. 6, 7, 13 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L238), [L272](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L272), [L575-L577](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L575-L577))**:
  - Menulis `ja, zh, th, dan my`, `ha, yo, ig, bn, dan pa`, `(ig, lg, ln, pcm, rn, sn, ti, yo)`, serta catatan tambalan `(bn, ml, or [Odia], pa)` langsung di dalam kalimat prosa. Gunakan nama bahasa lengkap dalam teks naskah (mis. Jepang, Mandarin, Thai, dan Burma; Igbo, Luganda, Lingala, Pidgin Nigeria, Kirundi, Shona, Tigrinya, dan Yoruba).
- **Penyebutan versi model `Claude Opus 5.5` (Hal. 6 & Hal. 17 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L251), [L674](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L674))**:
  - Periksa kembali penulisan nama/versi model `Claude Opus 5.5` (apakah salah ketik dari versi Claude Opus 4.5 / 4.6) serta pastikan paragraf deklarasi penggunaan AI pada *Ucapan Terima Kasih* memang sesuai dengan format kebijakan penerbit yang dituju dan bukan catatan internal penyusunan draf.

---

## Bagian 2: Temuan Paragraf Tidak Sinambung & Kalimat Tidak Saling Terkait

Berikut adalah daftar lengkap paragraf dan rangkaian kalimat yang alurnya terputus, melompat antar-topik, atau tidak jelas tujuan argumennya dari awal hingga akhir naskah.

### Abstrak (Hal. 1 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L42-L44))
- **Kalimat 6–8**: *"Evaluasi memakai test set baru berisi 12,744 artikel dalam 47 bahasa. Labelnya berasal dari anotator manusia atau editor redaksi dan bukan dari model yang membaca artikel. Perbedaan antarmodel diuji dengan site-cluster bootstrap..."*
  - **Masalah**: Klausa *"dan bukan dari model yang membaca artikel"* terdengar janggal dan defensif tanpa konteks di abstrak (maksudnya: label *ground truth* bebas dari anotasi otomatis LLM), lalu langsung melompat ke metode uji statistik.
- **Kalimat 12–15**: *"Pengklasifikasi Model2Vec yang dilatih pada label ini sama akuratnya dalam batas 0.01 macro-F1 (...). Pengklasifikasi tersebut memerlukan berkas 15 kali lebih besar dan memori dua kali lipat. StaticKD lemah pada bahasa yang tidak terdapat baik dalam data pelatihannya maupun dalam data prapelatihan teacher."*
  - **Masalah**: Antara kalimat 12 dan 13 tidak ada penanda pertentangan (*namun* / *tetapi*), lalu kalimat terakhir langsung melompat dari ukuran memori Model2Vec ke kelemahan bahasa StaticKD tanpa transisi penutup.

---

### Bab I. Pendahuluan (Hal. 1–2)

1. **Paragraf 1 (Hal. 1 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L56))**
   - **Masalah**: Perpindahan dari masalah skema lokal/terfragmentasi (kalimat 2) ke IPTC (kalimat 3) tidak dijembatani, dan kalimat penutup paragraf pembuka justru menggantung pada rujukan silang administratif (*"...yang dirinci pada Bagian II-A"*).
2. **Paragraf 2 (Hal. 1–2 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L58))** *(Contoh yang Anda temukan)*
   > *"...Multi-head self-attention di dalam model tersebut memiliki kompleksitas komputasi dan memori $O(N^2)$ terhadap panjang input $N$ [9]. Pengklasifikasi transformer dengan ratusan juta parameter juga menempati memori berukuran gigabyte. Setiap lapisannya menjalankan perkalian matriks padat pada sedikit core CPU yang dipakai bersama dengan tahap-tahap lain dari pipeline berita."*
   - **Masalah**:
     - Kalimat 1 dan kalimat 3 mengulang ide yang sama ("mendominasi... tetapi biaya komputasinya tinggi" lalu diulang "akurat tetapi menghadapi hambatan komputasi").
     - Tiga kalimat terakhir menumpuk tiga fakta teknis terpisah ($O(N^2)$, memori gigabyte, perkalian matriks pada sedikit *core* CPU bersama tahap lain *pipeline*) tanpa menyimpulkan **akibat praktisnya** (bahwa latensi per artikel membengkak hingga skala detik sehingga memacetkan pemrosesan berita *real-time*).
3. **Paragraf 3 (Hal. 2 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L60))**
   - **Masalah**: Kalimat 2 memperkenalkan MiniLM dan DistilBERT, lalu kalimat 3 langsung menulis *"Versi multibahasa dari transformer ringkas ini tetap memiliki..."* tanpa kata hubung kontras (*Namun,*), dan kalimat 4 berpindah ke jalur kedua (*fastText*) tanpa penanda transisi (*Sementara itu,* / *Di sisi lain,*).
4. **Paragraf 4 (Hal. 2 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L62))** *(Paragraf paling melompat-lompat di Bab I)*
   > *"Klasifikasi topik berita cocok untuk representasi statis karena model bag-of-words tetap kompetitif... [13]. Sparse Distillation [14] menunjukkan... Embedding token statis juga dapat diperoleh dengan mengompresi sentence transformer... [15], [16]. Embedding Model2Vec multibahasa selaras lintas 101 bahasa. Singh dkk. [17] mengonversi model klasifikasi linear atas embedding n-gram menjadi kamus lookup koefisien... Penerapan head linear pada setiap embedding token sekali sebelum model dipakai disebut folding dalam makalah ini."*
   - **Masalah**: Lima gagasan berbeda dijejalkan ke dalam enam kalimat pendek tanpa benang merah argumen yang mengikatnya menjadi satu kesatuan pikiran. Kalimat *"Embedding Model2Vec multibahasa selaras lintas 101 bahasa."* berdiri terisolasi di tengah, diikuti lompatan mendadak ke Singh dkk. [17].
5. **Daftar Celah Penelitian, Butir 1 (Hal. 2 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L66))**
   - **Masalah**: Judul butir membahas *pengklasifikasi IPTC multibahasa*, kalimat pertama membahas XLM-R Kuzman & Ljubešić, lalu kalimat kedua tiba-tiba membahas *Sparse Distillation* (yang bukan pengklasifikasi IPTC) tanpa kalimat penghubung mengapa metode tersebut relevan di poin ini.
6. **Daftar Kontribusi, Butir 2 (Hal. 2 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L73))**
   - **Masalah**: Kalimat *"Setiap label dipetakan ke taksonomi IPTC tanpa model yang membaca teks artikel."* membingungkan pembaca saat nanti membaca Bagian IV-C bahwa pemetaan rubrik memakai LLM; maksud sebenarnya adalah *pemetaan dilakukan pada tingkat nama rubrik (metadata), bukan dengan mengklasifikasikan isi artikel menggunakan model*.

---

### Bab II. Penelitian Terkait (Hal. 2–4)

1. **Bagian II-A, Paragraf 1 & 2 (Hal. 2 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L82-L84))**
   - **Masalah**:
     - Kalimat terakhir Paragraf 1 (*"Dataset berita publik berlabel topik biasanya mendefinisikan taksonominya sendiri..."*) tidak nyambung dengan dua kalimat awalnya yang mendeskripsikan struktur IPTC; kalimat tersebut seharusnya menjadi pembuka Paragraf 2.
     - Paragraf 2 justru dibuka seperti bab metodologi (*"Studi ini memakai lima dataset berita publik berlabel topik dalam test set (Bagian IV-C)..."*), bukan tinjauan pustaka atas lanskap *dataset* berita.
2. **Bagian II-A, Paragraf 4 (Hal. 3 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L88))**
   > *"Model XLM-R-large hasil fine-tuning Kuzman dan Ljubešić [6] menjadi teacher dalam studi ini (Bagian III-C). Teacher ini menumpuk 24 lapisan transformer dengan multi-head self-attention dan menjalani fine-tuning hanya pada empat bahasa. Akurasinya pada bahasa lain bergantung pada transfer lintas bahasa dari prapelatihan XLM-R. Sejauh ini belum ada pengklasifikasi IPTC berbiaya rendah yang dievaluasi pada berita nyata dalam banyak bahasa."*
   - **Masalah**:
     - Kalimat 2 menggabungkan dua hal yang tidak sejenis menggunakan `"dan"`: kedalaman komputasi (`24 lapisan transformer`) dan keterbatasan data pelatihan (`hanya pada empat bahasa`).
     - Kalimat 3 melanjutkan soal cakupan bahasa, tetapi kalimat 4 tiba-tiba menyimpulkan soal **biaya rendah** (*"belum ada pengklasifikasi IPTC berbiaya rendah..."*), yang tidak didukung oleh kalimat sebelumnya.
3. **Bagian II-B, Paragraf 1 & 4 (Hal. 3 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L92), [L103](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L103))**
   - **Paragraf 1**: Diakhiri dengan kalimat tentang *curriculum temperature* yang dapat dipelajari [32], padahal penelitian ini menggunakan $\tau$ konstan dan tidak menindaklanjuti ide tersebut.
   - **Paragraf 4**: Mencampur dua topik literatur berbeda dalam satu paragraf secara selang-seling: kesenjangan kapasitas (*capacity gap*) di kalimat 1–3, melompat ke masalah *agreement* vs generalisasi [37] di kalimat 4, lalu melompat balik ke kesenjangan parameter di kalimat 5.
4. **Bagian II-C, Paragraf 1 (Hal. 3 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L107))**
   - **Masalah**: Satu kalimat tentang Wada dkk. [15] terselip sendirian di tengah pembahasan ruang vektor `potion-multilingual-128M` tanpa dijelaskan relevansi atau perbedaannya terhadap Model2Vec.
5. **Bagian II-D, Paragraf 1 (Hal. 3–4 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L113))**
   - **Masalah**: Alurnya terbalik untuk subbab *Penelitian Terkait*: tiga kalimat pertama memaparkan cara kerja metode penulis sendiri (merujuk ke Bagian III-E), baru di kalimat keempat meninjau karya terdahulu Singh dkk. [17].
6. **Bagian II-E, Paragraf 1 (Hal. 4 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L116))**
   - **Masalah**: Menggabungkan paksa **tiga topik yang sama sekali tidak berkaitan** ke dalam satu paragraf 7 kalimat: (1) metadata `articleSection` schema.org, (2) deduplikasi dokumen dengan MinHash, dan (3) uji statistik *cluster bootstrap*. Paragraf ini terasa seperti daftar referensi untuk membenarkan langkah teknis di Bab IV dan V.

---

### Bab III. Metode StaticKD (Hal. 4–5)

1. **Bagian III-B, Formulasi Masalah (Hal. 4 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L163))**
   - **Masalah**: Menyebutkan input *"dipotong hingga 512 kata pertama"* seolah berlaku umum bagi masalah ini, padahal pemotongan *student* di Bagian III-D adalah 1,024 token (bukan 512 kata). Selain itu, batasan rekayasa perangkat lunak (*"tidak memerlukan... pustaka deep learning"*) dicampur ke dalam formulasi matematis masalah.
2. **Bagian III-C, Teacher dan Label Distilasi (Hal. 4 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L167))**
   > *"...Pengklasifikasi ini adalah model XLM-RoBERTa-large dengan 24 lapisan, 559.9 juta parameter, dan batas input 512 token XLM-R. Input 512 kata melampaui batas ini pada 60.7% artikel test set sehingga teacher membaca sekitar 287 dari 438 kata pada artikel semacam itu (median). Teacher melabeli seluruh data distilasi dengan biaya satu kali sekitar 1.6 jam GPU..."*
   - **Masalah**: Subbab ini menjelaskan pelabelan **data distilasi**, tetapi kalimat ketiga tiba-tiba melompat membahas statistik pemotongan kata pada **test set** (`60.7% artikel test set`), lalu kalimat keempat melompat kembali ke **data distilasi**.
3. **Bagian III-D, Paragraf 1 & 2 (Hal. 4–5 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L171-L184))**
   - **Paragraf 1**: Kalimat *"Panjang dokumen dipotong pada 1,024 token hanya untuk membatasi biaya karena perataan tidak bergantung pada posisi"* memiliki hubungan sebab-akibat (`karena`) yang tidak logis (ketidakbergantungan pada posisi bukan alasan membatasi biaya), lalu tiba-tiba diselipi statistik `Hanya 5.4% artikel test set yang melampaui batas ini` sebelum persamaan dasar *student* ditulis.
   - **Paragraf 2**: Memperkenalkan mekanisme kedua berupa *token dropout* (0.2) di tengah penjelasan *optimizer* dan jumlah *epoch*, terpisah jauh dari `drop(h(d))` (0.1) di Persamaan (2), sehingga pembaca mudah rancu antara kedua *dropout* tersebut.
4. **Bagian III-E, Paragraf 2 (Hal. 5 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L202))**
   > *"Model StaticKD akhir merata-ratakan tabel dari tiga student ($K = 3$) dan memiliki 8.5 juta parameter dengan ukuran 34 MB. Tabel hasil perataan menyumbang 16 MB dari ukuran tersebut. Struktur internal Unigram tokenizer memakai sebagian besar random-access memory (RAM) model ini. Varian ringkas memperkecil struktur tersebut... dan ukuran modelnya 23 MB."*
   - **Masalah**: Paragraf ini melompat dari **ukuran berkas di disk** (34 MB dan 16 MB) ke **pemakaian RAM saat runtime** di kalimat 3, lalu kembali ke **ukuran berkas di disk** (23 MB) di kalimat 5 tanpa membedakan keduanya secara eksplisit (padahal di Tabel 6 ukuran RAM-nya adalah 501 MB dan 277 MB).

---

### Bab IV. Data dan Test Set (Hal. 5–7)

1. **Bagian IV-C, Paragraf 1 & 3 (Hal. 6 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L247), [L251](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L251))**
   - **Paragraf 1**: Membahas cara ekstraksi teks JSON-LD dan HTML, tetapi kalimat terakhirnya tiba-tiba menyisipkan aturan filter kebocoran situs (*"Filter situs membuang artikel..."*) terpisah dari daftar filter utama di bawahnya, dengan istilah kabur *"dalam set evaluasi"*.
   - **Paragraf 3**: Seluruh paragraf membahas penerjemahan **tabel nama rubrik tunggal** ke *node* IPTC, tetapi kalimat penutupnya (*"Artikel yang rubrik-rubriknya terpetakan ke label berbeda ditolak."*) tiba-tiba berpindah ke kasus artikel yang memiliki banyak rubrik tanpa penjelasan awal bahwa satu artikel dapat memuat lebih dari satu rubrik.
2. **Bagian IV-C, Paragraf 4 (Hal. 7 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L263))**
   - **Masalah**: Alur kronologisnya terbalik: kalimat pertama sudah menyajikan angka akhir (`12,744 artikel dalam 47 bahasa`), lalu kalimat kedua dan ketiga **mundur** menjelaskan aturan penyampelan kuota 50 artikel dan pembuangan bahasa di bawah 30 artikel yang menghasilkan angka tersebut. Kalimat 3 juga menggabungkan dua aturan berbeda dengan kata `"dan"`.
3. **Bagian IV-C, Paragraf 5 (Hal. 7 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L272))**
   - **Masalah**: Menggabungkan dua karakteristik dari dua himpunan data berbeda ke dalam satu paragraf: ketersediaan **data distilasi** (pelatihan) per bahasa dan konsentrasi **situs efektif pada test set** (evaluasi).
4. **Bagian IV-D, Paragraf 1 (Hal. 7 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L299))**
   - **Masalah**: Kalimat terakhir (*"Analisis per kategori sumber juga memeriksa pemetaan..."*) ditempelkan di ujung paragraf yang sebelumnya khusus membahas perbandingan label rubrik JSON-LD vs label manusia pada bahasa yang sama.

---

### Bab V. Pengaturan Eksperimen (Hal. 7–9)

1. **Bagian V-A, Paragraf 1 & 2 (Hal. 7–8 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L307-L309))**
   - **Masalah**: Alur perkenalan *baseline* ringan melompat-lompat: menyebut tiga *baseline* + KD -> membahas *fastText + KD* (dengan frasa rancu *"memakai token kata dan n-gram karakter atau subword XLM-R"*, padahal itu dua varian berbeda di Tabel 4) -> melompat ke *fastText* tanpa KD (label GPT-4o) -> di Paragraf 2 membahas *TF-IDF + SVM* -> melompat balik ke *grid search* untuk *fastText + KD* -> baru membahas *Model2Vec + KD*.
2. **Bagian V-B, Paragraf 2, 5, dan 6 (Hal. 8–9 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L319), [L325-L327](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L325-L327))**
   - **Paragraf 2**: Kalimat pendek *"Development set di-resample per dokumen."* terselip sendirian di tengah pembahasan *site-cluster bootstrap test set*, lalu paragraf beralih dari metode penghitungan interval kepercayaan menjadi pemaparan statistik ketimpangan sumber (`bbc.com`, `13.8 klaster`) dan daftar uji sensitivitas *leave-one-source-out*.
   - **Paragraf 5**: Tujuh kalimat pendek patah-patah yang mengabsen variabel pembagian subkelompok dan nama uji statistik tanpa alur argumen.
   - **Paragraf 6**: Menggabungkan definisi metrik untuk analisis per bahasa (*selisih akurasi*, *cakupan token*) dengan eksperimen interpretabilitas *pemangkasan 10% token paling spesifik* yang sama sekali berbeda tujuan.
3. **Bagian V-C, Paragraf 2, 3, dan 4 (Hal. 9 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L333-L337))**
   - **Paragraf 2**: Diakhiri dengan kalimat lepas tentang eksperimen *profiling* internal (*"Profil satu thread terpisah untuk StaticKD mengukur waktu tokenisasi... pada 1,280 artikel"*) yang terpisah dari protokol perbandingan antarmodel.
   - **Paragraf 3**: Membahas format ekspor ONNX fp32 dan kuantisasi int8, lalu dua kalimat terakhir tiba-tiba melompat membahas batas panjang token (512 vs 1,024).
   - **Paragraf 4**: Tiba-tiba memperkenalkan **mode *cascade*** di subbab *Protokol Efisiensi* perangkat keras, padahal *cascade* adalah mekanisme inferensi dua tahap yang mengubah baik akurasi maupun *throughput* dan belum pernah diperkenalkan di Bab III (Metode).

---

### Bab VI. Hasil (Hal. 9–14)

1. **Bagian VI-A, Paragraf 1 (Hal. 9 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L351))**
   - **Masalah**: Kalimat topik paragraf secara spesifik membatasi klaim pada **bahasa Inggris** (*"...menunjukkan bahwa label rubrik bahasa Inggris tidak mengandung noise lebih banyak..."*), tetapi kalimat terakhir tiba-tiba menempelkan hasil **bahasa Prancis**.
2. **Bagian VI-B, Paragraf 1 & 6 (Hal. 10 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L389), [L429](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L429))**
   - **Paragraf 1**: Empat kalimat awal membandingkan StaticKD dengan *teacher*, lalu dua kalimat terakhir tiba-tiba beralih ke *fastText* berlabel GPT-4o tanpa kalimat penghubung (mis. untuk menunjukkan pentingnya pelabelan korpus multibahasa oleh *teacher*).
   - **Paragraf 6**: Menggabungkan dua pemeriksaan sensitivitas yang tidak berkaitan dalam satu paragraf: struktur klaster situs pada *test set* dan pelatihan ulang *transformer* dengan panjang input 512 token.
3. **Bagian VI-C, Paragraf 2 (Hal. 11 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L464))**
   - **Masalah**: Mencampur tiga hal berbeda dalam satu paragraf: perbandingan kecepatan antarmodel, satu kalimat sisipan tentang simpangan pengukuran antar-sesi (`menyimpang paling banyak 7%`), dan hasil *profiling* internal waktu tokenisasi vs *lookup* tabel (98% vs 26,843 artikel/detik).
4. **Bagian VI-D, Paragraf 3 (Hal. 12 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L527))**
   - **Masalah**: Tiga kalimat terakhir bolak-balik merujuk ke varian yang berbeda (*dropout* + *head MLP* digabung dengan `"dan"`, lalu kalimat berikutnya kembali ke *varian data* + *varian dropout*, lalu kalimat terakhir kembali lagi ke *head MLP*).
5. **Bagian VI-E, Paragraf 3 & 7 (Hal. 13–14 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L573), [L588](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L588))**
   - **Paragraf 3**: Mengumpulkan berbagai hasil korelasi dari *test set* dan *held-out set* serta uji Kruskal–Wallis aksara tanpa satu gagasan utama yang menyatukan paragraf.
   - **Paragraf 7**: Membahas eksperimen menyisakan 10% token paling spesifik di bawah subbab berjudul *Analisis per Bahasa, Label, dan Sumber* — topiknya tidak sesuai dengan subbabnya.
6. **Bagian VI-F, Paragraf 1 (Hal. 14 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L612))**
   > *"Tingkat keyakinan StaticKD lebih terkalibrasi daripada tingkat keyakinan teacher dan menentukan artikel yang diputuskan StaticKD dalam mode cascade."*
   - **Masalah**: Menggabungkan temuan empiris (*lebih terkalibrasi*) dan aturan mekanisme sistem (*menentukan artikel dalam mode cascade*) dengan kata `"dan"`, padahal hubungan sebenarnya adalah sebab-akibat (*karena lebih terkalibrasi, probabilitas StaticKD layak dipakai sebagai ambang penyaring pada mode cascade*).

---

### Bab VII. Pembahasan & Bab VIII. Kesimpulan (Hal. 14–17)

1. **Bagian VII-C, Bahasa yang Lemah dan Pemilihan Model (Hal. 15 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L632-L635))**
   - **Masalah**: Subbab ini menyatukan dua paragraf dengan topik yang tidak saling berhubungan: Paragraf 1 membahas kelemahan pada bahasa Afrika tanpa data, sedangkan Paragraf 2 membahas ketidaksesuaian antara metrik *agreement* terhadap *teacher* dan akurasi independen saat memilih model. (Paragraf 2 jauh lebih padu jika ditempatkan bersama pembahasan bias *development set* di Bagian VII-B).
2. **Bagian VII-D, Implikasi Praktis (Hal. 16 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L638))**
   - **Masalah**: Di tengah-tengah paragraf yang membahas penerapan praktis produksi (*deployment* StaticKD vs Model2Vec, mode *cascade*, dan varian ringkas), tiba-tiba terselip dua kalimat perbandingan arsitektur teoritis terhadap *Sparse Distillation* [14] yang memutus alur pembahasan praktis.
3. **Bagian VII-E, Keterbatasan — Paragraf 3, 4, 6, dan 8 (Hal. 16 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L646-L656))**
   - **Paragraf 3 ([L646](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L646))**: Empat kalimat pertama berderet memaparkan fakta latar belakang (MN-DS, *test set* Kuzman tidak publik, MasakhaNEWS, cara pemetaan rubrik) tanpa kalimat topik yang jelas, sebelum akhirnya tiba pada poin keterbatasan di kalimat terakhir (ketiadaan ukuran *inter-annotator agreement* pada verifikasi manual).
   - **Paragraf 4 ([L648](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L648))**: Mencampur keterbatasan jumlah topik per bahasa (yang memengaruhi perbandingan macro-F1 antarbahasa) dengan keterbatasan jumlah klaster efektif sumber (yang memengaruhi lebar interval kepercayaan *bootstrap*).
   - **Paragraf 6 ([L652](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L652))**: *"Panjang input model transformer dan mesin benchmark juga membatasi pengukuran."* — Menggabungkan dua hal yang sama sekali tidak berhubungan (pemotongan 256 token saat melatih *baseline transformer* untuk akurasi vs pengukuran latensi CPU pada satu mesin laptop).
   - **Paragraf 8 ([L656](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L656))**: Menggabungkan keterbatasan plafon kinerja/bias warisan *teacher* dengan keterbatasan cakupan tugas (IPTC tingkat atas dan *single-label*) di bawah kalimat pembuka generik *"Batas lain berasal dari rancangan studi ini."*
4. **Bab VIII, Kesimpulan — Paragraf 3 (Hal. 17 — [main_id.tex](file:///d:/study/s2/kuliah/semester_2/metpen/iptc_label_class/paper_final_id/main_id.tex#L664))**
   > *"Ablasi menunjukkan bahwa akurasi student kecil ini terutama berasal dari data distilasi multibahasa dan embedding multibahasa hasil prapelatihan. Soft label tidak lebih baik daripada hard label. Embedding beku menurunkan skor development set dan memberikan rata-rata skor yang lebih tinggi pada bahasa dengan data distilasi yang sedikit..."*
   - **Masalah**: Kalimat *"Soft label tidak lebih baik daripada hard label."* berdiri terisolasi tanpa jembatan ke kalimat sebelum atau sesudahnya, serta kalimat ketiga memakai kata sambung `"dan"` untuk dua efek yang berlawanan arah (seharusnya *tetapi* / *namun*).
