# Paper Rewrite (StaticKD)

Lembar kerja untuk menulis ulang naskah paragraf demi paragraf. Untuk setiap blok:

1. Baca **Current** (teks paragraf sekarang) dan **Makna** (artinya dalam bahasa Indonesia).
2. Tulis versi Anda di bawah **Paraphrase:** (ganti tulisan "(tulis di sini)"). Cukup teks biasa; sitasi ditulis seperti di Current, misalnya [2], dan rujukan seperti "Section IV-D" atau "Table IV".
3. Pastikan semua yang ada di **Locked** muncul lagi di tulisan Anda: angka sama persis, nomor sitasi tetap menempel pada klaim yang sama.

Aturan singkat: tanpa *we/our* (gaya impersonal); istilah teknis tetap bahasa Inggris; definisikan singkatan saat pertama muncul; hindari pola di [ai_text_criteria_to_be_avoided.txt](ai_text_criteria_to_be_avoided.txt). Tabel, gambar, dan persamaan hanya placeholder.

Membangun ulang file ini (`python scripts/rewrite_lock.py && python scripts/build_rewrite_md.py`) tidak menghapus Paraphrase yang sudah diisi.

## Judul, penulis, abstrak, kata kunci

---

### JDL · Title · main.tex baris 28

**Current:**
StaticKD: Distilling a Multilingual IPTC News Topic Classifier into a Static Token-Logit Table for Lightweight CPU Inference

**Makna:**
StaticKD: men-distill classifier topik berita IPTC multibahasa menjadi satu tabel logit per token (static token-logit table) untuk inference ringan di CPU.

**Locked:** —

**Paraphrase:**
(tulis di sini)

---

### PNL · Authors · main.tex baris 30

**Current:**
Obi Kastanya and Ananta Dwi Prayoga Alwy (O. Kastanya and A. D. P. Alwy are with the Department of Informatics Engineering, Sepuluh Nopember Institute of Technology, Surabaya, Indonesia (e-mail: 6025252015@student.its.ac.id; 6025252007@student.its.ac.id). Corresponding author: O. Kastanya. This work received no specific funding.

**Makna:**
Penulis: Obi Kastanya dan Ananta Dwi Prayoga Alwy. Catatan kaki afiliasi: keduanya dari Department of Informatics Engineering, Institut Teknologi Sepuluh Nopember (ITS), Surabaya, Indonesia; e-mail 6025252015@student.its.ac.id dan 6025252007@student.its.ac.id; penulis korespondensi O. Kastanya; penelitian ini tidak menerima pendanaan khusus.

**Locked:** angka 6025252015, 6025252007

**Paraphrase:**
(tulis di sini)

---

### ABS · Abstract · main.tex baris 38

**Current:**
Filtering news by the International Press Telecommunications Council (IPTC) Media Topic taxonomy needs a classifier that is cheap to run on a multilingual stream of articles. The open multilingual IPTC classifier, based on XLM-RoBERTa-large, takes 1,239 to 2,971 milliseconds per article on one processor core. It is distilled into StaticKD, a student built on multilingual static embeddings whose linear head is folded into the embedding table, so the whole model is one table of 17 logits per token. The teacher labels 285,999 unlabeled articles in 69 languages. A test set of 12,744 articles in 47 languages, labeled by human annotators or by the sections that newsroom editors assigned and mapped onto the IPTC tree, allows evaluation without labels from any model. On this test set StaticKD reaches a macro-F1 of 0.545, or 91.1% of the teacher score of 0.598. It classifies 473 articles per second per core, about 616 times faster than the 8-bit quantized teacher, with a 34-megabyte model, and it is more accurate than fastText and a linear support vector machine distilled on the same data. Ablations with five seeds per variant show that accuracy comes from multilingual distillation data and cross-lingually aligned embeddings, while hard labels match soft labels. Transfer to languages absent from the distillation data is carried by embedding rows that training never updates, and fine-tuning the embeddings trades this transfer for in-distribution fit. StaticKD is weakest on languages missing from both the distillation data and the pre-training data of the teacher.

**Makna:**
Masalah: menyaring berita dengan taksonomi IPTC Media Topic butuh classifier yang murah dijalankan pada aliran artikel multibahasa. Classifier IPTC multibahasa terbuka yang ada, berbasis XLM-RoBERTa-large, butuh 1,239 sampai 2,971 milidetik per artikel pada satu core prosesor. Model itu di-distill menjadi StaticKD: student berbasis static embedding multibahasa yang kepala linearnya dilipat ke dalam tabel embedding, sehingga seluruh model hanyalah satu tabel berisi 17 logit per token. Teacher melabeli 285,999 artikel tak berlabel dalam 69 bahasa. Test set berisi 12,744 artikel dalam 47 bahasa, berlabel dari anotator manusia atau dari rubrik (section) yang ditetapkan redaksi lalu dipetakan ke pohon IPTC; jadi evaluasi tidak memakai label dari model apa pun. Di test set ini StaticKD mencapai macro-F1 0.545, atau 91.1% dari skor teacher 0.598. StaticKD mengklasifikasi 473 artikel per detik per core, sekitar 616 kali lebih cepat daripada teacher yang dikuantisasi 8-bit, dengan model 34 megabyte, dan lebih akurat daripada fastText serta linear support vector machine yang di-distill dengan data yang sama. Ablasi dengan lima seed per varian menunjukkan akurasi berasal dari data distilasi multibahasa dan embedding yang selaras lintas bahasa, sedangkan hard label setara dengan soft label. Transfer ke bahasa yang tidak ada di data distilasi dibawa oleh baris embedding yang tidak pernah di-update saat training, dan fine-tuning embedding menukar transfer ini dengan kecocokan in-distribution. StaticKD paling lemah pada bahasa yang tidak ada di data distilasi maupun di data pre-training teacher.

**Locked:** angka 1,239, 2,971, 17, 285,999, 69, 12,744, 47, 0.545, 91.1%, 0.598, 473, 616, 8, 34

**Paraphrase:**
(tulis di sini)

---

### KK · Index Terms · main.tex baris 42

**Current:**
Efficient inference, knowledge distillation, multilingual text classification, news topic classification, static embeddings.

**Makna:**
Lima index terms (urut alfabet, huruf kapital hanya di kata pertama): efficient inference, knowledge distillation, multilingual text classification, news topic classification, static embeddings.

**Locked:** —

**Paraphrase:**
(tulis di sini)

## I. Introduction

---

### PEN-01 · main.tex baris 48

**Current:**
News analytics pipelines assign articles to standard topics early, so that the more expensive stages only process the topics that matter. The taxonomy most used by news agencies is Media Topic of the International Press Telecommunications Council (IPTC). Its top level has 17 topics, such as politics, economy, business and finance, and sport [1], [2].

**Makna:**
Pipeline analitik berita memberi topik standar pada artikel sejak awal, supaya tahap-tahap yang lebih mahal hanya memproses topik yang relevan. Taksonomi yang paling banyak dipakai kantor berita adalah Media Topic dari International Press Telecommunications Council (IPTC). Level teratasnya berisi 17 topik, misalnya politics, economy, business and finance, dan sport [1], [2].

**Locked:** angka 17 · sitasi [1] (IPTC) 2024; [2] Kuzman & Ljubešić 2025

**Paraphrase:**
(tulis di sini)

---

### PEN-02 · main.tex baris 50

**Current:**
Kuzman and Ljubešić [2] released the first open multilingual IPTC classifier. They labeled articles in four languages with GPT-4o and used these labels to fine-tune XLM-RoBERTa-large [3]. The resulting model matches GPT-4o on a human-annotated test set. A follow-up study by the same group concluded that fine-tuned encoders are more practical than large language models (LLMs) for large-scale annotation [4]. An encoder with 559.9 million parameters is still expensive without a graphics processing unit (GPU). On one central processing unit (CPU) core the model needs 2,971 ms per article, and its fastest CPU configuration, an Open Neural Network Exchange (ONNX) export quantized to 8-bit integers (int8), still needs 1,239 ms (Section IV-D). Neither study reports this inference cost, although inference accounts for a large share of the computing demand and energy use of machine learning in production, for example 60% of the machine learning energy use reported by Google [5].

**Makna:**
Kuzman dan Ljubešić [2] merilis classifier IPTC multibahasa terbuka yang pertama. Mereka melabeli artikel dalam empat bahasa dengan GPT-4o, lalu memakai label itu untuk fine-tune XLM-RoBERTa-large [3]. Model hasilnya setara GPT-4o pada test set beranotasi manusia. Studi lanjutan dari kelompok yang sama menyimpulkan encoder hasil fine-tuning lebih praktis daripada large language model (LLM) untuk anotasi skala besar [4]. Tetapi encoder dengan 559.9 juta parameter tetap mahal tanpa GPU: di satu core CPU butuh 2,971 ms per artikel, dan konfigurasi CPU tercepatnya (export ONNX yang dikuantisasi ke bilangan bulat 8-bit, int8) masih butuh 1,239 ms (Bagian IV-D). Kedua studi tidak melaporkan biaya inference ini, padahal inference menyumbang porsi besar kebutuhan komputasi dan energi machine learning di produksi; contohnya 60% energi machine learning yang dilaporkan Google [5]. Singkatan yang didefinisikan di paragraf ini: LLM, GPU, CPU, ONNX, int8.

**Locked:** angka 559.9, 2,971 ms, 8, 1,239 ms, 60% · sitasi [2] Kuzman & Ljubešić 2025; [3] Conneau et al. 2020; [4] Kuzman Pungeršek et al. 2026; [5] Luccioni et al. 2024 · rujukan Section IV-D

**Paraphrase:**
(tulis di sini)

---

### PEN-03 · main.tex baris 52

**Current:**
The target is a topic classifier that covers many languages, runs in milliseconds per article on an ordinary CPU, and has a small footprint and few dependencies. Existing options fall short of this. Small distilled transformers [6], [7] gain about one order of magnitude. In the CPU benchmark of this study, MiniLM-L6 is 21 times faster than the ONNX int8 teacher. Int8 quantization alone gives a factor of 2.3 over PyTorch fp32. fastText [8] is much faster, but it learns its embeddings from scratch and therefore does not transfer across languages. This matters here because labeled IPTC data exists for only four languages.

**Makna:**
Yang dituju: classifier topik yang mencakup banyak bahasa, berjalan dalam hitungan milidetik per artikel di CPU biasa, kecil, dan dependensinya sedikit. Pilihan yang ada belum memenuhi ini. Transformer kecil hasil distilasi [6], [7] hanya mempercepat sekitar satu orde besaran; di benchmark CPU studi ini, MiniLM-L6 21 kali lebih cepat daripada teacher ONNX int8. Kuantisasi int8 saja memberi faktor 2.3 dibanding PyTorch fp32. fastText [8] jauh lebih cepat, tetapi embedding-nya dipelajari dari nol sehingga tidak bisa transfer antarbahasa. Ini penting karena data IPTC berlabel hanya ada untuk empat bahasa.

**Locked:** angka 21, 2.3 · sitasi [6] Sanh et al. 2019; [7] Wang et al. 2021; [8] Joulin et al. 2017

**Paraphrase:**
(tulis di sini)

---

### PEN-04 · main.tex baris 54

**Current:**
This paper proposes StaticKD, which distills [9] the XLM-R-large IPTC classifier into a student that, after training, is a single table of 17 logits per token. A teacher that works across languages can label unlabeled news in any language at no annotation cost. Multilingual static embeddings distilled from sentence encoders [10], [11], [12] place translation equivalents close to each other, and looking them up costs almost nothing. News topics are also strongly lexical, which keeps bag-of-words representations competitive for topic classification [13]. Because the classification head of StaticKD is linear, the averaged embedding and the head can be folded into a token table. Inference then reduces to tokenization and averaging table rows, which NumPy handles.

**Makna:**
Paper ini mengusulkan StaticKD, yang men-distill [9] classifier IPTC XLM-R-large menjadi student yang setelah training hanyalah satu tabel berisi 17 logit per token. Alasannya: (a) teacher yang bekerja lintas bahasa bisa melabeli berita tak berlabel dalam bahasa apa pun tanpa biaya anotasi; (b) static embedding multibahasa yang di-distill dari sentence encoder [10], [11], [12] menempatkan padanan terjemahan berdekatan, dan mencarinya hampir tanpa biaya; (c) topik berita sangat leksikal, sehingga representasi bag-of-words tetap kompetitif untuk klasifikasi topik [13]. Karena kepala klasifikasi StaticKD linear, embedding rata-rata dan kepala itu bisa dilipat menjadi tabel token. Inference lalu tinggal tokenisasi dan merata-rata baris tabel, yang cukup dikerjakan NumPy.

**Locked:** angka 17 · sitasi [9] Hinton et al. 2015; [10] Tulkens & van Dongen 2024; [11] Wada et al. 2025; [12] Chen et al. 2024; [13] Galke & Scherp 2022

**Paraphrase:**
(tulis di sini)

---

### PEN-05 · main.tex baris 56

**Current:**
A multilingual claim requires a multilingual evaluation with labels that do not come from the teacher. The human test set of Kuzman and Ljubešić is not public and covers four languages. SIB-200 [14] covers 205 languages, but its items are short sentences with a non-IPTC taxonomy. To fill this gap, a unified test set of 12,744 articles in 47 languages is built. It combines CC-NEWS articles from 2024 to 2026 that carry their newsroom section in schema.org metadata [15] with public datasets labeled by humans or by publisher categories [16], [17], [18], [19]. The labels are editorial decisions of publishers or annotators, translated onto the official IPTC tree. No model that reads the article text is involved.

**Makna:**
Klaim multibahasa butuh evaluasi multibahasa dengan label yang tidak berasal dari teacher. Test set manusia milik Kuzman dan Ljubešić tidak publik dan hanya mencakup empat bahasa. SIB-200 [14] mencakup 205 bahasa, tetapi isinya kalimat pendek dengan taksonomi non-IPTC. Untuk mengisi celah ini dibangun unified test set berisi 12,744 artikel dalam 47 bahasa. Isinya gabungan artikel CC-NEWS 2024 sampai 2026 yang membawa rubrik redaksinya di metadata schema.org [15], dengan dataset publik yang dilabeli manusia atau kategori penerbit [16], [17], [18], [19]. Labelnya adalah keputusan redaksi penerbit atau anotator, diterjemahkan ke pohon IPTC resmi. Tidak ada model yang membaca teks artikel yang terlibat dalam pelabelan.

**Locked:** angka 200, 205, 12,744, 47, 2024, 2026 · sitasi [14] Adelani et al. 2024; [15] Guha et al. 2016; [16] Adelani et al. 2023; [17] Petukhova & Fachada 2023; [18] Mirashi et al. 2023; [19] Hossain et al. 2025

**Paraphrase:**
(tulis di sini)

---

### PEN-06 · main.tex baris 58

**Current:**
This study addresses four research questions (RQs).

**Makna:**
Kalimat pengantar: studi ini menjawab empat research question (RQ).

**Locked:** —

**Paraphrase:**
(tulis di sini)

---

### PEN-07 · main.tex baris 59

**Current:**
RQ1 (cost): What is the CPU inference cost of the IPTC teacher, and how much does StaticKD reduce it?

**Makna:**
RQ1 (biaya): berapa biaya inference CPU teacher IPTC, dan seberapa besar StaticKD menguranginya?

**Locked:** —

**Paraphrase:**
(tulis di sini)

---

### PEN-08 · main.tex baris 59

**Current:**
RQ2 (accuracy): How much of the teacher accuracy does StaticKD retain on labels that do not come from the teacher, compared with baselines of similar cost, namely fastText and a support vector machine (SVM) on term frequency–inverse document frequency (TF-IDF) features, and with baselines of higher cost, namely MiniLM-L6 and DistilmBERT?

**Makna:**
RQ2 (akurasi): berapa bagian akurasi teacher yang dipertahankan StaticKD pada label yang tidak berasal dari teacher, dibandingkan baseline berbiaya serupa (fastText, dan support vector machine/SVM di atas fitur TF-IDF) dan baseline berbiaya lebih tinggi (MiniLM-L6 dan DistilmBERT)? Singkatan yang didefinisikan di sini: SVM, TF-IDF (term frequency–inverse document frequency).

**Locked:** —

**Paraphrase:**
(tulis di sini)

---

### PEN-09 · main.tex baris 59

**Current:**
RQ3 (components): Which distillation components determine the accuracy of this very-low-capacity student: soft labels, multilingual data, embedding initialization, embedding fine-tuning, token dropout, temperature, or ensembling?

**Makna:**
RQ3 (komponen): komponen distilasi mana yang menentukan akurasi student berkapasitas sangat rendah ini: soft label, data multibahasa, inisialisasi embedding, fine-tuning embedding, token dropout, temperature, atau ensemble?

**Locked:** —

**Paraphrase:**
(tulis di sini)

---

### PEN-10 · main.tex baris 59

**Current:**
RQ4 (generalization): On which languages, labels, and label sources is StaticKD weak, and which factors explain it?

**Makna:**
RQ4 (generalisasi): pada bahasa, label, dan sumber label mana StaticKD lemah, dan faktor apa yang menjelaskannya?

**Locked:** —

**Paraphrase:**
(tulis di sini)

---

### PEN-11 · main.tex baris 66

**Current:**
Distilling into a student of such low capacity also shows where its accuracy comes from. The knowledge that matters comes from the coverage of the transfer set and from the aligned initialization, not from the shape of the soft labels. Hard labels match soft labels, and a high temperature hurts. Transfer to languages that never appear in the distillation data is carried by embedding rows that training never updates, which works because they share a space with the trained rows. Fine-tuning the embeddings trades this transfer for in-distribution fit, a representation drift that a development set in the training languages cannot detect.

**Makna:**
Paragraf temuan utama. Distilasi ke student berkapasitas sangat rendah juga memperlihatkan dari mana akurasinya berasal. Pengetahuan yang penting berasal dari cakupan transfer set dan inisialisasi yang selaras, bukan dari bentuk soft label. Hard label setara soft label, dan temperature tinggi merugikan. Transfer ke bahasa yang tidak pernah muncul di data distilasi dibawa oleh baris embedding yang tidak pernah di-update saat training; ini berhasil karena baris itu berada di ruang yang sama dengan baris yang dilatih. Fine-tuning embedding menukar transfer ini dengan kecocokan in-distribution, yaitu representation drift yang tidak terdeteksi oleh development set dalam bahasa training.

**Catatan:** Catatan: klaim 'tidak pernah di-update' masih menunggu keputusan untuk dipersempit; lihat N-05 butir 13 di checklist.

**Locked:** —

**Paraphrase:**
(tulis di sini)

---

### PEN-12 · main.tex baris 68

**Current:**
The contributions of this paper are as follows.

**Makna:**
Kalimat pengantar: kontribusi paper ini adalah sebagai berikut.

**Locked:** —

**Paraphrase:**
(tulis di sini)

---

### PEN-13 · main.tex baris 69

**Current:**
A multilingual IPTC test set of 12,744 articles in 47 languages whose editorial labels do not come from the teacher. It comes with a section-to-IPTC mapping table that is validated automatically and reviewed by the first author, cross-source deduplication, a contamination check against all training data, and site-cluster bootstrap confidence intervals, and it can be reused to evaluate other IPTC classifiers.

**Makna:**
Kontribusi 1: test set IPTC multibahasa berisi 12,744 artikel dalam 47 bahasa yang label editorialnya tidak berasal dari teacher. Dilengkapi tabel pemetaan rubrik ke IPTC yang divalidasi otomatis dan ditinjau penulis pertama, deduplikasi lintas sumber, pemeriksaan kontaminasi terhadap seluruh data training, dan confidence interval site-cluster bootstrap. Test set ini bisa dipakai ulang untuk mengevaluasi classifier IPTC lain.

**Locked:** angka 12,744, 47

**Paraphrase:**
(tulis di sini)

---

### PEN-14 · main.tex baris 69

**Current:**
The findings above, supported by ablations with five seeds per variant and by analyses by language, label, label origin, and source, including languages that are absent from the distillation data.

**Makna:**
Kontribusi 2: temuan-temuan di atas, didukung ablasi dengan lima seed per varian dan analisis per bahasa, label, asal label, dan sumber, termasuk bahasa yang tidak ada di data distilasi.

**Locked:** —

**Paraphrase:**
(tulis di sini)

---

### PEN-15 · main.tex baris 69

**Current:**
StaticKD, a recipe that distills a multilingual transformer teacher into a token-logit table initialized from cross-lingual static embeddings. The final model takes 34 MB, needs only numpy and tokenizers, and processes 473 articles per second per core. An ensemble of seeds folds into the same table at no extra inference cost. Its accuracy–cost trade-off is compared with nine configurations from six model families, measured on CPU only with core pinning for a hybrid CPU.

**Makna:**
Kontribusi 3: StaticKD, resep yang men-distill teacher transformer multibahasa menjadi tabel logit per token yang diinisialisasi dari static embedding lintas bahasa. Model final berukuran 34 MB, hanya butuh numpy dan tokenizers, dan memproses 473 artikel per detik per core. Ensemble beberapa seed bisa dilipat ke tabel yang sama tanpa biaya inference tambahan. Trade-off akurasi dan biayanya dibandingkan dengan sembilan konfigurasi dari enam keluarga model, diukur hanya di CPU dengan core pinning untuk CPU hybrid.

**Locked:** angka 34 MB, 473 · nama numpy, tokenizers

**Paraphrase:**
(tulis di sini)

---

### PEN-16 · main.tex baris 69

**Current:**
A release of the code, a notebook that reproduces every number in this paper, the models, and the test-set manifest.

**Makna:**
Kontribusi 4: rilis kode, notebook yang mereproduksi setiap angka di paper, model, dan manifest test set.

**Locked:** —

**Paraphrase:**
(tulis di sini)

---

### PEN-17 · main.tex baris 76

**Current:**
Section II reviews related work, Section III describes the data, test set, method, and evaluation protocol, Section IV presents the results, Section V discusses them, Section VI lists the limitations, and Section VII concludes.

**Makna:**
Peta paper: Bagian II membahas penelitian terkait, Bagian III menjelaskan data, test set, metode, dan protokol evaluasi, Bagian IV menyajikan hasil, Bagian V membahasnya, Bagian VI memuat keterbatasan, dan Bagian VII menyimpulkan.

**Locked:** rujukan Section II, Section III, Section IV, Section V, Section VI, Section VII

**Paraphrase:**
(tulis di sini)

## II. Related Work

---

### PT-01 · Multilingual News Topic Classification and the IPTC Scheme · main.tex baris 81

**Current:**
Resources for news topic classification usually define their own taxonomy and cover a single language or region. MasakhaNEWS covers 16 languages widely spoken in Africa, English and French included, with seven human-annotated categories [16]. L3Cube-IndicNews uses publisher site categories for 10 Indian languages [18], and a Bangla news dataset also relies on publisher categories [19]. MN-DS is the only public dataset annotated directly with top-level IPTC topics, and it is English only [17]. SIB-200 spans 205 languages, but its items are Flores-200 sentences with seven non-IPTC topics [14]. Kuzman and Ljubešić [2] showed that GPT-4o labels can replace human annotation for training an IPTC classifier, and their follow-up work found fine-tuned encoders more practical than LLMs for large-scale annotation [4]. As a result, the only open multilingual IPTC classifier is an encoder with 559.9 million parameters, and no inexpensive IPTC classifier has been evaluated on real news in many languages.

**Makna:**
Subbagian: klasifikasi topik berita multibahasa dan skema IPTC. Sumber data klasifikasi topik berita biasanya punya taksonomi sendiri dan hanya satu bahasa atau wilayah. MasakhaNEWS mencakup 16 bahasa yang banyak dipakai di Afrika, termasuk Inggris dan Prancis, dengan tujuh kategori beranotasi manusia [16]. L3Cube-IndicNews memakai kategori situs penerbit untuk 10 bahasa India [18], dan dataset berita Bangla juga memakai kategori penerbit [19]. MN-DS adalah satu-satunya dataset publik yang dianotasi langsung dengan topik IPTC level atas, dan hanya bahasa Inggris [17]. SIB-200 mencakup 205 bahasa, tetapi isinya kalimat Flores-200 dengan tujuh topik non-IPTC [14]. Kuzman dan Ljubešić [2] menunjukkan label GPT-4o bisa menggantikan anotasi manusia untuk melatih classifier IPTC, dan studi lanjutannya menemukan encoder hasil fine-tuning lebih praktis daripada LLM untuk anotasi skala besar [4]. Akibatnya, satu-satunya classifier IPTC multibahasa terbuka adalah encoder 559.9 juta parameter, dan belum ada classifier IPTC murah yang dievaluasi pada berita nyata dalam banyak bahasa.

**Locked:** angka 16, 10, 200, 205, 559.9 · sitasi [16] Adelani et al. 2023; [18] Mirashi et al. 2023; [19] Hossain et al. 2025; [17] Petukhova & Fachada 2023; [14] Adelani et al. 2024; [2] Kuzman & Ljubešić 2025; [4] Kuzman Pungeršek et al. 2026

**Paraphrase:**
(tulis di sini)

---

### PT-02 · Distilling Transformer Classifiers into Compact Students · main.tex baris 84

**Current:**
Knowledge distillation (KD) trains a cheap model to imitate the temperature-softened output distribution of an expensive one [9]. A recent survey covers its development for language models [20]. In text classification the student is almost always a transformer, for example DistilBERT [6] or MiniLMv2 [7], including cross-lingual distillation that moves teacher knowledge to languages with little data [21]. The closest prior work is Sparse Distillation [22], which distills RoBERTa-large into a deep averaging network [23] with a very large n-gram table and runs hundreds of times faster on English single-sentence classification tasks. StaticKD differs in three ways. Its table starts from multilingual static embeddings that are already aligned across languages instead of being trained from scratch. Its linear head is folded so that each token stores only 17 numbers. It targets news documents in 69 languages and is evaluated on languages that are missing from the distillation data.

**Makna:**
Subbagian: men-distill classifier transformer menjadi student kecil. Knowledge distillation (KD) melatih model murah untuk meniru distribusi output model mahal yang dilunakkan dengan temperature [9]. Survei terbaru membahas perkembangannya untuk language model [20]. Di klasifikasi teks, student hampir selalu transformer, misalnya DistilBERT [6] atau MiniLMv2 [7], termasuk distilasi lintas bahasa yang memindahkan pengetahuan teacher ke bahasa dengan data sedikit [21]. Karya terdekat adalah Sparse Distillation [22], yang men-distill RoBERTa-large menjadi deep averaging network [23] dengan tabel n-gram sangat besar, dan ratusan kali lebih cepat pada tugas klasifikasi kalimat tunggal bahasa Inggris. StaticKD berbeda dalam tiga hal: (1) tabelnya dimulai dari static embedding multibahasa yang sudah selaras lintas bahasa, bukan dilatih dari nol; (2) kepala linearnya dilipat sehingga tiap token hanya menyimpan 17 angka; (3) menargetkan dokumen berita dalam 69 bahasa dan dievaluasi pada bahasa yang tidak ada di data distilasi. Singkatan yang didefinisikan: KD.

**Locked:** angka 17, 69 · sitasi [9] Hinton et al. 2015; [20] Yang et al. 2025; [6] Sanh et al. 2019; [7] Wang et al. 2021; [21] Piperno et al. 2025; [22] Ye et al. 2022; [23] Iyyer et al. 2015

**Paraphrase:**
(tulis di sini)

---

### PT-03 · Temperature and the Capacity Gap · main.tex baris 86

**Current:**
How well distillation works depends on the capacity gap between teacher and student. For language models, the optimal teacher size even follows an empirical law with respect to the student size [24]. Temperature is treated as an important hyperparameter that can be learned [25]. The Kullback–Leibler (KL) divergence can be split into a target-class part and a non-target part (dark knowledge) [26], and recent analysis shows that KL under a capacity mismatch distorts the ranking of classes [27]. Fidelity to the teacher also does not always go together with generalization [28]. Nearly all of these findings come from deep students. How temperature and soft labels behave at an extreme gap, from 559.9 million parameters to one linear layer over averaged embeddings, has not been characterized.

**Makna:**
Subbagian: temperature dan capacity gap. Keberhasilan distilasi bergantung pada capacity gap antara teacher dan student. Untuk language model, ukuran teacher optimal bahkan mengikuti hukum empiris terhadap ukuran student [24]. Temperature diperlakukan sebagai hyperparameter penting yang bisa dipelajari [25]. Divergensi Kullback–Leibler (KL) bisa dipecah menjadi bagian kelas target dan bagian non-target (dark knowledge) [26], dan analisis terbaru menunjukkan KL pada kapasitas yang tidak seimbang merusak urutan kelas [27]. Kesetiaan (fidelity) pada teacher juga tidak selalu sejalan dengan generalisasi [28]. Hampir semua temuan ini berasal dari student yang dalam (deep). Perilaku temperature dan soft label pada gap ekstrem, dari 559.9 juta parameter ke satu layer linear di atas embedding rata-rata, belum pernah dikarakterisasi. Singkatan yang didefinisikan: KL.

**Locked:** angka 559.9 · sitasi [24] Zhang et al. 2025; [25] Li et al. 2023; [26] Zhao et al. 2022; [27] Wang et al. 2026; [28] Stanton et al. 2021

**Paraphrase:**
(tulis di sini)

---

### PT-04 · Static Embeddings and Efficient CPU Inference · main.tex baris 89

**Current:**
Model2Vec [10] and Wada et al. [11] extract static word embeddings from sentence transformers, and averaging them yields competitive sentence embeddings. The potion-multilingual variant is distilled from BGE-M3 [12] for 101 languages. The Model2Vec library also includes a trainer that turns such a model into a classifier. It fine-tunes the embeddings and a weight per token, adds a hidden layer of 512 units, and trains with cross-entropy on hard labels [10]. Such embeddings are usually evaluated as general-purpose embeddings, not as distillation students for a specific taxonomy. Simple bag-of-words representations remain competitive for topic classification [13]. A survey of efficient natural language processing (NLP) methods recommends reporting inference cost next to accuracy [29], since inference makes up a large share of the energy use of deployed models [5]. Few studies, however, report CPU latency at low thread counts against a quantized teacher.

**Makna:**
Subbagian: static embedding dan inference CPU yang efisien. Model2Vec [10] dan Wada dkk. [11] mengekstrak static word embedding dari sentence transformer; merata-ratakannya menghasilkan sentence embedding yang kompetitif. Varian potion-multilingual di-distill dari BGE-M3 [12] untuk 101 bahasa. Library Model2Vec juga punya trainer untuk mengubah model seperti ini menjadi classifier: trainer itu fine-tune embedding dan bobot per token, menambah hidden layer 512 unit, dan dilatih dengan cross-entropy pada hard label [10]. Embedding seperti ini biasanya dievaluasi sebagai embedding serbaguna, bukan sebagai student distilasi untuk taksonomi tertentu. Representasi bag-of-words sederhana tetap kompetitif untuk klasifikasi topik [13]. Survei metode NLP yang efisien menganjurkan melaporkan biaya inference di samping akurasi [29], karena inference menyumbang porsi besar penggunaan energi model yang sudah di-deploy [5]. Namun hanya sedikit studi yang melaporkan latency CPU dengan jumlah thread kecil terhadap teacher yang dikuantisasi. Singkatan yang didefinisikan: NLP (natural language processing).

**Catatan:** Catatan review: kalimat terakhir adalah klaim literatur tanpa sitasi.

**Locked:** angka 101, 512 · sitasi [10] Tulkens & van Dongen 2024; [11] Wada et al. 2025; [12] Chen et al. 2024; [13] Galke & Scherp 2022; [29] Treviso et al. 2023; [5] Luccioni et al. 2024

**Paraphrase:**
(tulis di sini)

---

### PT-05 · Test Sets with Editorial Labels, Contamination, and Sample Dependence · main.tex baris 92

**Current:**
Publisher site categories have been used as topic labels before [18], usually for one publisher or one region. The schema.org articleSection metadata [15] has not yet been mapped to a standard taxonomy across dozens of languages. Leakage between training and test data also has to be controlled, and near-duplicate removal with MinHash [30] has become standard practice [31]. Significance tests in NLP such as the paired bootstrap [32] usually assume independent samples [33], while articles from the same site are dependent. The cluster bootstrap [34] handles this dependence by resampling clusters instead of documents. Table I summarizes how StaticKD relates to these approaches.

**Makna:**
Subbagian: test set dengan label editorial, kontaminasi, dan dependensi sampel. Kategori situs penerbit sudah pernah dipakai sebagai label topik [18], biasanya untuk satu penerbit atau satu wilayah. Metadata schema.org articleSection [15] belum pernah dipetakan ke taksonomi standar lintas puluhan bahasa. Kebocoran antara data training dan test juga harus dikendalikan, dan penghapusan near-duplicate dengan MinHash [30] sudah menjadi praktik standar [31]. Uji signifikansi di NLP seperti paired bootstrap [32] biasanya mengasumsikan sampel independen [33], padahal artikel dari situs yang sama saling bergantung. Cluster bootstrap [34] menangani dependensi ini dengan me-resample klaster, bukan dokumen. Tabel I merangkum posisi StaticKD terhadap pendekatan-pendekatan ini.

**Locked:** sitasi [18] Mirashi et al. 2023; [15] Guha et al. 2016; [30] Broder 1997; [31] Lee et al. 2022; [32] Koehn 2004; [33] Dror et al. 2018; [34] Field & Welsh 2007 · rujukan Table I · nama articleSection

**Paraphrase:**
(tulis di sini)

[TABEL I — placeholder] Posisi StaticKD dibanding pendekatan terkait (kolom: pendekatan, model saat inference, multibahasa, transfer lintas bahasa, tanpa library deep learning, biaya CPU dilaporkan, label IPTC).

## III. Methodology

---

### MET-01 · Study Design · main.tex baris 118

**Current:**
Fig. 1 summarizes the study. The teacher labels a distillation corpus, on which StaticKD and all baselines are trained. A separate test set, whose labels do not come from the teacher, serves for the accuracy evaluation with statistical tests and for the subgroup analyses, and inference cost is measured on CPU. The StaticKD recipe was fixed on the development set before the test set was built, and no hyperparameter was selected on the test set. During this exploration, a small check set of 320 CC-News articles in 16 languages, labeled by Claude with the prompt and label descriptions of [2], also served as a secondary signal next to the development set and the held-out agreement. It accompanied the choice of τ=1 and of the three-seed table ensemble, formed from the three seeds run in that phase. LLM labels are too close to the teacher setting to serve as an independent test, so this check set was dropped from the final evaluation.

**Makna:**
Subbagian: desain studi. Gambar 1 merangkum studi. Teacher melabeli korpus distilasi, lalu StaticKD dan semua baseline dilatih pada data itu. Test set terpisah yang labelnya tidak berasal dari teacher dipakai untuk evaluasi akurasi dengan uji statistik dan analisis subgrup, dan biaya inference diukur di CPU. Resep StaticKD sudah ditetapkan pada development set SEBELUM test set dibangun, dan tidak ada hyperparameter yang dipilih memakai test set. Selama eksplorasi itu, check set kecil berisi 320 artikel CC-News dalam 16 bahasa, dilabeli Claude dengan prompt dan deskripsi label dari [2], juga dipakai sebagai sinyal sekunder di samping development set dan held-out agreement. Check set ini ikut menyertai pemilihan tau=1 dan ensemble tabel tiga seed (dari tiga seed yang dijalankan di fase itu). Label LLM terlalu dekat dengan setting teacher untuk dijadikan uji independen, sehingga check set ini dibuang dari evaluasi final.

**Locked:** angka 320, 16 · sitasi [2] Kuzman & Ljubešić 2025 · rujukan Fig. 1 · simbol τ=1

**Paraphrase:**
(tulis di sini)

[GAMBAR 1 — placeholder] Pipeline StaticKD. Teacher melabeli korpus berita tak berlabel satu kali, secara offline. Student belajar meniru distribusi teacher, lalu kepala linearnya dilipat ke tabel token, sehingga inference hanya mencari dan merata-rata baris tabel. Evaluasi final memakai test set yang labelnya tidak berasal dari teacher. (Diagram TikZ di main.tex; tidak perlu digambar ulang.)

---

### MET-02 · Problem Formulation · main.tex baris 151

**Current:**
Given a news article d in any language, consisting of its title and body truncated to the first 512 words as in [2], a model predicts one label y∈Y among the 17 top-level IPTC topics. The teacher f_T outputs logits z^T(d)∈ℝ^17. The goal is a student f_S whose inference costs O(n) additions for n tokens, without matrix multiplication and without a deep learning library, and whose accuracy against independent labels is as close as possible to that of f_T.

**Makna:**
Subbagian: formulasi masalah. Diberikan artikel berita d dalam bahasa apa pun (judul dan isi, dipotong ke 512 kata pertama seperti di [2]), model memprediksi satu label y dari 17 topik IPTC level atas. Teacher f_T mengeluarkan logit z^T(d) berdimensi 17. Tujuannya student f_S yang biaya inference-nya O(n) penjumlahan untuk n token, tanpa perkalian matriks dan tanpa library deep learning, dan akurasinya terhadap label independen sedekat mungkin dengan f_T.

**Locked:** angka 512, 17 · sitasi [2] Kuzman & Ljubešić 2025 · simbol d, y∈Y, f_T, z^T(d)∈ℝ^17, f_S, O(n), n

**Paraphrase:**
(tulis di sini)

---

### MET-03 · Distillation Data and Development Set · main.tex baris 154

**Current:**
EMMediaTopic 1.0 [35], [2] contains Catalan, Greek, Croatian, and Slovenian articles labeled by GPT-4o. Its training split (20,000 articles) is part of the distillation data. Its development split (1,000 articles, median 218 words) was never used to train the teacher or the student and serves as the development set for choosing the recipe.

**Makna:**
Subbagian: data distilasi dan development set. EMMediaTopic 1.0 [35] [2] berisi artikel berbahasa Katalan, Yunani, Kroasia, dan Slovenia yang dilabeli GPT-4o. Split training-nya (20,000 artikel) menjadi bagian data distilasi. Split development-nya (1,000 artikel, median 218 kata) tidak pernah dipakai untuk melatih teacher maupun student, dan menjadi development set untuk memilih resep.

**Locked:** angka 1.0, 20,000, 1,000, 218 · sitasi [35] Kuzman & Ljubešić 2025; [2] Kuzman & Ljubešić 2025

**Paraphrase:**
(tulis di sini)

---

### MET-04 · Distillation Data and Development Set · main.tex baris 156

**Current:**
The multilingual CC-News corpus [36] from the 2019–2021 archive was processed with the following steps.

**Makna:**
Kalimat pengantar daftar: korpus CC-News multibahasa [36] dari arsip 2019-2021 diproses dengan langkah-langkah berikut.

**Locked:** angka 2019, 2021 · sitasi [36] CloverSearch n.d.

**Paraphrase:**
(tulis di sini)

---

### MET-05 · Distillation Data and Development Set · main.tex baris 157

**Current:**
Keep only documents whose metadata language agrees with fastText language identification [8].

**Makna:**
Langkah 1: simpan hanya dokumen yang bahasa di metadatanya sama dengan hasil identifikasi bahasa fastText [8].

**Locked:** sitasi [8] Joulin et al. 2017

**Paraphrase:**
(tulis di sini)

---

### MET-06 · Distillation Data and Development Set · main.tex baris 157

**Current:**
Take the title and body with at least 75 words (200 characters for ja, zh, th, my) and truncate to 512 words.

**Makna:**
Langkah 2: ambil judul dan isi dengan minimal 75 kata (200 karakter untuk ja, zh, th, my), lalu potong ke 512 kata.

**Locked:** angka 75, 200, 512

**Paraphrase:**
(tulis di sini)

---

### MET-07 · Distillation Data and Development Set · main.tex baris 157

**Current:**
Keep at most 150 documents per site.

**Makna:**
Langkah 3: simpan paling banyak 150 dokumen per situs.

**Locked:** angka 150

**Paraphrase:**
(tulis di sini)

---

### MET-08 · Distillation Data and Development Set · main.tex baris 157

**Current:**
Form a held-out split by domain: all documents of some sites are set aside, at most 20% per language.

**Makna:**
Langkah 4: bentuk split held-out berdasarkan domain: semua dokumen dari sebagian situs disisihkan, paling banyak 20% per bahasa.

**Locked:** angka 20%

**Paraphrase:**
(tulis di sini)

---

### MET-09 · Distillation Data and Development Set · main.tex baris 157

**Current:**
Add 156,157 documents from training sites only (at most 5,000 per language and 400 per site).

**Makna:**
Langkah 5: tambahkan 156,157 dokumen hanya dari situs training (paling banyak 5,000 per bahasa dan 400 per situs).

**Locked:** angka 156,157, 5,000, 400

**Paraphrase:**
(tulis di sini)

---

### MET-10 · Distillation Data and Development Set · main.tex baris 164

**Current:**
The distillation data contain 285,999 documents in 69 languages from 2,960 sites, with a median of 269 words. The held-out split is domain-disjoint within each language. A few multilingual sites (for example bbc.com) also appear in other languages of the distillation data, so their articles were removed. The remaining fidelity set has 12,554 documents in 63 languages from 217 sites that never occur in the distillation data.

**Makna:**
Data distilasi berisi 285,999 dokumen dalam 69 bahasa dari 2,960 situs, dengan median 269 kata. Split held-out terpisah domain di setiap bahasa. Beberapa situs multibahasa (misalnya bbc.com) juga muncul di bahasa lain dalam data distilasi, jadi artikelnya dibuang. Fidelity set yang tersisa berisi 12,554 dokumen dalam 63 bahasa dari 217 situs yang tidak pernah muncul di data distilasi.

**Locked:** angka 285,999, 69, 2,960, 269, 12,554, 63, 217

**Paraphrase:**
(tulis di sini)

---

### MET-11 · Unified Test Set Construction · main.tex baris 168

**Current:**
The first source is CC-NEWS 2024–2026, sampled as one WARC file per month from January 2024 to August 2026 (32 files) of the Common Crawl CC-NEWS archive [37]. From each page, the schema.org JSON-LD blocks [15] of type NewsArticle and its subtypes were extracted, with the fields articleSection, BreadcrumbList, inLanguage, datePublished, isAccessibleForFree, and articleBody. The text comes from articleBody when it has at least 30 words and is otherwise extracted with Trafilatura [38]. Sites that occur in the distillation data or in another evaluation set were removed, which leaves 399,468 articles from new sites.

**Makna:**
Subbagian: konstruksi unified test set. Sumber pertama adalah CC-NEWS 2024-2026: satu file WARC per bulan dari Januari 2024 sampai Agustus 2026 (32 file) dari arsip Common Crawl CC-NEWS [37]. Dari setiap halaman diekstrak blok JSON-LD schema.org [15] bertipe NewsArticle dan subtipenya, dengan field articleSection, BreadcrumbList, inLanguage, datePublished, isAccessibleForFree, dan articleBody. Teks diambil dari articleBody bila minimal 30 kata; kalau tidak, diekstrak dengan Trafilatura [38]. Situs yang muncul di data distilasi atau di set evaluasi lain dibuang, sehingga tersisa 399,468 artikel dari situs baru.

**Locked:** angka 2024, 2026, 32, 30, 399,468 · sitasi [37] Foundation 2026; [15] Guha et al. 2016; [38] Barbaresi 2021 · nama NewsArticle, articleSection, BreadcrumbList, inLanguage, datePublished, isAccessibleForFree, articleBody

**Paraphrase:**
(tulis di sini)

---

### MET-12 · Unified Test Set Construction · main.tex baris 170

**Current:**
The second source is public datasets. MasakhaNEWS [16] and MN-DS [17] are labeled by humans. L3Cube-IndicNews (LDC split) [18], Bangla news [19], IndoNews [39], Amharic news [40], and Swahili news [41] carry publisher categories. The last two were processed in the same way but were not selected during sampling, because MasakhaNEWS already filled their language cells with human labels. The second source holds 312,551 articles in total.

**Makna:**
Sumber kedua adalah dataset publik. MasakhaNEWS [16] dan MN-DS [17] dilabeli manusia. L3Cube-IndicNews (split LDC) [18], berita Bangla [19], IndoNews [39], berita Amharik [40], dan berita Swahili [41] membawa kategori penerbit. Dua yang terakhir diproses dengan cara yang sama tetapi tidak terpilih saat sampling, karena sel bahasanya sudah diisi MasakhaNEWS dengan label manusia. Total sumber kedua berisi 312,551 artikel.

**Locked:** angka 312,551 · sitasi [16] Adelani et al. 2023; [17] Petukhova & Fachada 2023; [18] Mirashi et al. 2023; [19] Hossain et al. 2025; [39] Research 2021; [40] Azime & Mohammed 2021; [41] David 2020

**Paraphrase:**
(tulis di sini)

---

### MET-13 · Unified Test Set Construction · main.tex baris 172

**Current:**
Section and category labels represent editorial topics, that is, the decision of the newsroom about where an article belongs when it is published. These labels come from no model that reads the article text, so they are not circular with respect to the teacher. The only added step is translating section names onto the official IPTC Media Topic tree (1,085 active nodes). Section names are normalized (case, diacritics) and mapped conservatively. A name is mapped only if the content usually found under that section belongs to one top-level topic. This gives 814 names mapped to 79 nodes. The table deliberately leaves 62 ambiguous names unmapped, for example tech, which can be science and technology or the information technology industry under economy. Likewise, entertainment can be arts or lifestyle, and terrorism can be conflict or crime. Astrology sections are rejected. An automatic check verifies that every target node exists in the IPTC tree, is not retired, and has the target label as its level-1 ancestor. Articles whose sections map to different labels are rejected. MN-DS needs no mapping, because its labels already are IPTC topics. The mapping table was drafted with artificial intelligence assistance, and the first author reviewed all of its 916 rules (814 mapped names and 102 ambiguous, language-specific, and rejecting rules) and accepted them without changes.

**Makna:**
Label rubrik (section) dan kategori mewakili topik editorial, yaitu keputusan redaksi tentang di mana artikel ditempatkan saat diterbitkan. Label ini tidak berasal dari model yang membaca teks artikel, jadi tidak sirkular terhadap teacher. Satu-satunya langkah tambahan adalah menerjemahkan nama rubrik ke pohon IPTC Media Topic resmi (1,085 node aktif). Nama rubrik dinormalisasi (huruf besar-kecil, diakritik) dan dipetakan secara konservatif: sebuah nama hanya dipetakan bila isi yang biasa ada di rubrik itu masuk ke satu topik level atas. Hasilnya 814 nama dipetakan ke 79 node. Tabel pemetaan sengaja membiarkan 62 nama ambigu tidak dipetakan, misalnya tech (bisa science and technology, atau industri teknologi informasi di bawah economy); begitu juga entertainment (arts atau lifestyle) dan terrorism (conflict atau crime). Rubrik astrologi ditolak. Pemeriksaan otomatis memastikan setiap node tujuan ada di pohon IPTC, tidak retired, dan leluhur level-1-nya adalah label tujuan. Artikel yang rubrik-rubriknya terpetakan ke label berbeda ditolak. MN-DS tidak perlu dipetakan karena labelnya sudah topik IPTC. Tabel pemetaan disusun dengan bantuan artificial intelligence, dan penulis pertama meninjau seluruh 916 aturannya (814 nama terpetakan dan 102 aturan ambigu, khusus bahasa, dan penolakan) lalu menerimanya tanpa perubahan.

**Locked:** angka 1,085, 814, 79, 62, 1, 916, 102

**Paraphrase:**
(tulis di sini)

---

### MET-14 · Unified Test Set Construction · main.tex baris 174

**Current:**
Quality and leakage filters are applied in order.

**Makna:**
Kalimat pengantar daftar: filter kualitas dan kebocoran diterapkan secara berurutan.

**Locked:** —

**Paraphrase:**
(tulis di sini)

---

### MET-15 · Unified Test Set Construction · main.tex baris 175

**Current:**
Publication date on or after 1 January 2024 (CC-NEWS only).

**Makna:**
Filter 1: tanggal terbit pada atau setelah 1 Januari 2024 (hanya CC-NEWS). Tulis tanggal dalam format IEEE: 1 January 2024.

**Locked:** angka 1, 2024

**Paraphrase:**
(tulis di sini)

---

### MET-16 · Unified Test Set Construction · main.tex baris 175

**Current:**
Not behind a paywall.

**Makna:**
Filter 2: tidak berada di balik paywall.

**Locked:** —

**Paraphrase:**
(tulis di sini)

---

### MET-17 · Unified Test Set Construction · main.tex baris 175

**Current:**
Declared language agrees with fastText language identification when the language is supported.

**Makna:**
Filter 3: bahasa yang dideklarasikan sama dengan identifikasi bahasa fastText, bila bahasanya didukung fastText.

**Locked:** —

**Paraphrase:**
(tulis di sini)

---

### MET-18 · Unified Test Set Construction · main.tex baris 175

**Current:**
At least 75 words.

**Makna:**
Filter 4: minimal 75 kata.

**Locked:** angka 75

**Paraphrase:**
(tulis di sini)

---

### MET-19 · Unified Test Set Construction · main.tex baris 175

**Current:**
Deduplication of uniform resource locators (URLs), identical texts, and near-duplicates (MinHash [30], Jaccard ≥ 0.7 [31]) across all sources.

**Makna:**
Filter 5: deduplikasi URL, teks identik, dan near-duplicate (MinHash [30], Jaccard >= 0.7 [31]) lintas semua sumber. Singkatan URL didefinisikan di sini (uniform resource locator).

**Locked:** sitasi [30] Broder 1997; [31] Lee et al. 2022 · simbol ≥ 0.7

**Paraphrase:**
(tulis di sini)

---

### MET-20 · Unified Test Set Construction · main.tex baris 175

**Current:**
Contamination: an article is dropped when ≥ 30% of its word 8-grams occur in any of about 300 thousand training or evaluation documents. This filter removed 50 articles, mostly from MasakhaNEWS, which also draws on BBC.

**Makna:**
Filter 6, kontaminasi: artikel dibuang bila >= 30% word 8-gram-nya muncul di salah satu dari sekitar 300 ribu dokumen training atau evaluasi. Filter ini membuang 50 artikel, sebagian besar dari MasakhaNEWS, yang juga mengambil dari BBC.

**Locked:** angka 8, 300, 50 · simbol ≥ 30

**Paraphrase:**
(tulis di sini)

---

### MET-21 · Unified Test Set Construction · main.tex baris 184

**Current:**
Sampling takes at most 50 articles per language × label cell, preferring human-labeled sources, then publisher categories, then JSON-LD sections. Languages with fewer than 30 articles are dropped. The number of articles per site is not capped, because most cells come from few sites and such a cap would discard data without adding diversity. The evaluation handles the dependence between articles of the same site with a site-cluster bootstrap, and Section IV-A reports a sensitivity test with at most five articles per site.

**Makna:**
Sampling mengambil paling banyak 50 artikel per sel bahasa x label, dengan urutan prioritas: sumber berlabel manusia, lalu kategori penerbit, lalu rubrik JSON-LD. Bahasa dengan kurang dari 30 artikel dibuang. Jumlah artikel per situs tidak dibatasi, karena kebanyakan sel hanya berasal dari sedikit situs, dan batas seperti itu hanya membuang data tanpa menambah keragaman. Evaluasi menangani dependensi antarartikel dari situs yang sama dengan site-cluster bootstrap, dan Bagian IV-A melaporkan uji sensitivitas dengan paling banyak lima artikel per situs.

**Locked:** angka 50, 30 · rujukan Section IV-A · simbol ×

**Paraphrase:**
(tulis di sini)

---

### MET-22 · Unified Test Set Construction · main.tex baris 186

**Current:**
The final test set contains 12,744 articles in 47 languages, with 17 labels and 1,227 sites or sources and a median of 313 words. Of these, 4,094 articles (32.1%) carry human annotations, 2,600 (20.4%) publisher categories, and 6,050 (47.5%) JSON-LD sections. By the distillation data in their language, 13 languages are entirely absent from the distillation data (for example ha, yo, ig, bn, pa), 9 have fewer than 500 documents, and 25 have more. By site diversity, 13 languages have at least 10 effective sites, while 27 languages come from practically one publisher. Table II summarizes the sources and Fig. 2 shows the language × label coverage.

**Makna:**
Test set final berisi 12,744 artikel dalam 47 bahasa, dengan 17 label dan 1,227 situs atau sumber, median 313 kata. Dari jumlah itu, 4,094 artikel (32.1%) berlabel anotasi manusia, 2,600 (20.4%) kategori penerbit, dan 6,050 (47.5%) rubrik JSON-LD. Menurut data distilasi di bahasanya: 13 bahasa sama sekali tidak ada di data distilasi (misalnya ha, yo, ig, bn, pa), 9 bahasa punya kurang dari 500 dokumen, dan 25 bahasa punya lebih. Menurut keragaman situs: 13 bahasa punya minimal 10 situs efektif, sedangkan 27 bahasa praktis hanya dari satu penerbit. Tabel II merangkum sumbernya dan Gambar 2 menunjukkan cakupan bahasa x label.

**Locked:** angka 12,744, 47, 17, 1,227, 313, 4,094, 32.1%, 2,600, 20.4%, 6,050, 47.5%, 13, 9, 500, 25, 10, 27 · rujukan Table II, Fig. 2 · simbol ×

**Paraphrase:**
(tulis di sini)

[TABEL II — placeholder] Komposisi unified test set per sumber.

[GAMBAR 2 — placeholder] Jumlah artikel di unified test set per bahasa dan label (paling banyak 50 per sel). Asal label di dalam kurung adalah asal label dominan untuk bahasa tersebut.

---

### MET-23 · Teacher and Distillation Labels · main.tex baris 219

**Current:**
The teacher is the IPTC classifier of [2], published as classla/multilingual-IPTC-news-topic-classifier. It is an XLM-RoBERTa-large model with 24 layers, 559.9 million parameters, and inputs of at most 512 tokens. Its architecture limits the input to 512 XLM-R tokens. The 512-word input of 60.7% of the test articles exceeds this limit, and for those articles the teacher reads about 287 of 438 words (medians). For every distillation document the teacher produces a logit vector z^T(d)∈ℝ^17, which is stored in full. Labeling ran once in fp16 on an NVIDIA RTX 4050 Laptop GPU (6 GB), with batches sorted by text length. On a fixed sample of 5,000 documents the teacher labels 50.1 documents per second, so the whole distillation set takes about 1.58 GPU hours. The arg-max predictions of this re-run agree with the stored labels for 100% of the documents. The teacher label distribution on CC-News is plausible (politics 16.5%, economy 12.7%, sport 11.6%, weather 1.3%), and its mean confidence per language ranges from 0.90 to 1.00.

**Makna:**
Subbagian: teacher dan label distilasi. Teacher adalah classifier IPTC dari [2], dipublikasikan sebagai classla/multilingual-IPTC-news-topic-classifier. Model XLM-RoBERTa-large dengan 24 layer, 559.9 juta parameter, dan input paling banyak 512 token. Arsitekturnya membatasi input ke 512 token XLM-R. Input 512 kata dari 60.7% artikel uji melebihi batas ini, dan untuk artikel tersebut teacher hanya membaca sekitar 287 dari 438 kata (median). Untuk setiap dokumen distilasi, teacher menghasilkan vektor logit z^T(d) berdimensi 17, yang disimpan utuh. Pelabelan dijalankan sekali dalam fp16 di GPU NVIDIA RTX 4050 Laptop (6 GB), dengan batch diurutkan menurut panjang teks. Pada sampel tetap 5,000 dokumen, teacher melabeli 50.1 dokumen per detik, sehingga seluruh data distilasi butuh sekitar 1.58 jam GPU. Prediksi arg-max dari run ulang ini sama dengan label tersimpan untuk 100% dokumen. Distribusi label teacher di CC-News masuk akal (politics 16.5%, economy 12.7%, sport 11.6%, weather 1.3%), dan rata-rata confidence per bahasa berkisar 0.90 sampai 1.00.

**Locked:** angka 24, 559.9, 512, 60.7%, 287, 438, 4050, 6 GB, 5,000, 50.1, 1.58, 100%, 16.5%, 12.7%, 11.6%, 1.3%, 0.90, 1.00 · sitasi [2] Kuzman & Ljubešić 2025 · simbol z^T(d)∈ℝ^17 · nama classla/multilingual-IPTC-news-topic-classifier

**Paraphrase:**
(tulis di sini)

---

### MET-24 · The StaticKD Student · main.tex baris 222

**Current:**
A document d is tokenized into t_1,…,t_n with the potion-multilingual-128M Unigram tokenizer [10], which has 500,353 tokens. Special tokens are not used, unknown tokens are dropped, and the length is capped at 1,024 tokens. Averaging has no position limit, so the cap only bounds the cost. Only 5.4% of the test articles exceed it. Each token has an embedding e_t∈ℝ^256, and

**Makna:**
Subbagian: student StaticKD. Dokumen d ditokenisasi menjadi t_1 ... t_n dengan Unigram tokenizer potion-multilingual-128M [10], yang punya 500,353 token. Special token tidak dipakai, token tak dikenal dibuang, dan panjang dibatasi 1,024 token. Perataan tidak punya batas posisi, jadi batas ini hanya membatasi biaya; hanya 5.4% artikel uji yang melebihinya. Tiap token punya embedding e_t berdimensi 256, dan [kalimat berlanjut ke Persamaan (1)].

**Locked:** angka 500,353, 1,024, 5.4% · sitasi [10] Tulkens & van Dongen 2024 · simbol d, t_1,…,t_n, e_t∈ℝ^256

**Paraphrase:**
(tulis di sini)

[PERSAMAAN (1) — placeholder] Persamaan student: h(d) = rata-rata embedding token dokumen; z^S(d) = W dikali dropout(h(d)) ditambah b. Jangan ditulis ulang; salin persamaannya apa adanya.

---

### MET-25 · The StaticKD Student · main.tex baris 228

**Current:**
with W∈ℝ^17× 256 and dropout 0.1 during training. The embedding matrix E is initialized from potion-multilingual, static embeddings for 101 languages distilled from BGE-M3 [12].

**Makna:**
Lanjutan setelah Persamaan (1): dengan W berukuran 17 x 256 dan dropout 0.1 saat training. Matriks embedding E diinisialisasi dari potion-multilingual, yaitu static embedding untuk 101 bahasa yang di-distill dari BGE-M3 [12].

**Locked:** angka 0.1, 101 · sitasi [12] Chen et al. 2024 · simbol W∈ℝ^17× 256, E

**Paraphrase:**
(tulis di sini)

---

### MET-26 · The StaticKD Student · main.tex baris 230

**Current:**
The loss in (2) uses p^T=softmax(z^T/τ) and p^S=softmax(z^S/τ):

**Makna:**
Kalimat pengantar Persamaan (2): loss di Persamaan (2) memakai p^T = softmax(z^T / tau) dan p^S = softmax(z^S / tau).

**Locked:** rujukan (2) · simbol p^T=softmax(z^T/τ), p^S=softmax(z^S/τ)

**Paraphrase:**
(tulis di sini)

[PERSAMAAN (2) — placeholder] Loss distilasi: L_KD = tau^2 dikali divergensi KL antara p^T dan p^S, dengan tau = 1. Salin apa adanya.

---

### MET-27 · The StaticKD Student · main.tex baris 235

**Current:**
E, W, and b are trained end to end. E uses SparseAdam (η=10^-3), so only the rows of tokens present in a batch are updated, while W and b use AdamW (η=3· 10^-3, weight decay 10^-4). The batch size is 64 and training runs for 6 epochs. As a cheap augmentation, each token of a training document is dropped with probability 0.2 at every step (token dropout), so that the model does not rely on a few keywords. Internal validation uses 3% of the distillation data (8,579 documents), and the epoch with the highest arg-max agreement with the teacher is kept.

**Makna:**
E, W, dan b dilatih end to end. E memakai SparseAdam (eta = 10^-3), sehingga hanya baris token yang ada di batch yang di-update; W dan b memakai AdamW (eta = 3 x 10^-3, weight decay 10^-4). Batch size 64 dan training berjalan 6 epoch. Sebagai augmentasi murah, setiap token dokumen training dibuang dengan probabilitas 0.2 di setiap langkah (token dropout), supaya model tidak bergantung pada sedikit kata kunci. Validasi internal memakai 3% data distilasi (8,579 dokumen), dan epoch dengan arg-max agreement tertinggi terhadap teacher yang disimpan.

**Locked:** angka 64, 6, 0.2, 3%, 8,579 · simbol E, W, b, η=10^-3, η=3· 10^-3, 10^-4

**Paraphrase:**
(tulis di sini)

---

### MET-28 · The StaticKD Student · main.tex baris 237

**Current:**
Because the classification head is linear, (1) can be rewritten as

**Makna:**
Kalimat pengantar Persamaan (3): karena kepala klasifikasi linear, Persamaan (1) bisa ditulis ulang sebagai [Persamaan (3)].

**Locked:** rujukan (1)

**Paraphrase:**
(tulis di sini)

[PERSAMAAN (3) — placeholder] Pelipatan kepala: W dikali rata-rata embedding ditambah b sama dengan rata-rata baris tabel T ditambah b. Salin apa adanya.

---

### MET-29 · The StaticKD Student · main.tex baris 242

**Current:**
with the table T=EW^ᵀ∈ℝ^|V|× 17. The production model consists only of T (fp16), b, and the tokenizer, and inference is tokenization, averaging table rows, and adding the bias. Averaging the logits of K students trained with different seeds is likewise equal to the logits of a single student with table T̄=1/KΣ_k T^(k) and bias b̄, because (3) is linear in T. The final StaticKD model is the table ensemble of seeds 0, 1, and 2 (K=3), with 8.5 million parameters and a size of 34 MB.

**Makna:**
Lanjutan setelah Persamaan (3): dengan tabel T = E W^T berukuran |V| x 17. Model produksi hanya terdiri dari T (fp16), b, dan tokenizer; inference = tokenisasi, merata-rata baris tabel, lalu menambahkan bias. Merata-ratakan logit dari K student dengan seed berbeda juga sama dengan logit satu student dengan tabel T-bar = (1/K) jumlah T^(k) dan bias b-bar, karena Persamaan (3) linear terhadap T. Model StaticKD final adalah ensemble tabel dari seed 0, 1, dan 2 (K=3), dengan 8.5 juta parameter dan ukuran 34 MB.

**Locked:** angka 0, 1, 2, 8.5, 34 MB · rujukan (3) · simbol T=EW^ᵀ∈ℝ^|V|× 17, T, b, K, T̄=1/KΣ_k T^(k), b̄, K=3

**Paraphrase:**
(tulis di sini)

---

### MET-30 · The StaticKD Student · main.tex baris 244

**Current:**
The internal structure of the Unigram tokenizer needs much more random-access memory (RAM) than the table itself (16 MB). A compact variant shrinks that structure by removing token pieces that occur fewer than 10 times in the distillation corpus, keeping single characters. Its vocabulary has 335,441 tokens and the model takes 23 MB. Table III lists the hyperparameters of all models.

**Makna:**
Struktur internal Unigram tokenizer butuh RAM jauh lebih besar daripada tabelnya sendiri (16 MB). Varian compact mengecilkan struktur itu dengan membuang potongan token yang muncul kurang dari 10 kali di korpus distilasi, tetapi tetap menyimpan karakter tunggal. Vocabulary-nya 335,441 token dan modelnya 23 MB. Tabel III mencantumkan hyperparameter semua model. Singkatan RAM (random-access memory) didefinisikan di sini.

**Locked:** angka 16 MB, 10, 335,441, 23 MB · rujukan Table III

**Paraphrase:**
(tulis di sini)

[TABEL III — placeholder] Hyperparameter semua model.

---

### MET-31 · Baselines · main.tex baris 284

**Current:**
All baselines are trained on the same distillation data (285,999 documents) with targets from the same teacher. fastText + KD [8] is trained on teacher arg-max labels with word tokens (plus character n-grams) or XLM-R subwords, together with a quantized variant (.ftz). As a reference equivalent to the labeled data available to [2], fastText is also trained only on the GPT-4o labels of the EMMediaTopic training split. TF-IDF + SVM + KD is the non-neural baseline of Kuzman and Ljubešić (TF-IDF features with a support vector classifier), trained on teacher arg-max labels over XLM-R subwords. The hyperparameters of these non-transformer baselines were set once, as listed in Table III, without a search. MiniLM-L6 + KD uses mMiniLMv2-L6-H384, distilled from XLM-R-large [7], with the temperature τ∈{1,2} chosen on the development set. DistilmBERT + KD uses distilbert-base-multilingual-cased [6] with the same recipe. Both transformers read at most 256 tokens, which truncates 91.1% (MiniLM-L6) and 93.2% (DistilmBERT) of the test articles to about 142 and 116 words. Both are exported to ONNX fp32 for the CPU benchmark. The ONNX int8 teacher is a dynamic int8 quantized export of the teacher and is its fastest CPU configuration.

**Makna:**
Subbagian: baseline. Semua baseline dilatih pada data distilasi yang sama (285,999 dokumen) dengan target dari teacher yang sama. fastText + KD [8] dilatih pada label arg-max teacher dengan token kata (plus character n-gram) atau subword XLM-R, ditambah varian terkuantisasi (.ftz). Sebagai pembanding yang setara dengan data berlabel yang dimiliki [2], fastText juga dilatih hanya pada label GPT-4o dari split training EMMediaTopic. TF-IDF + SVM + KD adalah baseline non-neural milik Kuzman dan Ljubešić (fitur TF-IDF dengan support vector classifier), dilatih pada label arg-max teacher di atas subword XLM-R. Hyperparameter baseline non-transformer ini ditetapkan sekali, seperti di Tabel III, tanpa pencarian. MiniLM-L6 + KD memakai mMiniLMv2-L6-H384 (hasil distilasi dari XLM-R-large) [7], dengan temperature tau dalam {1, 2} yang dipilih pada development set. DistilmBERT + KD memakai distilbert-base-multilingual-cased [6] dengan resep yang sama. Kedua transformer hanya membaca paling banyak 256 token, sehingga memotong 91.1% (MiniLM-L6) dan 93.2% (DistilmBERT) artikel uji menjadi sekitar 142 dan 116 kata. Keduanya di-export ke ONNX fp32 untuk benchmark CPU. Teacher ONNX int8 adalah export teacher yang dikuantisasi int8 secara dinamis, dan merupakan konfigurasi CPU tercepatnya.

**Locked:** angka 285,999, 256, 91.1%, 93.2%, 142, 116 · sitasi [8] Joulin et al. 2017; [2] Kuzman & Ljubešić 2025; [7] Wang et al. 2021; [6] Sanh et al. 2019 · rujukan Table III · simbol τ∈{1,2} · nama .ftz, distilbert-base-multilingual-cased

**Paraphrase:**
(tulis di sini)

---

### MET-32 · Baselines · main.tex baris 286

**Current:**
Sparse Distillation [22], the closest prior work, was not reproduced in full. Two experiments of this study cover its core ideas. fastText + KD is a bag-of-n-grams model with averaging and a linear head, trained on teacher labels over the same transfer set. The random-initialization ablation (Section III-J) trains the embedding table from scratch. What remains different is the hidden layers of the deep averaging network, the soft-logit targets, and the billion-parameter n-gram table designed for English. The frozen-embedding ablation, which trains only the linear head on top of potion, serves as an off-the-shelf Model2Vec baseline with a distilled head. The classifier trainer of the Model2Vec library was not run as a baseline. Its default configuration trains on hard labels and adds a 512-unit hidden layer. It also learns a weight per token. The ablations replace soft labels with hard labels and the linear head with a 512-unit hidden layer one at a time, and neither change differs significantly from the final recipe on the development or test set (Table VIII). The two changes together and the learned token weights were not tested. A hidden layer also prevents folding the model into one table.

**Makna:**
Sparse Distillation [22], karya terdekat, tidak direproduksi sepenuhnya. Dua eksperimen studi ini mencakup ide intinya: fastText + KD adalah model bag-of-n-grams dengan perataan dan kepala linear yang dilatih pada label teacher di transfer set yang sama; ablasi inisialisasi acak (Bagian III-J) melatih tabel embedding dari nol. Perbedaan yang tersisa: hidden layer pada deep averaging network, target berupa soft logit, dan tabel n-gram miliaran parameter yang dirancang untuk bahasa Inggris. Ablasi embedding beku (hanya melatih kepala linear di atas potion) berfungsi sebagai baseline Model2Vec siap pakai dengan kepala hasil distilasi. Trainer classifier dari library Model2Vec tidak dijalankan sebagai baseline. Konfigurasi default-nya dilatih pada hard label, menambah hidden layer 512 unit, dan juga mempelajari bobot per token. Ablasi mengganti soft label dengan hard label, dan kepala linear dengan hidden layer 512 unit, satu per satu; tidak satu pun berbeda signifikan dari resep final di development maupun test set (Tabel VIII). Kombinasi keduanya dan bobot token yang dipelajari tidak diuji. Hidden layer juga membuat model tidak bisa dilipat menjadi satu tabel.

**Locked:** angka 512 · sitasi [22] Ye et al. 2022 · rujukan Section III-J, Table VIII

**Paraphrase:**
(tulis di sini)

---

### MET-33 · Evaluation Protocol and Statistical Tests · main.tex baris 290

**Current:**
The main metric is macro-F1 over the labels present in the reference, as in the evaluation code of [2], because the label distribution is imbalanced. The evaluation also reports accuracy, macro-F1 per language averaged over languages so that large languages do not dominate, and agreement with the teacher on the fidelity set. On editorial labels, macro-F1 reads as agreement between the predictions and the categorization of the publisher.

**Makna:**
Subbagian: protokol evaluasi dan uji statistik. Metrik utama adalah macro-F1 atas label yang ada di referensi, seperti kode evaluasi [2], karena distribusi label tidak seimbang. Evaluasi juga melaporkan akurasi, macro-F1 per bahasa yang dirata-rata antarbahasa (supaya bahasa besar tidak mendominasi), dan agreement dengan teacher pada fidelity set. Pada label editorial, macro-F1 dibaca sebagai kesesuaian prediksi dengan kategorisasi penerbit.

**Locked:** sitasi [2] Kuzman & Ljubešić 2025

**Paraphrase:**
(tulis di sini)

---

### MET-34 · Evaluation Protocol and Statistical Tests · main.tex baris 292

**Current:**
95% confidence intervals come from a percentile bootstrap with 1,000 resamples. On the test set, the resampling unit is the site, or the source dataset when no URL is available (site-cluster bootstrap [34]). On the development set, the unit is the document. Two models on the same documents are compared with a paired (cluster) bootstrap with 2,000 resamples for Δmacro-F1 [32] and an exact McNemar test for accuracy [42], with Holm correction [43] within each set. Each ablation variant uses five seeds, and the final recipe uses ten. Differences between variants are tested with Welch tests at the seed level, Holm-corrected. Whether a difference holds consistently across languages is tested with a Wilcoxon signed-rank test on per-language macro-F1 and accuracy [33]. Subgroup analyses cover label origin, region, the amount of distillation data of the language, site diversity, script, language, and label. Calibration is measured with the expected calibration error (ECE, 15 bins) [44].

**Makna:**
Confidence interval 95% berasal dari percentile bootstrap dengan 1,000 resample. Di test set, unit resample-nya adalah situs, atau dataset sumber bila tidak ada URL (site-cluster bootstrap [34]); di development set, unitnya dokumen. Dua model pada dokumen yang sama dibandingkan dengan paired (cluster) bootstrap 2,000 resample untuk selisih macro-F1 [32] dan uji McNemar exact untuk akurasi [42], dengan koreksi Holm [43] di dalam setiap set. Setiap varian ablasi memakai lima seed, dan resep final sepuluh. Perbedaan antarvarian diuji dengan uji Welch di level seed, dikoreksi Holm. Konsistensi perbedaan antarbahasa diuji dengan Wilcoxon signed-rank pada macro-F1 dan akurasi per bahasa [33]. Analisis subgrup mencakup asal label, wilayah, jumlah data distilasi bahasa, keragaman situs, aksara, bahasa, dan label. Kalibrasi diukur dengan expected calibration error (ECE, 15 bin) [44].

**Locked:** angka 95%, 1,000, 2,000, 15 · sitasi [34] Field & Welsh 2007; [32] Koehn 2004; [42] McNemar 1947; [43] Holm 1979; [33] Dror et al. 2018; [44] Guo et al. 2017 · simbol Δ

**Paraphrase:**
(tulis di sini)

---

### MET-35 · Efficiency Protocol · main.tex baris 296

**Current:**
The benchmark runs on CPU only, with the GPU hidden. Each model runs in a fresh process, and the thread count is pinned in every library (OMP, MKL, OpenBLAS, ONNX Runtime, tokenizers). Latency is measured with one article per call, as in a streaming setting (p50 and p95), and throughput with batches of 64, or 8 for transformers. Model RAM is the increase in resident set size after loading the model. The text sample is drawn at random from the unified test set and is the same for every model: 30 latency and 48 throughput documents for the teacher and the transformers, and 500 and 3,000 for the light models. All models are measured in one session on an idle machine. StaticKD reads up to 1,024 tokens per article while the teacher stops at 512, so the speed comparison does not favor StaticKD.

**Makna:**
Subbagian: protokol efisiensi. Benchmark berjalan hanya di CPU dengan GPU disembunyikan. Setiap model dijalankan di proses baru, dan jumlah thread dikunci di semua library (OMP, MKL, OpenBLAS, ONNX Runtime, tokenizers). Latency diukur dengan satu artikel per panggilan seperti skenario streaming (p50 dan p95), dan throughput dengan batch 64, atau 8 untuk transformer. RAM model adalah kenaikan resident set size setelah model dimuat. Sampel teks diambil acak dari unified test set dan sama untuk semua model: 30 dokumen latency dan 48 dokumen throughput untuk teacher dan transformer, serta 500 dan 3,000 untuk model ringan. Semua model diukur dalam satu sesi pada mesin yang idle. StaticKD membaca sampai 1,024 token per artikel sementara teacher berhenti di 512, jadi perbandingan kecepatan ini tidak menguntungkan StaticKD.

**Locked:** angka 64, 8, 30, 48, 500, 3,000, 1,024, 512 · nama tokenizers

**Paraphrase:**
(tulis di sini)

---

### MET-36 · Efficiency Protocol · main.tex baris 298

**Current:**
The machine is a laptop with an Intel Core Ultra 7 155H, 31 GB of RAM, and Windows 11. This CPU is hybrid, with 6 P-cores, 8 slower E-cores, and 2 low-power E-cores. Without pinning, the operating system moves threads to the slow cores. Each process is therefore pinned to physical P-cores (logical CPU 0 for one core; logical CPUs 0, 10, 12, and 14 for four cores), which approximates a small server with uniform cores.

**Makna:**
Mesinnya laptop dengan Intel Core Ultra 7 155H, RAM 31 GB, dan Windows 11. CPU ini hybrid: 6 P-core, 8 E-core yang lebih lambat, dan 2 low-power E-core. Tanpa pinning, sistem operasi memindahkan thread ke core yang lambat. Karena itu setiap proses di-pin ke P-core fisik (logical CPU 0 untuk satu core; logical CPU 0, 10, 12, dan 14 untuk empat core), yang mendekati server kecil dengan core seragam.

**Locked:** angka 7, 31 GB, 11, 6, 8, 2, 0, 10, 12, 14

**Paraphrase:**
(tulis di sini)

---

### MET-37 · Experimental Design and Implementation Details · main.tex baris 302

**Current:**
Table VIII lists all 72 StaticKD runs. Each ablation variant changes one factor of the final recipe: hard labels instead of soft labels; EMMediaTopic data only (4 languages, 10 epochs); 130 thousand documents without the extra CC-News data; random initialization with the same standard deviation as potion; frozen embeddings; no token dropout; a 512-unit multilayer perceptron (MLP) head; and τ∈{0.5, 2, 4}. Four exploratory variants were also tested: bigram hashing (2^21 buckets), idf^0.5-weighted pooling, document weights from teacher confidence, and an additional cross-entropy term on the GPT-4o labels. One StaticKD run takes 5.2 minutes on average on the same GPU. The implementation uses Python 3.12, NumPy 2.5, PyTorch 2.6, transformers 5.17, tokenizers 0.23, ONNX Runtime 1.30, scikit-learn 1.9, and fastText 0.9.2. The seed, library versions, and training history of every run are stored with the model.

**Makna:**
Subbagian: desain eksperimen dan detail implementasi. Tabel VIII mencantumkan seluruh 72 run StaticKD. Setiap varian ablasi mengubah satu faktor dari resep final: hard label alih-alih soft label; hanya data EMMediaTopic (4 bahasa, 10 epoch); 130 ribu dokumen tanpa data CC-News tambahan; inisialisasi acak dengan standar deviasi yang sama dengan potion; embedding beku; tanpa token dropout; kepala multilayer perceptron (MLP) 512 unit; dan tau dalam {0.5, 2, 4}. Empat varian eksplorasi juga diuji: bigram hashing (2^21 bucket), pooling berbobot idf^0.5, bobot dokumen dari confidence teacher, dan suku cross-entropy tambahan pada label GPT-4o. Satu run StaticKD rata-rata butuh 5.2 menit di GPU yang sama. Implementasinya memakai Python 3.12, NumPy 2.5, PyTorch 2.6, transformers 5.17, tokenizers 0.23, ONNX Runtime 1.30, scikit-learn 1.9, dan fastText 0.9.2. Seed, versi library, dan riwayat training setiap run disimpan bersama modelnya.

**Locked:** angka 72, 4, 10, 130, 512, 5.2, 3.12, 2.5, 2.6, 5.17, 0.23, 1.30, 1.9, 0.9 · rujukan Table VIII · simbol τ∈{0.5, 2, 4}, 2^21, ^0.5 · nama tokenizers

**Paraphrase:**
(tulis di sini)

## IV. Results

[TABEL IV — placeholder] Hasil utama: macro-F1 [95% CI] dan akurasi di development set (EMMediaTopic dev, label GPT-4o), agreement dengan teacher di held-out CC-News, dan hasil di unified test set (47 bahasa, label independen). CI uji dari site-cluster bootstrap.

---

### HAS-01 · Accuracy of All Models (RQ2) · main.tex baris 339

**Current:**
Table IV gives the main results. On the unified test set the teacher reaches a macro-F1 of 0.598 [0.546–0.643] and an accuracy of 0.700. This is far below its development score (0.822), because the test set covers languages, sources, and a period the teacher has never seen. StaticKD reaches a macro-F1 of 0.545 [0.490–0.605] and an accuracy of 0.641, which is 91.1% of the teacher macro-F1. On the development set StaticKD keeps 92.6% of the teacher macro-F1 (0.761 against 0.822), and its agreement with the teacher over the 63 held-out languages is 0.811. A single student without ensembling reaches 0.542 ± 0.004 on the test set and 0.755 ± 0.010 on the development set (10 seeds).

**Makna:**
Subbagian: akurasi semua model (RQ2). Tabel IV memuat hasil utama. Di unified test set, teacher mencapai macro-F1 0.598 [0.546-0.643] dan akurasi 0.700. Angka ini jauh di bawah skor development-nya (0.822), karena test set mencakup bahasa, sumber, dan periode waktu yang belum pernah dilihat teacher. StaticKD mencapai macro-F1 0.545 [0.490-0.605] dan akurasi 0.641, yaitu 91.1% dari macro-F1 teacher. Di development set, StaticKD mempertahankan 92.6% macro-F1 teacher (0.761 vs 0.822), dan agreement-nya dengan teacher di 63 bahasa held-out adalah 0.811. Satu student tanpa ensemble mencapai 0.542 ± 0.004 di test set dan 0.755 ± 0.010 di development set (10 seed).

**Locked:** angka 0.598, 0.546, 0.643, 0.700, 0.822, 0.545, 0.490, 0.605, 0.641, 91.1%, 92.6%, 0.761, 63, 0.811, 0.542, 0.004, 0.755, 0.010, 10 · rujukan Table IV · simbol ±

**Paraphrase:**
(tulis di sini)

---

### HAS-02 · Accuracy of All Models (RQ2) · main.tex baris 341

**Current:**
Among the small distilled transformers, MiniLM-L6 (0.566), which starts from a distillation of XLM-R-large, lies between StaticKD and the teacher. DistilmBERT (0.534) is on par with StaticKD while being much more expensive. The two baselines of similar cost, fastText + KD and TF-IDF + SVM + KD, fall below StaticKD. fastText trained only on the GPT-4o labels of four languages fails outside its training languages, with a test macro-F1 of 0.086 against 0.665 on the development set.

**Makna:**
Di antara transformer kecil hasil distilasi, MiniLM-L6 (0.566), yang berawal dari distilasi XLM-R-large, berada di antara StaticKD dan teacher. DistilmBERT (0.534) setara (on par) dengan StaticKD padahal jauh lebih mahal. Dua baseline berbiaya serupa, fastText + KD dan TF-IDF + SVM + KD, berada di bawah StaticKD. fastText yang hanya dilatih pada label GPT-4o dari empat bahasa gagal di luar bahasa training-nya: macro-F1 uji 0.086, padahal 0.665 di development set.

**Locked:** angka 0.566, 0.534, 0.086, 0.665

**Paraphrase:**
(tulis di sini)

[TABEL V — placeholder] Uji berpasangan StaticKD terhadap model lain (delta = StaticKD dikurangi model pembanding; p dikoreksi Holm di dalam tiap set; b/c = dokumen yang benar hanya untuk StaticKD / hanya untuk model pembanding).

---

### HAS-03 · Accuracy of All Models (RQ2) · main.tex baris 378

**Current:**
Table V reports paired tests with Δ = StaticKD − comparison model on the test set. Against the teacher, Δmacro-F1 is −0.053 [−0.065, −0.033] (p < 0.001), and against MiniLM-L6 it is −0.021 [−0.038, −0.008] (p 0.004). The difference to DistilmBERT, +0.010 [−0.026, +0.041], is not significant (p 0.800). StaticKD is correct on more test documents than DistilmBERT (McNemar p < 0.001), but the McNemar test treats documents as independent and ignores the dependence between articles of the same site. StaticKD outperforms the quantized fastText + KD by +0.110 [+0.042, +0.151] (p < 0.001) and TF-IDF + SVM by +0.059 [+0.009, +0.093] (p 0.030).

**Makna:**
Tabel V melaporkan uji berpasangan dengan delta = StaticKD dikurangi model pembanding, di test set. Terhadap teacher, selisih macro-F1 -0.053 [-0.065, -0.033] (p < 0.001); terhadap MiniLM-L6, -0.021 [-0.038, -0.008] (p 0.004). Selisih terhadap DistilmBERT, +0.010 [-0.026, +0.041], tidak signifikan (p 0.800). StaticKD benar pada lebih banyak dokumen uji daripada DistilmBERT (McNemar p < 0.001), tetapi uji McNemar menganggap dokumen independen dan mengabaikan dependensi antarartikel dari situs yang sama. StaticKD mengungguli fastText + KD terkuantisasi sebesar +0.110 [+0.042, +0.151] (p < 0.001) dan TF-IDF + SVM sebesar +0.059 [+0.009, +0.093] (p 0.030).

**Locked:** angka 0.053, 0.065, 0.033, 0.001, 0.021, 0.038, 0.008, 0.004, +0.010, 0.026, +0.041, 0.800, +0.110, +0.042, +0.151, +0.059, +0.009, +0.093, 0.030 · rujukan Table V · simbol Δ, −, p, <

**Paraphrase:**
(tulis di sini)

---

### HAS-04 · Accuracy of All Models (RQ2) · main.tex baris 380

**Current:**
These two advantages differ in strength when examined per language. StaticKD has a higher macro-F1 than fastText + KD in 43 of 47 languages (Wilcoxon p < 0.001) and a higher accuracy in 41 (Wilcoxon p < 0.001). Against TF-IDF + SVM it has a higher macro-F1 in 31 of 47 languages (Wilcoxon p 0.003), but a higher accuracy in only 24 and a lower accuracy in 22 (Wilcoxon p 0.127). The advantage over TF-IDF + SVM is thus consistent across languages on the main metric, while the per-language accuracies of the two models are close. Against DistilmBERT neither metric differs consistently across languages (Wilcoxon p 0.604 for macro-F1 and 0.322 for accuracy). StaticKD beats the teacher in only 7 of 47 languages (Wilcoxon p < 0.001).

**Makna:**
Kekuatan dua keunggulan itu berbeda bila dilihat per bahasa. Terhadap fastText + KD, macro-F1 StaticKD lebih tinggi di 43 dari 47 bahasa (Wilcoxon p < 0.001) dan akurasinya lebih tinggi di 41 (Wilcoxon p < 0.001). Terhadap TF-IDF + SVM, macro-F1-nya lebih tinggi di 31 dari 47 bahasa (Wilcoxon p 0.003), tetapi akurasinya hanya lebih tinggi di 24 dan lebih rendah di 22 (Wilcoxon p 0.127). Jadi keunggulan atas TF-IDF + SVM konsisten antarbahasa pada metrik utama, sedangkan akurasi per bahasa kedua model berdekatan. Terhadap DistilmBERT, tidak ada metrik yang berbeda secara konsisten antarbahasa (Wilcoxon p 0.604 untuk macro-F1 dan 0.322 untuk akurasi). StaticKD hanya mengalahkan teacher di 7 dari 47 bahasa (Wilcoxon p < 0.001).

**Locked:** angka 43, 47, 0.001, 41, 31, 0.003, 24, 22, 0.127, 0.604, 0.322, 7 · simbol p, <

**Paraphrase:**
(tulis di sini)

---

### HAS-05 · Accuracy of All Models (RQ2) · main.tex baris 382

**Current:**
By label origin (Table VI, Fig. 3), StaticKD retains 95.8% of the teacher score on articles with JSON-LD section labels, 89.5% on publisher categories, and 89.6% on human annotations. By the distillation data of the language, it retains 97.5% in languages with sufficient data, 89.1% in languages with little data, and 72.3% in languages with no distillation data at all. On a subset with at most five articles per site and cell (8,139 articles), the model ranking is unchanged relative to the full test set (StaticKD 0.587, teacher 0.627).

**Makna:**
Menurut asal label (Tabel VI, Gambar 3), StaticKD mempertahankan 95.8% skor teacher pada artikel berlabel rubrik JSON-LD, 89.5% pada kategori penerbit, dan 89.6% pada anotasi manusia. Menurut data distilasi bahasanya, StaticKD mempertahankan 97.5% di bahasa dengan data cukup, 89.1% di bahasa dengan data sedikit, dan 72.3% di bahasa tanpa data distilasi sama sekali. Pada subset dengan paling banyak lima artikel per situs dan sel (8,139 artikel), urutan model tidak berubah dibanding test set penuh (StaticKD 0.587, teacher 0.627).

**Locked:** angka 95.8%, 89.5%, 89.6%, 97.5%, 89.1%, 72.3%, 8,139, 0.587, 0.627 · rujukan Table VI, Fig. 3

**Paraphrase:**
(tulis di sini)

[TABEL VI — placeholder] Macro-F1 uji per subgrup. Kolom terakhir: macro-F1 StaticKD sebagai persentase macro-F1 teacher di subgrup yang sama.

---

### HAS-06 · Accuracy of All Models (RQ2) · main.tex baris 418

**Current:**
Table VII splits the test set by source. On MN-DS, the only source annotated directly with IPTC topics, StaticKD keeps 97.4% of the teacher score (0.535 against 0.549), and the teacher itself reaches its lowest macro-F1 of all sources there. The lowest retention is on MasakhaNEWS (80.0%), whose human labels follow a non-IPTC scheme and most of whose African languages are absent from the distillation data. Bangla is also absent from the distillation data. On Bangla news, fastText + KD and TF-IDF + SVM collapse to 0.150 and 0.010, while StaticKD reaches 0.748. This gap shows the cross-lingual transfer that baselines without pre-trained embeddings lack.

**Makna:**
Tabel VII memecah test set menurut sumber. Di MN-DS, satu-satunya sumber yang dianotasi langsung dengan topik IPTC, StaticKD mempertahankan 97.4% skor teacher (0.535 vs 0.549), dan teacher sendiri mendapat macro-F1 terendahnya di sana dibanding sumber lain. Retensi terendah ada di MasakhaNEWS (80.0%), yang label manusianya memakai skema non-IPTC dan sebagian besar bahasa Afrikanya tidak ada di data distilasi. Bahasa Bangla juga tidak ada di data distilasi. Pada berita Bangla, fastText + KD dan TF-IDF + SVM jatuh ke 0.150 dan 0.010, sedangkan StaticKD mencapai 0.748. Selisih ini memperlihatkan transfer lintas bahasa yang tidak dimiliki baseline tanpa embedding pra-latih.

**Locked:** angka 97.4%, 0.535, 0.549, 80.0%, 0.150, 0.010, 0.748 · rujukan Table VII

**Paraphrase:**
(tulis di sini)

[TABEL VII — placeholder] Macro-F1 uji per sumber unified test set. MN-DS satu-satunya sumber yang dianotasi langsung dengan topik IPTC; MasakhaNEWS dianotasi manusia dengan skemanya sendiri. Kolom terakhir: StaticKD sebagai persentase teacher.

[GAMBAR 3 — placeholder] Macro-F1 uji menurut asal label (kiri) dan menurut jumlah data distilasi di bahasa artikel (kanan).

---

### HAS-07 · Validity of the Editorial Labels · main.tex baris 449

**Current:**
Section and category labels record newsroom decisions, so the question is whether they agree with human labels in the same language.

**Makna:**
Subbagian: validitas label editorial. Label rubrik dan kategori mencatat keputusan redaksi, jadi pertanyaannya: apakah label itu sejalan dengan label manusia dalam bahasa yang sama?

**Locked:** —

**Paraphrase:**
(tulis di sini)

---

### HAS-08 · Validity of the Editorial Labels · main.tex baris 451

**Current:**
All English cells of the test set are filled with human-labeled sources. As a comparison within the same language, a separate sample of 754 English CC-NEWS articles with JSON-LD section labels was therefore drawn from 306 sites outside the test set, at most 50 per label. The teacher reaches an accuracy of 0.646 on the section labels (macro-F1 0.617) and 0.559 on the MN-DS and MasakhaNEWS human labels (macro-F1 0.565). Accuracy on section labels is higher for 12 of the 17 labels (Wilcoxon p 0.033), and the ranking of label difficulty correlates between the two label types (ρ=0.63, p 0.007). The hard labels, society, human interest, and religion, are hard under both.

**Makna:**
Semua sel bahasa Inggris di test set sudah diisi sumber berlabel manusia. Karena itu, sebagai pembanding dalam bahasa yang sama, diambil sampel terpisah berisi 754 artikel CC-NEWS bahasa Inggris berlabel rubrik JSON-LD dari 306 situs di luar test set, paling banyak 50 per label. Teacher mencapai akurasi 0.646 pada label rubrik (macro-F1 0.617) dan 0.559 pada label manusia MN-DS dan MasakhaNEWS (macro-F1 0.565). Akurasi pada label rubrik lebih tinggi untuk 12 dari 17 label (Wilcoxon p 0.033), dan urutan tingkat kesulitan label berkorelasi antara kedua jenis label (rho = 0.63, p 0.007). Label yang sulit, yaitu society, human interest, dan religion, sulit di kedua jenis label.

**Locked:** angka 754, 306, 50, 0.646, 0.617, 0.559, 0.565, 12, 17, 0.033, 0.007 · simbol p, ρ=0.63

**Paraphrase:**
(tulis di sini)

---

### HAS-09 · Validity of the Editorial Labels · main.tex baris 453

**Current:**
French offers a second comparison inside the test set. The teacher reaches an accuracy of 0.716 on 208 articles with section labels and 0.500 on 200 human-labeled MasakhaNEWS articles.

**Makna:**
Bahasa Prancis memberi perbandingan kedua di dalam test set: teacher mencapai akurasi 0.716 pada 208 artikel berlabel rubrik dan 0.500 pada 200 artikel MasakhaNEWS berlabel manusia.

**Locked:** angka 0.716, 208, 0.500, 200

**Paraphrase:**
(tulis di sini)

---

### HAS-10 · Validity of the Editorial Labels · main.tex baris 455

**Current:**
A last check looks at each source category, restricted to languages covered by XLM-R pre-training. The median agreement of the teacher with the mapped label per category is 0.46 for MasakhaNEWS, 0.50 for MN-DS, 0.81 for L3Cube, and 0.87 for CC-NEWS sections. The lowest agreement occurs for source categories that are broader than their IPTC target. For example, only 0.45 of MasakhaNEWS politics articles are predicted as politics, with conflict (29%) as the main alternative. Agreement is also low for labels whose boundaries are unclear even to human annotators, such as society in MN-DS, in line with the low inter-annotator agreement reported for that label [2]. Such mismatches reflect differences between the category scheme of a source and IPTC. Since every model is scored against the same labels, comparisons between models are unaffected, and absolute scores read as agreement with the editorial categorization.

**Makna:**
Pemeriksaan terakhir melihat tiap kategori sumber, dibatasi pada bahasa yang tercakup pre-training XLM-R. Median agreement teacher dengan label hasil pemetaan per kategori adalah 0.46 untuk MasakhaNEWS, 0.50 untuk MN-DS, 0.81 untuk L3Cube, dan 0.87 untuk rubrik CC-NEWS. Agreement terendah terjadi pada kategori sumber yang lebih luas daripada target IPTC-nya; misalnya hanya 0.45 artikel politics MasakhaNEWS yang diprediksi politics, dengan conflict (29%) sebagai alternatif utama. Agreement juga rendah untuk label yang batasnya tidak jelas bahkan bagi anotator manusia, seperti society di MN-DS, sejalan dengan rendahnya inter-annotator agreement yang dilaporkan untuk label itu [2]. Ketidakcocokan seperti ini mencerminkan perbedaan skema kategori sumber dengan IPTC. Karena semua model dinilai terhadap label yang sama, perbandingan antarmodel tidak terpengaruh, dan skor absolut dibaca sebagai kesesuaian dengan kategorisasi editorial.

**Locked:** angka 0.46, 0.50, 0.81, 0.87, 0.45, 29% · sitasi [2] Kuzman & Ljubešić 2025

**Paraphrase:**
(tulis di sini)

[TABEL VIII — placeholder] Ablasi, temperature, dan varian eksplorasi (satu student, tanpa ensemble). Rata-rata ± standar deviasi antar-seed; tanda bintang: berbeda dari resep final (Welch, Holm, p < 0.05).

[TABEL IX — placeholder] Macro-F1 varian terpilih menurut jumlah data distilasi di bahasa artikel (test set; rata-rata antar-seed).

---

### HAS-11 · Ablation, Temperature, and Ensembling (RQ3) · main.tex baris 510

**Current:**
Table VIII gives the contribution of each component with five seeds per variant. Multilingual distillation data matter most. Distilling on EMMediaTopic alone (4 languages) lowers the test macro-F1 by −0.051 and the held-out agreement by −0.077. Random initialization changes the test macro-F1 by −0.120 (p < 0.001), so the cross-lingually aligned embeddings account for a large share of the accuracy.

**Makna:**
Subbagian: ablasi, temperature, dan ensemble (RQ3). Tabel VIII memberi kontribusi tiap komponen dengan lima seed per varian. Data distilasi multibahasa paling berpengaruh: distilasi hanya dengan EMMediaTopic (4 bahasa) menurunkan macro-F1 uji sebesar -0.051 dan held-out agreement sebesar -0.077. Inisialisasi acak mengubah macro-F1 uji sebesar -0.120 (p < 0.001), jadi embedding yang selaras lintas bahasa menyumbang porsi besar akurasi.

**Locked:** angka 4, 0.051, 0.077, 0.120, 0.001 · rujukan Table VIII · simbol −, p, <

**Paraphrase:**
(tulis di sini)

---

### HAS-12 · Ablation, Temperature, and Ensembling (RQ3) · main.tex baris 512

**Current:**
Freezing the embeddings and training only the linear head on top of potion lowers the development score by −0.035 (p < 0.001) and the held-out agreement by −0.020 (p < 0.001). On the test set the same variant raises macro-F1 by +0.014 (p < 0.001). Table IX shows that this gain comes from languages with little distillation data (+0.059) and with none (+0.037). In data-rich languages the frozen embeddings are weaker (−0.007).

**Makna:**
Membekukan embedding dan hanya melatih kepala linear di atas potion menurunkan skor development sebesar -0.035 (p < 0.001) dan held-out agreement sebesar -0.020 (p < 0.001). Di test set, varian yang sama justru menaikkan macro-F1 sebesar +0.014 (p < 0.001). Tabel IX menunjukkan kenaikan ini berasal dari bahasa dengan data distilasi sedikit (+0.059) dan tanpa data (+0.037). Di bahasa yang datanya banyak, embedding beku lebih lemah (-0.007).

**Locked:** angka 0.035, 0.001, 0.020, +0.014, +0.059, +0.037, 0.007 · rujukan Table IX · simbol −, p, <

**Paraphrase:**
(tulis di sini)

---

### HAS-13 · Ablation, Temperature, and Ensembling (RQ3) · main.tex baris 514

**Current:**
Hard teacher labels change the test macro-F1 by +0.005 (p 0.433) and the development score by −0.006 (p 1.000), so they do not differ from soft labels. Reducing the data to 130 thousand documents changes the test macro-F1 by −0.002 (p 1.000), removing token dropout by +0.001 (p 1.000), and an MLP head that cannot be folded by +0.009 (p 0.281).

**Makna:**
Hard label dari teacher mengubah macro-F1 uji sebesar +0.005 (p 0.433) dan skor development sebesar -0.006 (p 1.000), jadi tidak berbeda dari soft label. Mengurangi data menjadi 130 ribu dokumen mengubah macro-F1 uji sebesar -0.002 (p 1.000), menghapus token dropout +0.001 (p 1.000), dan kepala MLP yang tidak bisa dilipat +0.009 (p 0.281).

**Locked:** angka +0.005, 0.433, 0.006, 1.000, 130, 0.002, +0.001, +0.009, 0.281 · simbol p, −

**Paraphrase:**
(tulis di sini)

---

### HAS-14 · Ablation, Temperature, and Ensembling (RQ3) · main.tex baris 516

**Current:**
Fig. 4 shows the effect of temperature. Compared with τ=1, the common choice τ=2 [9] lowers the held-out agreement by −0.007 (p < 0.001). Its changes on the test macro-F1 (−0.005, p 0.059) and on the development set (−0.006, p 0.705) are not significant. With τ=4 all three metrics drop, and on the test set the drop is −0.014 (p < 0.001). None of the four exploratory variants improves the test set: bigram hashing +0.002, idf pooling −0.001, confidence weights +0.000, and cross-entropy on GPT-4o labels −0.004.

**Makna:**
Gambar 4 menunjukkan efek temperature. Dibanding tau=1, pilihan umum tau=2 [9] menurunkan held-out agreement sebesar -0.007 (p < 0.001); perubahannya di macro-F1 uji (-0.005, p 0.059) dan di development set (-0.006, p 0.705) tidak signifikan. Dengan tau=4, ketiga metrik turun, dan di test set penurunannya -0.014 (p < 0.001). Tidak satu pun dari empat varian eksplorasi memperbaiki test set: bigram hashing +0.002, idf pooling -0.001, bobot confidence +0.000, dan cross-entropy pada label GPT-4o -0.004.

**Locked:** angka 0.007, 0.001, 0.005, 0.059, 0.006, 0.705, 0.014, +0.002, +0.000, 0.004 · sitasi [9] Hinton et al. 2015 · rujukan Fig. 4 · simbol τ=1, τ=2, −, p, <, τ=4

**Paraphrase:**
(tulis di sini)

---

### HAS-15 · Ablation, Temperature, and Ensembling (RQ3) · main.tex baris 518

**Current:**
Growing the ensemble from one to three seeds moves the test macro-F1 from 0.542 to 0.544 and the held-out agreement from 0.808 to 0.810 (Fig. 5). Ten seeds give 0.546. The gain is small, but it costs nothing at inference time because every ensemble size is still one table.

**Makna:**
Memperbesar ensemble dari satu ke tiga seed menggeser macro-F1 uji dari 0.542 ke 0.544 dan held-out agreement dari 0.808 ke 0.810 (Gambar 5); sepuluh seed memberi 0.546. Kenaikannya kecil, tetapi tanpa biaya saat inference karena berapa pun ukuran ensemble-nya tetap satu tabel.

**Locked:** angka 0.542, 0.544, 0.808, 0.810, 0.546 · rujukan Fig. 5

**Paraphrase:**
(tulis di sini)

[GAMBAR 4 — placeholder] Efek temperature distilasi. Tiap titik adalah satu seed; garis menghubungkan rata-ratanya.

[GAMBAR 5 — placeholder] Efek ukuran ensemble tabel (rata-rata ± standar deviasi antar-kombinasi seed).

[TABEL X — placeholder] Efisiensi CPU (P-core di-pin; sampel dari unified test set; 1c/4c = 1/4 core).

---

### HAS-16 · CPU Efficiency (RQ1) · main.tex baris 558

**Current:**
Table X and Figs. 6 and 7 show the efficiency of all models. In PyTorch fp32 the teacher needs 2,971 ms per article on one core, with a throughput of 0.33 articles per second. Its best CPU configuration, ONNX int8, needs 1,239 ms with a throughput of 0.77 articles per second. Classifying one million articles per day would take 15.1 cores (ONNX int8) to 34.8 cores (PyTorch) for this step alone.

**Makna:**
Subbagian: efisiensi CPU (RQ1). Tabel X serta Gambar 6 dan Gambar 7 menunjukkan efisiensi semua model. Dengan PyTorch fp32, teacher butuh 2,971 ms per artikel di satu core, dengan throughput 0.33 artikel per detik. Konfigurasi CPU terbaiknya, ONNX int8, butuh 1,239 ms dengan throughput 0.77 artikel per detik. Mengklasifikasi satu juta artikel per hari akan butuh 15.1 core (ONNX int8) sampai 34.8 core (PyTorch) hanya untuk langkah ini.

**Locked:** angka 2,971 ms, 0.33, 1,239 ms, 0.77, 15.1, 34.8 · rujukan Table X, Fig. 6, Fig. 7

**Paraphrase:**
(tulis di sini)

---

### HAS-17 · CPU Efficiency (RQ1) · main.tex baris 560

**Current:**
StaticKD needs 2.03 ms per article and processes 473 articles per second per core. This is about 616 times faster than the ONNX int8 teacher, 1,420 times faster than the PyTorch teacher, and 29 times faster than MiniLM-L6. Its file is 16 times smaller than the ONNX int8 teacher, and one million articles per day need 0.02 cores. Its model RAM is 501 MB, mostly for the tokenizer structure. The compact variant reduces it to 277 MB. With one thread, tokenization (579 tokens per article on average) takes 98% of the prediction time, while table pooling alone runs at 28,755 articles per second. The speed limit of StaticKD is thus set by the tokenizer implementation.

**Makna:**
StaticKD butuh 2.03 ms per artikel dan memproses 473 artikel per detik per core: sekitar 616 kali lebih cepat daripada teacher ONNX int8, 1,420 kali lebih cepat daripada teacher PyTorch, dan 29 kali lebih cepat daripada MiniLM-L6. Ukuran file-nya 16 kali lebih kecil daripada teacher ONNX int8, dan satu juta artikel per hari hanya butuh 0.02 core. RAM modelnya 501 MB, sebagian besar untuk struktur tokenizer; varian compact menurunkannya ke 277 MB. Dengan satu thread, tokenisasi (rata-rata 579 token per artikel) memakan 98% waktu prediksi, sementara pooling tabel saja berjalan 28,755 artikel per detik. Jadi batas kecepatan StaticKD ditentukan oleh implementasi tokenizer.

**Locked:** angka 2.03 ms, 473, 616, 1,420, 29, 16, 0.02, 501 MB, 277 MB, 579, 98%, 28,755

**Paraphrase:**
(tulis di sini)

---

### HAS-18 · CPU Efficiency (RQ1) · main.tex baris 562

**Current:**
StaticKD is not the fastest model in the table. The quantized fastText + KD is slightly faster (532 articles per second), smaller (18 MB), and uses less RAM (300 MB), but its test macro-F1 is 0.435 against 0.545 for StaticKD. Among the models with comparable latency, namely StaticKD, its compact variant, fastText, and TF-IDF + SVM, StaticKD is the most accurate (Fig. 6).

**Makna:**
StaticKD bukan model tercepat di tabel. fastText + KD terkuantisasi sedikit lebih cepat (532 artikel per detik), lebih kecil (18 MB), dan memakai RAM lebih sedikit (300 MB), tetapi macro-F1 ujinya 0.435 dibanding 0.545 untuk StaticKD. Di antara model dengan latency sebanding (StaticKD, varian compact-nya, fastText, dan TF-IDF + SVM), StaticKD paling akurat (Gambar 6).

**Locked:** angka 532, 18 MB, 300 MB, 0.435, 0.545 · rujukan Fig. 6

**Paraphrase:**
(tulis di sini)

[GAMBAR 6 — placeholder] Macro-F1 uji di unified test set terhadap throughput CPU satu core (skala log).

[GAMBAR 7 — placeholder] Latency per artikel, RAM model, dan ukuran file (skala log).

[TABEL XI — placeholder] Faktor-faktor di balik kinerja per bahasa.

---

### HAS-19 · Analysis by Language and Label (RQ4) · main.tex baris 600

**Current:**
The median per-language accuracy gap between StaticKD and the teacher is −0.024 (Fig. 8). On the held-out set (53 languages with at least 20 documents), the agreement of StaticKD correlates with the log amount of distillation data (ρ=0.31, p 0.022) and with teacher confidence (ρ=0.71, p < 0.001), but does not differ across scripts (Kruskal–Wallis H=10.37, p 0.065; Table XI). On languages outside XLM-R pre-training that are also absent from the distillation data (ig, lg, ln, pcm, rn, sn, ti, yo), StaticKD reaches a mean accuracy of only 0.407, while the teacher reaches 0.643.

**Makna:**
Subbagian: analisis per bahasa dan label (RQ4). Median selisih akurasi per bahasa antara StaticKD dan teacher adalah -0.024 (Gambar 8). Di held-out set (53 bahasa dengan minimal 20 dokumen), agreement StaticKD berkorelasi dengan log jumlah data distilasi (rho = 0.31, p 0.022) dan dengan confidence teacher (rho = 0.71, p < 0.001), tetapi tidak berbeda antaraksara (Kruskal-Wallis H = 10.37, p 0.065; Tabel XI). Pada bahasa di luar pre-training XLM-R yang juga tidak ada di data distilasi (ig, lg, ln, pcm, rn, sn, ti, yo), rata-rata akurasi StaticKD hanya 0.407, sedangkan teacher 0.643.

**Locked:** angka 0.024, 53, 20, 0.022, 0.001, 0.065, 0.407, 0.643 · rujukan Fig. 8, Table XI · simbol −, ρ=0.31, p, ρ=0.71, <, H=10.37

**Paraphrase:**
(tulis di sini)

---

### HAS-20 · Analysis by Language and Label (RQ4) · main.tex baris 602

**Current:**
In 4 languages (bn, ml, or, pa), fewer than half of the token occurrences belong to tokens seen at least 10 times in the distillation data (at most 32%). StaticKD still reaches a mean accuracy of 0.624 in these languages (teacher 0.738). The compact variant removes those tokens from the tokenizer, and its accuracy in these languages falls to 0.013. In the remaining languages the compact variant reaches 0.578 against 0.640 for the full StaticKD. Section V discusses the mechanism behind this pattern.

**Makna:**
Di 4 bahasa (bn, ml, or, pa), kurang dari separuh kemunculan token berasal dari token yang muncul minimal 10 kali di data distilasi (paling banyak 32%). StaticKD tetap mencapai rata-rata akurasi 0.624 di bahasa-bahasa ini (teacher 0.738). Varian compact membuang token-token itu dari tokenizer, dan akurasinya di bahasa ini jatuh ke 0.013. Di bahasa lainnya, varian compact mencapai 0.578 dibanding 0.640 untuk StaticKD penuh. Bagian V membahas mekanisme di balik pola ini.

**Locked:** angka 4, 10, 32%, 0.624, 0.738, 0.013, 0.578, 0.640 · rujukan Section V

**Paraphrase:**
(tulis di sini)

[GAMBAR 8 — placeholder] Kiri: akurasi StaticKD terhadap akurasi teacher per bahasa uji (warna = data distilasi bahasa, ukuran = jumlah artikel). Kanan: held-out agreement per bahasa terhadap jumlah data distilasi, untuk StaticKD dan fastText + KD.

---

### HAS-21 · Analysis by Language and Label (RQ4) · main.tex baris 611

**Current:**
Per label (Fig. 9), the hardest labels for StaticKD are society (0.16), disaster (0.29), human int. (0.32), and conflict (0.37), and the easiest are sport (0.88), weather (0.84), and education (0.75). The largest gaps to the teacher occur for environment (−0.17), religion (−0.15), and arts (−0.08), and the most frequent confusions are society → human int. (20%); human int. → arts (20%); society → politics (20%).

**Makna:**
Per label (Gambar 9), label tersulit bagi StaticKD adalah society (0.16), disaster (0.29), human interest (0.32), dan conflict (0.37); yang termudah sport (0.88), weather (0.84), dan education (0.75). Selisih terbesar terhadap teacher ada pada environment (-0.17), religion (-0.15), dan arts (-0.08), dan kebingungan paling sering adalah society -> human interest (20%), human interest -> arts (20%), dan society -> politics (20%).

**Catatan:** Catatan review: di naskah tertulis singkatan ad hoc 'human int.'; tulis lengkap 'human interest'.

**Locked:** angka 0.16, 0.29, 0.32, 0.37, 0.88, 0.84, 0.75, 0.17, 0.15, 0.08, 20% · rujukan Fig. 9 · simbol −, →

**Paraphrase:**
(tulis di sini)

[GAMBAR 9 — placeholder] F1 per label di test set (kiri) dan confusion matrix StaticKD yang dinormalisasi per label referensi (kanan).

[TABEL XII — placeholder] Cascade StaticKD ke teacher ONNX int8 di test set (throughput efektif satu core).

---

### HAS-22 · Calibration and Cascade Mode · main.tex baris 642

**Current:**
The ECE of StaticKD is 0.041 on the development set and 0.139 on the test set, lower than that of the teacher (0.090 and 0.245). Both models become overconfident on the test set, which lies outside the training distribution (Fig. 10). In cascade mode, StaticKD decides the articles on which its confidence is at least θ and sends the rest to the teacher (Table XII). With θ=0.8, StaticKD decides 60.3% of the articles with an accuracy of 0.813. The cascade reaches an overall accuracy of 0.703 and a macro-F1 of 0.602, against 0.699 and 0.600 for the teacher alone, at an effective throughput of 1.92 articles per second per core against 0.77. Sending about two fifths of the articles to the teacher is enough for the cascade to match the teacher accuracy.

**Makna:**
Subbagian: kalibrasi dan mode cascade. ECE StaticKD 0.041 di development set dan 0.139 di test set, lebih rendah daripada teacher (0.090 dan 0.245). Kedua model menjadi overconfident di test set, yang berada di luar distribusi training (Gambar 10). Dalam mode cascade, StaticKD memutuskan artikel yang confidence-nya minimal theta dan mengirim sisanya ke teacher (Tabel XII). Dengan theta = 0.8, StaticKD memutuskan 60.3% artikel dengan akurasi 0.813. Cascade mencapai akurasi keseluruhan 0.703 dan macro-F1 0.602, dibanding 0.699 dan 0.600 untuk teacher saja, dengan throughput efektif 1.92 artikel per detik per core dibanding 0.77. Mengirim sekitar dua per lima artikel ke teacher sudah cukup agar cascade menyamai akurasi teacher.

**Locked:** angka 0.041, 0.139, 0.090, 0.245, 60.3%, 0.813, 0.703, 0.602, 0.699, 0.600, 1.92, 0.77 · rujukan Fig. 10, Table XII · simbol θ, θ=0.8

**Paraphrase:**
(tulis di sini)

[GAMBAR 10 — placeholder] Reliability diagram StaticKD di development set dan test set.

## V. Discussion

---

### PEM-01 · main.tex baris 653

**Current:**
StaticKD keeps 91.1% of the teacher macro-F1 on the unified test set and 92.6% on the development set, at an inference cost two to three orders of magnitude lower (616× against ONNX int8 and 1,420× against PyTorch). This agrees with the finding that order-free representations are already competitive for topic classification [13]. Names of institutions, sports terms, and medical vocabulary carry most of the topic signal. The remaining gap concentrates in two places. The first is labels whose vocabulary overlaps with other topics. The largest gaps to the teacher occur for environment (−0.17), religion (−0.15), and arts (−0.08). The hardest labels for StaticKD, given here with their F1, are society (0.16), disaster (0.29), human int. (0.32), and conflict (0.37). They are also among the hardest for the teacher. Society and human interest are known to be difficult for human annotators as well [2]. The second is languages that are underrepresented in the distillation data. StaticKD reaches 97.5% of the teacher score in languages with sufficient distillation data but only 72.3% in languages without any.

**Makna:**
StaticKD mempertahankan 91.1% macro-F1 teacher di unified test set dan 92.6% di development set, dengan biaya inference dua sampai tiga orde besaran lebih rendah (616 kali terhadap ONNX int8 dan 1,420 kali terhadap PyTorch). Ini sejalan dengan temuan bahwa representasi tanpa urutan kata (order-free) sudah kompetitif untuk klasifikasi topik [13]. Nama institusi, istilah olahraga, dan kosakata medis membawa sebagian besar sinyal topik. Selisih yang tersisa terkumpul di dua tempat. Pertama, label yang kosakatanya tumpang tindih dengan topik lain: selisih terbesar terhadap teacher ada pada environment (-0.17), religion (-0.15), dan arts (-0.08); label tersulit bagi StaticKD (dengan F1-nya) adalah society (0.16), disaster (0.29), human interest (0.32), dan conflict (0.37), yang juga termasuk tersulit bagi teacher; society dan human interest diketahui sulit juga bagi anotator manusia [2]. Kedua, bahasa yang kurang terwakili di data distilasi: StaticKD mencapai 97.5% skor teacher di bahasa dengan data distilasi cukup, tetapi hanya 72.3% di bahasa tanpa data.

**Catatan:** Catatan review: kalimat 'nama institusi, istilah olahraga, dan kosakata medis' tidak didukung analisis atau sitasi; pertimbangkan dihapus atau diberi dasar. Tulis 'human interest', bukan 'human int.'.

**Locked:** angka 91.1%, 92.6%, 616, 1,420, 0.17, 0.15, 0.08, 0.16, 0.29, 0.32, 0.37, 97.5%, 72.3% · sitasi [13] Galke & Scherp 2022; [2] Kuzman & Ljubešić 2025 · simbol ×, −

**Paraphrase:**
(tulis di sini)

---

### PEM-02 · main.tex baris 655

**Current:**
With the same distillation data, MiniLM-L6 closes part of this gap (StaticKD − MiniLM −0.021, p 0.004), whereas DistilmBERT does not (+0.010, p 0.800). Both cost one to two orders of magnitude more than StaticKD. MiniLM comes from a distillation of XLM-R-large and DistilmBERT from mBERT, which suggests that the quality of the initial multilingual representation matters more than the number of layers.

**Makna:**
Dengan data distilasi yang sama, MiniLM-L6 menutup sebagian selisih ini (StaticKD dikurangi MiniLM -0.021, p 0.004), sedangkan DistilmBERT tidak (+0.010, p 0.800). Keduanya satu sampai dua orde besaran lebih mahal daripada StaticKD. MiniLM berasal dari distilasi XLM-R-large dan DistilmBERT dari mBERT, yang mengisyaratkan bahwa kualitas representasi multibahasa awal lebih menentukan daripada jumlah layer.

**Locked:** angka 0.021, 0.004, +0.010, 0.800 · simbol −, p

**Paraphrase:**
(tulis di sini)

---

### PEM-03 · main.tex baris 657

**Current:**
The ablations show where the knowledge of the student comes from (RQ3). The largest source is the multilingual distillation data labeled by the teacher. Without it the test macro-F1 drops by −0.051, consistent with Sparse Distillation [22], which also relies on a large transfer set. The cross-lingually aligned initialization comes next, and the compact variant gives the most direct evidence for it. In Indian languages that hardly occur in the distillation data (for example bn, pa, or), most tokens are never updated during training and keep their initial potion embeddings. StaticKD still classifies these languages with a mean accuracy of 0.624, yet the accuracy falls to 0.013 once the tokenizer pruning removes those tokens. Transfer to languages without data is therefore carried by embedding rows that were never trained but live in the same space as the trained rows, so the linear head learned on data-rich languages applies to them too.

**Makna:**
Ablasi menunjukkan dari mana pengetahuan student berasal (RQ3). Sumber terbesar adalah data distilasi multibahasa yang dilabeli teacher: tanpanya macro-F1 uji turun -0.051, sejalan dengan Sparse Distillation [22] yang juga bergantung pada transfer set besar. Berikutnya inisialisasi yang selaras lintas bahasa, dan varian compact memberi bukti paling langsung untuk ini. Di bahasa India yang hampir tidak ada di data distilasi (misalnya bn, pa, or), sebagian besar token tidak pernah di-update saat training dan tetap memakai embedding awal potion. StaticKD masih mengklasifikasi bahasa-bahasa ini dengan rata-rata akurasi 0.624, tetapi akurasinya jatuh ke 0.013 begitu pemangkasan tokenizer membuang token-token itu. Jadi transfer ke bahasa tanpa data dibawa oleh baris embedding yang tidak pernah dilatih tetapi berada di ruang yang sama dengan baris yang dilatih, sehingga kepala linear yang dipelajari dari bahasa kaya data juga berlaku untuknya.

**Catatan:** Catatan: klaim 'tidak pernah di-update' lebih luas dari yang diukur, yaitu token yang muncul < 10 kali di 4 bahasa bn, ml, or, pa; lihat N-05 butir 13.

**Locked:** angka 0.051, 0.624, 0.013 · sitasi [22] Ye et al. 2022 · simbol −

**Paraphrase:**
(tulis di sini)

---

### PEM-04 · main.tex baris 659

**Current:**
The frozen-embedding ablation shows the other side of the same mechanism. Fine-tuning moves the embeddings of tokens from languages in the distillation data and improves in-distribution fit. Freezing them lowers the development score by −0.035. Tokens of other languages stay in the original potion space, where the head that adapted to the moved space fits them less well. In languages with no or little distillation data, frozen embeddings are better by +0.037 and +0.059. This is representation drift caused by fine-tuning part of the vocabulary. The final recipe keeps fine-tuning because the recipe was selected on a development set in the training languages. For deployments that target languages without distillation data, frozen embeddings or a regularizer that keeps the embeddings close to potion are the safer choice, and both still fold into a table.

**Makna:**
Ablasi embedding beku memperlihatkan sisi lain dari mekanisme yang sama. Fine-tuning menggeser embedding token dari bahasa yang ada di data distilasi dan memperbaiki kecocokan in-distribution; membekukannya menurunkan skor development -0.035. Token bahasa lain tetap di ruang potion asli, di mana kepala yang sudah menyesuaikan diri dengan ruang yang bergeser kurang cocok untuknya. Di bahasa tanpa data atau dengan data distilasi sedikit, embedding beku lebih baik +0.037 dan +0.059. Ini adalah representation drift akibat fine-tuning sebagian vocabulary. Resep final tetap memakai fine-tuning karena resep dipilih pada development set dalam bahasa training. Untuk deployment yang menyasar bahasa tanpa data distilasi, embedding beku atau regularizer yang menjaga embedding dekat dengan potion adalah pilihan yang lebih aman, dan keduanya tetap bisa dilipat ke tabel.

**Locked:** angka 0.035, +0.037, +0.059 · simbol −

**Paraphrase:**
(tulis di sini)

---

### PEM-05 · main.tex baris 661

**Current:**
Soft labels, by contrast, add little or nothing. Hard teacher labels do not differ significantly from soft labels at τ=1 (development −0.006, test +0.005). Raising the temperature to τ=2 lowers the held-out agreement without a significant change on the test set, and τ=4 lowers every metric. This pattern fits the capacity-gap literature [24], [27]. A plausible explanation is that one linear layer over averaged embeddings cannot represent the inter-class structure in the tail of the teacher distribution (dark knowledge) [26]. A sharp distribution (τ=1) or hard labels give an attainable target, whereas a high τ asks the student to imitate a distribution shape beyond its capacity. For very-low-capacity students, effort is better spent on the coverage of the transfer set (languages, sites, topics) and on aligned initialization than on shaping the soft labels.

**Makna:**
Sebaliknya, soft label hampir tidak menambah apa-apa. Hard label dari teacher tidak berbeda signifikan dari soft label pada tau=1 (development -0.006, uji +0.005). Menaikkan temperature ke tau=2 menurunkan held-out agreement tanpa perubahan signifikan di test set, dan tau=4 menurunkan semua metrik. Pola ini sejalan dengan literatur capacity gap [24] [27]. Penjelasan yang masuk akal: satu layer linear di atas embedding rata-rata tidak mampu merepresentasikan struktur antarkelas di ekor distribusi teacher (dark knowledge) [26]. Distribusi yang tajam (tau=1) atau hard label memberi target yang bisa dicapai, sedangkan tau tinggi meminta student meniru bentuk distribusi di luar kapasitasnya. Untuk student berkapasitas sangat rendah, usaha lebih baik dicurahkan pada cakupan transfer set (bahasa, situs, topik) dan inisialisasi yang selaras daripada membentuk soft label.

**Locked:** angka 0.006, +0.005 · sitasi [24] Zhang et al. 2025; [27] Wang et al. 2026; [26] Zhao et al. 2022 · simbol τ=1, −, τ=2, τ=4, τ

**Paraphrase:**
(tulis di sini)

---

### PEM-06 · main.tex baris 663

**Current:**
Per-language performance is driven mainly by how well a language is represented in the distillation and pre-training data, not by its script (Kruskal–Wallis p 0.065). The weakest languages are African languages outside XLM-R pre-training that are also missing from the distillation data (ig, lg, ln, pcm, rn, sn, ti, yo). The teacher still reaches an accuracy of 0.643 on them, perhaps thanks to subwords shared with other languages and to sentence context, while StaticKD reaches only 0.407. The cheapest fix needs no change of architecture. News in these languages can simply be added to the transfer set and labeled by the teacher. The data ablation and the correlation between agreement and the amount of data (ρ=0.31) support this step.

**Makna:**
Kinerja per bahasa terutama ditentukan oleh seberapa baik bahasa itu terwakili di data distilasi dan data pre-training, bukan oleh aksaranya (Kruskal-Wallis p 0.065). Bahasa terlemah adalah bahasa Afrika di luar pre-training XLM-R yang juga tidak ada di data distilasi (ig, lg, ln, pcm, rn, sn, ti, yo). Teacher masih mencapai akurasi 0.643 di bahasa-bahasa ini, mungkin berkat subword yang dibagi dengan bahasa lain dan konteks kalimat, sedangkan StaticKD hanya 0.407. Perbaikan termurah tidak butuh perubahan arsitektur: berita dalam bahasa-bahasa ini cukup ditambahkan ke transfer set dan dilabeli teacher. Ablasi data dan korelasi antara agreement dan jumlah data (rho = 0.31) mendukung langkah ini.

**Locked:** angka 0.065, 0.643, 0.407 · simbol p, ρ=0.31

**Paraphrase:**
(tulis di sini)

---

### PEM-07 · main.tex baris 665

**Current:**
Some variants move agreement with the teacher and accuracy on editorial labels in opposite directions. Frozen embeddings lower the held-out agreement (−0.020) but raise the test macro-F1 (+0.014). Cutting the data to 130 thousand documents lowers the agreement (−0.009) without a significant change on the test set (−0.002). In line with [28], students should be selected with labels that do not come from the teacher and with test data outside the training distribution, not with fidelity alone.

**Makna:**
Beberapa varian menggeser agreement dengan teacher dan akurasi pada label editorial ke arah berlawanan. Embedding beku menurunkan held-out agreement (-0.020) tetapi menaikkan macro-F1 uji (+0.014). Memangkas data menjadi 130 ribu dokumen menurunkan agreement (-0.009) tanpa perubahan signifikan di test set (-0.002). Sejalan dengan [28], student sebaiknya dipilih dengan label yang tidak berasal dari teacher dan data uji di luar distribusi training, bukan hanya dengan fidelity.

**Locked:** angka 0.020, +0.014, 130, 0.009, 0.002 · sitasi [28] Stanton et al. 2021 · simbol −

**Paraphrase:**
(tulis di sini)

---

### PEM-08 · main.tex baris 667

**Current:**
Test scores should be read according to the nature of the labels. Two thirds of the labels are editorial topics, the place where the newsroom put an article, translated to IPTC. An article about a criminal case involving an athlete can rightly sit in a sports section while a model labels it crime. Both answers are plausible under IPTC, which allows several topics per article. Absolute scores on editorial labels thus measure agreement with the categorization of publishers. Comparisons between models remain fair because all models are scored against the same labels, and the model ranking does not change on the subset with capped site counts.

**Makna:**
Skor uji sebaiknya dibaca sesuai sifat labelnya. Dua pertiga label adalah topik editorial, yaitu tempat redaksi menaruh artikel, diterjemahkan ke IPTC. Artikel tentang kasus kriminal yang melibatkan atlet bisa saja dengan benar ditaruh di rubrik olahraga, sementara model melabelinya crime; kedua jawaban masuk akal menurut IPTC, yang memperbolehkan beberapa topik per artikel. Jadi skor absolut pada label editorial mengukur kesesuaian dengan kategorisasi penerbit. Perbandingan antarmodel tetap adil karena semua model dinilai terhadap label yang sama, dan urutan model tidak berubah pada subset dengan jumlah per situs dibatasi.

**Locked:** —

**Paraphrase:**
(tulis di sini)

---

### PEM-09 · main.tex baris 669

**Current:**
Compared with Sparse Distillation [22], StaticKD replaces an n-gram table trained from scratch with pre-trained multilingual embeddings folded into 17 logits per token. The model becomes far smaller and transfers to languages that are missing from the transfer set. Inference is the recurring cost in a news pipeline, since every incoming article is classified, and one million articles per day need 15.1 cores with the ONNX int8 teacher against 0.02 with StaticKD (Section IV-D). Int8 quantization of the teacher only gives a constant-factor speed-up.

**Makna:**
Dibanding Sparse Distillation [22], StaticKD mengganti tabel n-gram yang dilatih dari nol dengan embedding multibahasa pra-latih yang dilipat menjadi 17 logit per token; modelnya jadi jauh lebih kecil dan bisa transfer ke bahasa yang tidak ada di transfer set. Inference adalah biaya berulang dalam pipeline berita karena setiap artikel yang masuk diklasifikasi; satu juta artikel per hari butuh 15.1 core dengan teacher ONNX int8 dibanding 0.02 dengan StaticKD (Bagian IV-D). Kuantisasi int8 pada teacher hanya memberi percepatan dengan faktor konstan.

**Locked:** angka 17, 15.1, 0.02 · sitasi [22] Ye et al. 2022 · rujukan Section IV-D

**Paraphrase:**
(tulis di sini)

---

### PEM-10 · main.tex baris 671

**Current:**
For a real-time pipeline on servers without a GPU, StaticKD processes one million articles per day with 0.02 cores, a 34 MB file, and numpy and tokenizers as its only dependencies. Its probabilities are better calibrated than those of the teacher (test ECE 0.139 against 0.245), which makes a cascade possible. With θ=0.8, the cascade accuracy (0.703) equals that of the teacher alone (0.699) at 1.92 articles per second per core, against 0.77 for the teacher alone. This mode suits batch jobs that need teacher-level accuracy. The compact variant lowers RAM but is only advisable when all target languages are covered by the distillation data.

**Makna:**
Untuk pipeline real-time di server tanpa GPU, StaticKD memproses satu juta artikel per hari dengan 0.02 core, file 34 MB, dan hanya bergantung pada numpy dan tokenizers. Probabilitasnya lebih terkalibrasi daripada teacher (ECE uji 0.139 vs 0.245), sehingga mode cascade bisa dipakai. Dengan theta = 0.8, akurasi cascade (0.703) setara teacher saja (0.699) pada 1.92 artikel per detik per core, dibanding 0.77 untuk teacher saja. Mode ini cocok untuk batch job yang butuh akurasi setingkat teacher. Varian compact menurunkan RAM, tetapi hanya disarankan bila semua bahasa target tercakup data distilasi.

**Locked:** angka 0.02, 34 MB, 0.139, 0.245, 0.703, 0.699, 1.92, 0.77 · simbol θ=0.8 · nama numpy, tokenizers

**Paraphrase:**
(tulis di sini)

## VI. Limitations

---

### BTS-01 · main.tex baris 675

**Current:**
Of the test articles, 47.5% carry JSON-LD section labels and 20.4% publisher categories. These labels are editorial topics. Section conventions differ between publishers, and one section reflects one editorial view of an article that may cover several topics. The mapping table is conservative, validated against the IPTC tree, and reviewed by the first author, but the articles were not re-annotated by humans. Absolute scores on these labels therefore measure agreement with the categorization of publishers, especially for labels with unclear boundaries (society, human interest, lifestyle).

**Makna:**
Dari artikel uji, 47.5% berlabel rubrik JSON-LD dan 20.4% kategori penerbit. Label-label ini adalah topik editorial. Konvensi rubrik berbeda antarpenerbit, dan satu rubrik hanya mencerminkan satu sudut pandang redaksi atas artikel yang bisa mencakup beberapa topik. Tabel pemetaannya konservatif, divalidasi terhadap pohon IPTC, dan ditinjau penulis pertama, tetapi artikelnya tidak dianotasi ulang oleh manusia. Karena itu skor absolut pada label ini mengukur kesesuaian dengan kategorisasi penerbit, terutama untuk label yang batasnya tidak jelas (society, human interest, lifestyle).

**Locked:** angka 47.5%, 20.4%

**Paraphrase:**
(tulis di sini)

---

### BTS-02 · main.tex baris 677

**Current:**
MasakhaNEWS labels were assigned by human annotators under their own scheme and then mapped to IPTC. Only MN-DS is annotated directly with IPTC topics, and it is English only. The multilingual IPTC test set of Kuzman and Ljubešić [2] is not public, so it was not available for this study.

**Makna:**
Label MasakhaNEWS diberikan anotator manusia dengan skema mereka sendiri, lalu dipetakan ke IPTC. Hanya MN-DS yang dianotasi langsung dengan topik IPTC, dan hanya bahasa Inggris. Test set IPTC multibahasa milik Kuzman dan Ljubešić [2] tidak publik, sehingga tidak tersedia untuk studi ini.

**Locked:** sitasi [2] Kuzman & Ljubešić 2025

**Paraphrase:**
(tulis di sini)

---

### BTS-03 · main.tex baris 679

**Current:**
Label coverage per language is uneven, with a median of 7 labels per language, and the African languages cover only four or five broad topics. Macro-F1 is computed over the labels present, so per-language scores are not directly comparable between languages with different label coverage. Publisher diversity is also limited for some languages: 27 languages come from practically one publisher, namely BBC/VOA for the African languages, L3Cube for the Indian languages, and a single site for several CC-NEWS languages. Claims for these languages are limited to those publishers, and the site-cluster bootstrap widens their confidence intervals accordingly.

**Makna:**
Cakupan label per bahasa tidak merata, dengan median 7 label per bahasa, dan bahasa Afrika hanya mencakup empat atau lima topik luas. Macro-F1 dihitung atas label yang ada, jadi skor per bahasa tidak bisa dibandingkan langsung antarbahasa dengan cakupan label berbeda. Keragaman penerbit juga terbatas untuk sebagian bahasa: 27 bahasa praktis hanya dari satu penerbit, yaitu BBC/VOA untuk bahasa Afrika, L3Cube untuk bahasa India, dan satu situs untuk beberapa bahasa CC-NEWS. Klaim untuk bahasa-bahasa ini terbatas pada penerbit tersebut, dan site-cluster bootstrap melebarkan confidence interval-nya sesuai keadaan itu.

**Locked:** angka 7, 27

**Paraphrase:**
(tulis di sini)

---

### BTS-04 · main.tex baris 681

**Current:**
The teacher bounds the student from above, and its errors and biases are copied. The fidelity set measures similarity to the teacher, not correctness.

**Makna:**
Teacher menjadi batas atas bagi student, dan kesalahan serta biasnya ikut tersalin. Fidelity set mengukur kemiripan dengan teacher, bukan kebenaran.

**Locked:** —

**Paraphrase:**
(tulis di sini)

---

### BTS-05 · main.tex baris 683

**Current:**
The hyperparameters of fastText and TF-IDF + SVM were set once without a search, whereas the StaticKD recipe was selected from 72 runs on the development set. This unequal tuning budget may widen the accuracy gap to these two baselines. The classifier trainer of the Model2Vec library was not run as a baseline, and only its components were tested separately as ablations. MiniLM-L6 and DistilmBERT were trained and evaluated with at most 256 tokens and therefore see less than half of most test articles, so their accuracy may be underestimated.

**Makna:**
Hyperparameter fastText dan TF-IDF + SVM ditetapkan sekali tanpa pencarian, sedangkan resep StaticKD dipilih dari 72 run pada development set. Budget tuning yang tidak setara ini bisa memperlebar selisih akurasi terhadap kedua baseline itu. Trainer classifier dari library Model2Vec tidak dijalankan sebagai baseline; hanya komponennya yang diuji terpisah sebagai ablasi. MiniLM-L6 dan DistilmBERT dilatih dan dievaluasi dengan paling banyak 256 token, sehingga hanya melihat kurang dari separuh sebagian besar artikel uji, dan akurasinya mungkin terlalu rendah (underestimated).

**Locked:** angka 72, 256

**Paraphrase:**
(tulis di sini)

---

### BTS-06 · main.tex baris 685

**Current:**
The benchmark ran on one machine in one session, with 30 latency documents for the teacher and the transformers. Absolute numbers on a server will differ and the latency sample of the heavy models is small, but ratios between models measured under the same protocol remain informative.

**Makna:**
Benchmark dijalankan di satu mesin dalam satu sesi, dengan 30 dokumen latency untuk teacher dan transformer. Angka absolut di server akan berbeda, dan sampel latency model berat kecil, tetapi rasio antarmodel yang diukur dengan protokol yang sama tetap informatif.

**Locked:** angka 30

**Paraphrase:**
(tulis di sini)

---

### BTS-07 · main.tex baris 687

**Current:**
StaticKD is much less accurate on African languages that are absent from both the distillation data and pre-training (Section V). Used as a topic filter, it could misclassify or drop news in these languages more often. Before deployment, accuracy should be monitored per language, and low-confidence articles should be routed to the teacher through the cascade mode.

**Makna:**
StaticKD jauh kurang akurat pada bahasa Afrika yang tidak ada di data distilasi maupun pre-training (Bagian V). Bila dipakai sebagai filter topik, StaticKD bisa lebih sering salah mengklasifikasi atau membuang berita dalam bahasa-bahasa ini. Sebelum deployment, akurasi perlu dipantau per bahasa, dan artikel dengan confidence rendah perlu dialihkan ke teacher lewat mode cascade.

**Locked:** rujukan Section V

**Paraphrase:**
(tulis di sini)

---

### BTS-08 · main.tex baris 689

**Current:**
The study covers only top-level IPTC labels and single-label classification. Deeper levels and multi-topic articles remain untested.

**Makna:**
Studi ini hanya mencakup label IPTC level atas dan klasifikasi single-label. Level yang lebih dalam dan artikel multi-topik belum diuji.

**Locked:** —

**Paraphrase:**
(tulis di sini)

## VII. Conclusion

---

### KES-01 · main.tex baris 693

**Current:**
This study measured the CPU cost of the open multilingual IPTC classifier and showed that it can be cut by two to three orders of magnitude while keeping most of its accuracy. StaticKD distills the XLM-R-large teacher into a table of 17 logits per token, initialized from multilingual static embeddings and trained on 285,999 teacher-labeled articles in 69 languages. On a unified test set of 12,744 articles in 47 languages, with labels that do not come from the teacher, StaticKD reaches a macro-F1 of 0.545, or 91.1% of the teacher. It processes 473 articles per second per CPU core, about 616 times the speed of the ONNX int8 teacher, with a 34 MB model that needs no deep learning library.

**Makna:**
Studi ini mengukur biaya CPU classifier IPTC multibahasa terbuka dan menunjukkan biaya itu bisa dipangkas dua sampai tiga orde besaran sambil mempertahankan sebagian besar akurasinya. StaticKD men-distill teacher XLM-R-large menjadi tabel 17 logit per token, diinisialisasi dari static embedding multibahasa dan dilatih pada 285,999 artikel berlabel teacher dalam 69 bahasa. Di unified test set berisi 12,744 artikel dalam 47 bahasa, dengan label yang tidak berasal dari teacher, StaticKD mencapai macro-F1 0.545, atau 91.1% dari teacher. StaticKD memproses 473 artikel per detik per core CPU, sekitar 616 kali kecepatan teacher ONNX int8, dengan model 34 MB yang tidak butuh library deep learning.

**Locked:** angka 17, 285,999, 69, 12,744, 47, 0.545, 91.1%, 473, 616, 34 MB

**Paraphrase:**
(tulis di sini)

---

### KES-02 · main.tex baris 695

**Current:**
The accuracy of this very-low-capacity student comes mainly from the multilingual transfer set and from cross-lingually aligned embeddings. Soft labels are no better than hard labels, τ=4 lowers every metric, and an ensemble of seeds folds into one table at no cost. Fine-tuning the embeddings raises accuracy in languages present in the distillation data but lowers it in languages without distillation data, because transfer to those languages is carried by pre-trained embeddings that are never updated. The test set, with editorial labels mapped onto the IPTC tree, contamination control, and site-cluster bootstrap, can be reused to evaluate other IPTC classifiers.

**Makna:**
Akurasi student berkapasitas sangat rendah ini terutama berasal dari transfer set multibahasa dan embedding yang selaras lintas bahasa. Soft label tidak lebih baik daripada hard label, tau=4 menurunkan semua metrik, dan ensemble seed bisa dilipat ke satu tabel tanpa biaya. Fine-tuning embedding menaikkan akurasi di bahasa yang ada di data distilasi tetapi menurunkannya di bahasa tanpa data distilasi, karena transfer ke bahasa itu dibawa oleh embedding pra-latih yang tidak pernah di-update. Test set ini, dengan label editorial yang dipetakan ke pohon IPTC, kontrol kontaminasi, dan site-cluster bootstrap, bisa dipakai ulang untuk mengevaluasi classifier IPTC lain.

**Catatan:** Catatan: klaim 'tidak pernah di-update' sama dengan PEN-11/PEM-03, menunggu dipersempit.

**Locked:** simbol τ=4

**Paraphrase:**
(tulis di sini)

---

### KES-03 · main.tex baris 697

**Current:**
Several directions follow from these results. The transfer set should be extended to the weak languages, especially the African languages outside XLM-R pre-training. A fine-tuning regularizer that keeps embeddings close to the pre-trained space could give in-distribution accuracy and cross-lingual transfer at the same time. Tokenization takes 98% of the inference time, so a faster tokenizer, for instance a direct word-to-table lookup, is the next source of speed. Any pruning of the vocabulary has to keep cross-lingual tokens. Cheap non-linear heads with quantized embeddings, hierarchical classification into IPTC level 2, multi-label classification, and human IPTC annotation of a multilingual sample of the test set remain open.

**Makna:**
Beberapa arah lanjutan: (a) transfer set diperluas ke bahasa yang lemah, terutama bahasa Afrika di luar pre-training XLM-R; (b) regularizer fine-tuning yang menjaga embedding dekat ruang pra-latih bisa memberi akurasi in-distribution sekaligus transfer lintas bahasa; (c) tokenisasi memakan 98% waktu inference, jadi tokenizer yang lebih cepat (misalnya lookup langsung kata ke tabel) adalah sumber kecepatan berikutnya; (d) pemangkasan vocabulary apa pun harus mempertahankan token lintas bahasa; (e) kepala non-linear murah dengan embedding terkuantisasi, klasifikasi hierarkis ke IPTC level 2, klasifikasi multi-label, dan anotasi IPTC manusia untuk sampel multibahasa test set masih terbuka.

**Locked:** angka 98%, 2

**Paraphrase:**
(tulis di sini)

## Code, Model, and Data Availability

---

### KTS-01 · main.tex baris 700

**Current:**
The code for data collection, test-set construction, teacher labeling, training, evaluation, and benchmarking, together with a notebook that reproduces every number, table, and figure of this paper, is available at [TODO: GitHub URL and Zenodo DOI]. The StaticKD models are available at [TODO: Hugging Face URL]. The test set is released as a manifest of article URLs or IDs, labels, label origins, and text hashes, with a reconstruction script. Article texts are not redistributed because of publisher copyright and the licenses of the source datasets. The section-to-IPTC mapping table is released with the code. EMMediaTopic is available from the CLARIN.SI repository [35]. Table XIII lists the licenses of the assets used and released in this study.

**Makna:**
Kode untuk pengumpulan data, konstruksi test set, pelabelan teacher, training, evaluasi, dan benchmark, beserta notebook yang mereproduksi setiap angka, tabel, dan gambar di paper ini, tersedia di [TODO: URL GitHub dan DOI Zenodo]. Model StaticKD tersedia di [TODO: URL Hugging Face]. Test set dirilis sebagai manifest berisi URL atau ID artikel, label, asal label, dan hash teks, beserta skrip rekonstruksi. Teks artikel tidak didistribusikan ulang karena hak cipta penerbit dan lisensi dataset sumber. Tabel pemetaan rubrik ke IPTC dirilis bersama kode. EMMediaTopic tersedia di repositori CLARIN.SI [35]. Tabel XIII mencantumkan lisensi aset yang dipakai dan dirilis dalam studi ini.

**Locked:** sitasi [35] Kuzman & Ljubešić 2025 · rujukan Table XIII

**Paraphrase:**
(tulis di sini)

[TABEL XIII — placeholder] Lisensi aset yang dipakai dan dirilis.

## Acknowledgment

---

### UCP-01 · main.tex baris 729

**Current:**
The authors thank the Department of Informatics Engineering, Sepuluh Nopember Institute of Technology, for its support of this work. The authors used Claude (Anthropic) to assist with writing the experiment code, with data processing (including the first draft of the section-to-IPTC mapping table, all of which the authors reviewed), and with reviewing and minor editing of the manuscript. The manuscript was written by the authors, who take full responsibility for its content and claims.

**Makna:**
Ucapan terima kasih kepada Departemen Teknik Informatika, Institut Teknologi Sepuluh Nopember, atas dukungannya. Pernyataan penggunaan AI: penulis memakai Claude (Anthropic) untuk membantu penulisan kode eksperimen, pengolahan data (termasuk draf pertama tabel pemetaan rubrik ke IPTC, yang seluruhnya ditinjau penulis), serta review dan penyuntingan minor naskah. Naskah ditulis oleh penulis, yang bertanggung jawab penuh atas isi dan klaimnya.

**Catatan:** Catatan: kalimat 'all of which the authors reviewed' perlu disamakan menjadi penulis pertama; pernyataan AI dikunci penulis, sesuaikan saat menulis ulang dan tambahkan versi model Claude saat submit.

**Locked:** —

**Paraphrase:**
(tulis di sini)
