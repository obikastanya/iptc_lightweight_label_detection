# Lightweight Multi-Language IPTC Label News Classification with Knowledge Distillation & LoRA

1st Obi Kastanya
Dept. of Informatics Engineering
Institut Teknologi Sepuluh Nopember
Surabaya, Indonesia
6025252015@student.its.ac.id

2nd Ananta Dwi Prayoga Alwy
Dept. of Informatics Engineering
Institut Teknologi Sepuluh Nopember
Surabaya, Indonesia
[NRP]@student.its.ac.id

*Abstract*—

*Index Terms*—IPTC, Knowledge Distillation, LoRA, Multilingual Text Classification, Model Compression

## I. INTRODUCTION

Artikel berita adalah salah satu sumber informasi utama dalam industri data analitik karena sifatnya yang faktual dan up-to-date. Agar hasil analisis tetap terarah, artikel perlu dikelompokkan ke kategori terstandar, kemudian dipilah hanya topik yang relevan. International Press Telecommunications Council (IPTC) Media Topic adalah standar pengkategorian artikel berita yang teruji dan digunakan di dunia industri oleh beberapa situs berita besar seperti Reuters, Associated Press (AP), Agence France-Presse (AFP), dan Bloomberg [9].

Large language model (LLM) dapat digunakan untuk mengklasifikasikan artikel berita ke topik IPTC tanpa anotator manusia dengan akurasi yang tinggi, mendekati kesepakatan antar-anotator manusia [1]. Namun beban komputasi yang dihasilkan tidak praktis untuk diterapkan pada skala produksi [1]. Sebagai alternatif, Kuzman & Ljubešić [1] mengusulkan pendekatan teacher-student: LLM melakukan klasifikasi dan memberi label secara zero-shot, kemudian label tersebut digunakan untuk melatih model yang lebih kecil (XLM-RoBERTa-large) sebagai student melalui full fine-tuning. Pendekatan ini berhasil memangkas kebutuhan anotasi manual, namun belum menjawab dimensi efisiensi yang sama pentingnya: model yang dihasilkan tidak terkompresi, sehingga beban komputasi tetap menjadi kendala pada penerapan di lingkungan produksi.

Dua strategi kompresi model dapat digunakan untuk mengatasi masalah ini. Pertama, low-rank adaptation (LoRA) [2] dan teknik parameter-efficient fine-tuning (PEFT) lain terbukti mampu memangkas parameter terlatih hingga ratusan kali lipat dengan penurunan akurasi minimum pada klasifikasi teks multibahasa [3]. Namun Razuvayevskaya dkk. [3] mencatat bahwa untuk bahasa yang belum pernah dilihat model (kasus: bahasa Georgia), full fine-tuning justru menghasilkan akurasi lebih tinggi dibanding LoRA pada 8 dari 9 kombinasi subtugas dan skenario yang diuji — meski mereka sendiri mencatat kemungkinan faktor perancu (teks uji Georgia jauh lebih pendek dari bahasa lain, dan gold label untuk bahasa tersebut tidak dirilis oleh penyelenggara SemEval-2023 sehingga tidak dapat diverifikasi lebih lanjut). Strategi kedua adalah knowledge distillation (KD) [4], yaitu mentransfer pengetahuan lintas bahasa dan domain dari satu model ke model lain. Contoh penerapannya dilakukan oleh Piperno dkk. [5] dengan mentransfer pengetahuan dari model biomedis berbahasa Inggris ke model multibahasa.

Azimi dkk. [6] membuktikan bahwa KD dan LoRA dapat digabungkan: model full fine-tuning diambil sebagai teacher, lalu mengajarkan pengetahuannya ke model student yang sudah terintegrasi modul LoRA. Pendekatan ini menghasilkan model dengan performa 97-98% dari model teacher-nya, dengan penurunan ukuran parameter hingga 99%, penurunan penggunaan memori hingga 75%, dan penurunan latensi hingga sekitar 30%. Namun benchmark ini terbatas pada klasifikasi kalimat pendek berbahasa Inggris (benchmark GLUE) dengan pilihan kelas yang sedikit, belum diuji pada teks yang lebih panjang dan multibahasa dengan taxonomy label yang lebih kompleks seperti topik berita IPTC.

Berdasarkan penelitian-penelitian ini, penelitian ini mengadaptasi pipeline KD+LoRA ke konteks yang belum pernah divalidasi: klasifikasi topik berita multibahasa berskema IPTC Media Topic. Kontribusinya bukan pada orisinalitas teknik KD atau LoRA — keduanya sudah established dan telah dibuktikan Azimi dkk. [6] untuk klasifikasi monolingual — melainkan pada tiga hal:

1) Pembuktian empiris apakah pipeline KD+LoRA tetap bekerja pada domain dan struktur task yang berbeda (multibahasa, taxonomy IPTC) tanpa kehilangan akurasi signifikan dari model SOTA IPTC classifier [1].
2) Pengujian apakah kerapuhan PEFT pada bahasa *unseen*/low-resource yang dilaporkan pada literatur lain [3] juga muncul pada tugas klasifikasi topik berita, dan apakah distilasi dari teacher multibahasa dapat memitigasinya.
3) Penyediaan pengukuran efisiensi model yang terpadu — parameter terlatih, ukuran model, latensi inferensi — yang belum dilaporkan pada literatur IPTC classification, sekaligus baseline knowledge-distillation-saja yang absen pada karya Azimi dkk. [6], untuk mengisolasi kontribusi masing-masing komponen.

Untuk mengarahkan penelitian ini, dirumuskan pertanyaan penelitian berikut: **RQ1**: Seberapa jauh model IPTC classifier dapat dikompresi melalui kombinasi KD dan LoRA sebelum akurasi klasifikasi turun signifikan dari SOTA? **RQ2**: Apakah kerapuhan LoRA/PEFT pada bahasa unseen/low-resource yang dilaporkan pada tugas klasifikasi lain [3] juga muncul pada klasifikasi topik berita IPTC, dan dapatkah knowledge distillation memitigasinya? **RQ3**: Bagaimana profil trade-off akurasi-versus-efisiensi dari model KD+LoRA dapat divalidasi untuk skenario deployment nyata?

Pada tahap ini, penelitian menawarkan hasil sintesis literatur berikut dan evaluasi yang direncanakan:

1) Sintesis enam studi rujukan yang menunjukkan bahwa KD dan LoRA masing-masing telah terbukti efektif secara terpisah, namun belum pernah dikombinasikan dan diuji pada klasifikasi teks multibahasa dengan taxonomy berlabel banyak seperti IPTC (Bagian II).
2) Identifikasi gap penelitian yang dipetakan menjadi tiga area — efisiensi model untuk taxonomy IPTC multibahasa, keandalan model ringan pada bahasa low-resource, dan kebutuhan baseline pembanding yang adil — beserta cara mengisinya (Bagian II-E).
3) *Direncanakan, belum dikerjakan:* pipeline eksperimen KD+LoRA pada dataset IPTC multibahasa, dievaluasi terhadap baseline SOTA [1], baseline LoRA-saja, dan baseline KD-saja, dengan metrik akurasi (macro-F1/micro-F1) dan efisiensi (parameter, ukuran model, latensi). Hasil akan dilaporkan pada revisi berikutnya setelah eksperimen selesai.

Sisa dokumen disusun sebagai berikut. Bagian II meninjau klasifikasi topik berita IPTC, PEFT/LoRA untuk klasifikasi multibahasa, dan knowledge distillation lintas bahasa, kemudian mengidentifikasi gap penelitian. Bagian III akan menyajikan metode, dataset, dan eksperimen yang diusulkan. Bagian IV akan menyajikan hasil, diskusi, dan kesimpulan setelah eksperimen selesai.

## II. RELATED WORK

### A. Klasifikasi Topik Berita Berbasis IPTC

Kuzman & Ljubešić [1] mengusulkan kerangka teacher-student berbasis LLM untuk membangun classifier IPTC Media Topic multibahasa tanpa data anotasi manual. GPT-4o digunakan sebagai teacher dalam mode zero-shot untuk memberi label pada korpus berita berbahasa Kroasia, Slovenia, Yunani, dan Katalan (dataset EMMediaTopic 1.0, 17 label top-level IPTC), lalu label tersebut (hard-label) dipakai untuk fine-tuning penuh XLM-RoBERTa-large sebagai student. Model student pada 15.000 instans mencapai macro-F1 0,746 dan micro-F1 0,734, sedikit melampaui performa teacher (macro-F1 0,731). Studi ini juga menunjukkan bahwa satu model multibahasa berkinerja setara atau lebih baik daripada model monolingual per bahasa, dan transfer lintas bahasa zero-shot tetap kompetitif. Namun, seluruh proses fine-tuning dilakukan secara penuh — tidak ada penggunaan LoRA, adapter, atau teknik PEFT lain — dan efisiensi yang diklaim bersifat efisiensi anotasi data, bukan efisiensi ukuran model, parameter, memori, atau latensi inferensi.

### B. PEFT/LoRA untuk Klasifikasi Teks Multibahasa

Razuvayevskaya dkk. [3] membandingkan LoRA, bottleneck adapter (konfigurasi Pfeiffer), dan full fine-tuning (FFT) pada XLM-RoBERTa-large untuk tiga subtugas SemEval-2023 Task 3 (deteksi genre, framing, dan teknik persuasi) dalam skenario multibahasa (6 bahasa sumber, 9 bahasa uji termasuk 3 bahasa "unseen": Spanyol, Yunani, Georgia). PEFT mereduksi parameter terlatih 140–280x dan waktu pelatihan 32–44% dibanding FFT, tetapi untuk bahasa Georgia — satu-satunya bahasa yang konsisten menyukai FFT di seluruh subtugas dan skenario — FFT unggul pada 8 dari 9 kombinasi subtugas×skenario yang diuji, bertentangan dengan klaim sebelumnya bahwa adapter selalu unggul pada transfer lintas bahasa zero-shot. LoRA unggul pada teks pendek (persuasion, rata-rata 74 token), sedangkan FFT/adapter lebih baik pada dokumen panjang (genre, framing, >1.000 token).

Nwaiwu [7] melengkapi temuan ini dengan membandingkan LoRA, IA3, dan ReFT (varian selektif ala BitFit) pada DistilBERT dalam kondisi low-resource ekstrem (1.000 instans pelatihan). Full fine-tuning dapat mengalami *catastrophic failure* pada tugas klasifikasi biner (F1 turun ke level setara tebakan acak pada dataset Amazon Reviews) ketika kapasitas model jauh melebihi ukuran data — meski pada tugas 4 kelas (AG News), FFT hanya underperform secara moderat, tidak kolaps. LoRA konsisten memberikan F1 tertinggi pada kedua dataset, sementara ReFT memberikan efisiensi tertinggi (hanya ~3% parameter LoRA untuk ~98% performanya). Namun, studi ini sepenuhnya berbahasa Inggris tanpa elemen multibahasa, dan tidak menggunakan knowledge distillation sama sekali.

### C. Knowledge Distillation Lintas Bahasa

Piperno dkk. [5] mengusulkan distilasi berbasis sentence transformer untuk transfer pengetahuan domain biomedis dari teacher BERT berbahasa Inggris (BioBERT) ke student mBERT multibahasa (Spanyol, Prancis, Jerman), menggunakan Multiple Negatives Ranking Loss (MNRL) yang terbukti lebih robust terhadap noise hasil mesin penerjemah dibanding MSE (unggul pada 32 dari 36 konfigurasi uji). Manfaat distilasi paling terasa pada kondisi data target-language sangat terbatas (rata-rata peningkatan F1 +7,95% pada 10% data, menyusut menjadi +0,45% pada 100% data). Namun, baik teacher maupun student dalam studi ini sama-sama berarsitektur BERT-base penuh (~110M parameter) — ini adalah transfer pengetahuan lintas bahasa/domain, bukan kompresi model, dan sama sekali tidak menggunakan LoRA atau PEFT.

Ulčar dkk. [8] melakukan studi sistematis yang membandingkan model ELMo/BERT monolingual, trilingual, dan masif-multibahasa (mBERT, XLM-R) pada 14 tugas klasifikasi di 9 bahasa Eropa (termasuk bahasa less-resourced seperti Estonia, Latvia, Lituania, Slovenia). Model monolingual umumnya terbaik jika dilatih pada korpus cukup besar, model trilingual yang dirancang khusus (mis. CroSloEngual BERT) sering mengungguli model masif-multibahasa, dan transfer lintas bahasa zero-shot hanya kehilangan performa kecil pada sebagian besar tugas (rata-rata 4,9% pada NER, 4,7% pada POS-tagging), tetapi kehilangan jauh lebih besar (18,48%) pada dependency parsing. Studi ini tidak mencakup tugas klasifikasi topik berita/IPTC sama sekali, dan tidak menyinggung knowledge distillation maupun LoRA/PEFT.

### D. Kombinasi KD dan LoRA untuk Klasifikasi (Monolingual)

Azimi dkk. [6] mengusulkan KD-LoRA: teacher (BERT-base/RoBERTa-base/DeBERTa-v3-base) di-fine-tuning penuh terlebih dahulu, dibekukan, lalu pengetahuannya didistilasi ke student (DistilBERT/DistilRoBERTa/DeBERTa-v3-small yang sudah memiliki modul LoRA pada proyeksi query dan value) melalui kombinasi loss task (cross-entropy) dan loss distilasi (KL-divergence pada logit yang di-soften). Pada benchmark GLUE (9-10 tugas NLU standar), KD-LoRA mempertahankan ~97% performa FFT dan ~98% performa LoRA-saja, dengan reduksi parameter terlatih 99% (vs. FFT) dan 49% (vs. LoRA), reduksi memori GPU hingga 75% (vs. FFT), dan reduksi latensi inferensi rata-rata ~30% (bervariasi antar keluarga model berdasarkan data mentah yang dilaporkan: ~12% untuk BERT, ~38% untuk RoBERTa, ~28% untuk DeBERTa).

Studi ini adalah preseden metodologis terdekat dengan penelitian yang diusulkan, namun cakupannya secara eksplisit terbatas pada: (a) benchmark GLUE yang sepenuhnya berbahasa Inggris, tanpa data atau evaluasi lintas bahasa sama sekali; (b) tugas klasifikasi kalimat tunggal standar, bukan taxonomy berlabel-banyak/hierarkis seperti IPTC Media Topic; (c) tidak menyertakan baseline KD-saja (tanpa LoRA), sehingga kontribusi marjinal distilasi versus efek ukuran model student tidak terisolasi; dan (d) tidak memuat bagian keterbatasan maupun arah penelitian lanjutan.

### E. Research Gap and Positioning

Tabel I membandingkan pendekatan-pendekatan yang direview dengan posisi penelitian yang diusulkan.

**TABEL I. HUBUNGAN PENELITIAN YANG DIUSULKAN DENGAN PENDEKATAN YANG SUDAH ADA**

| Pendekatan | Contoh | Kelebihan | Keterbatasan untuk penelitian ini |
|---|---|---|---|
| Teacher-student LLM untuk IPTC | Kuzman & Ljubešić [1] | Multibahasa, tanpa anotasi manual, akurasi mendekati manusia | Full fine-tuning saja; efisiensi = efisiensi anotasi, bukan efisiensi model |
| PEFT/LoRA untuk klasifikasi multibahasa | Razuvayevskaya dkk. [3] | Reduksi parameter 140–280x, kompetitif pada bahasa yang terwakili di data latih | Rapuh pada bahasa unseen (Georgia); bukan taxonomy IPTC |
| PEFT low-resource | Nwaiwu [7] | LoRA konsisten terbaik saat FFT kolaps pada data sangat terbatas | English-only; tanpa distillation |
| KD lintas bahasa | Piperno dkk. [5] | Transfer domain-task efektif untuk bahasa target berdata terbatas | Bukan kompresi ukuran (teacher dan student sama-sama BERT-base penuh); tanpa LoRA/PEFT |
| Evaluasi representasi low-resource | Ulčar dkk. [8] | Peta lengkap trade-off monolingual/trilingual/masif-multibahasa | Tidak mencakup IPTC; tanpa KD atau PEFT |
| KD+LoRA untuk klasifikasi | Azimi dkk. [6] | Membuktikan KD+LoRA layak: retensi 97–98% performa dengan reduksi parameter hingga 99% | English-only, kalimat tunggal; bukan IPTC; tanpa baseline KD-saja |
| **Diusulkan (OURS)** | — | Mengadaptasi KD+LoRA ke IPTC multibahasa, dengan baseline KD-saja dan LoRA-saja, serta pengukuran efisiensi terpadu | Evaluasi end-to-end masih direncanakan (Bagian III) |

Dalam literatur yang direview sejauh ini, belum ditemukan pendekatan yang menggabungkan knowledge distillation dan LoRA untuk klasifikasi topik berita multibahasa berskema IPTC, sekaligus melaporkan trade-off efisiensi-akurasi secara terpadu. Ini adalah gap sementara (*provisional gap*), bukan klaim bahwa masalah ini belum pernah diteliti oleh siapa pun. Pencarian literatur kami dibatasi pada studi peer-reviewed dalam empat tahun terakhir (2022–2026) dan mengecualikan preprint arXiv tanpa review; kemungkinan ada studi relevan di luar cakupan pencarian ini, termasuk pada venue yang tidak terindeks oleh mesin pencari yang digunakan.

Dalam batasan tersebut, gap ini memotivasi RQ1–RQ3 dan metode yang direncanakan pada Bagian III.

## III. METHODOLOGY

## IV. CONCLUSION

## REFERENCES


[1] T. Kuzman and N. Ljubešić, "LLM teacher-student framework for text classification with no manually annotated data: A case study in IPTC news topic classification," IEEE Access, vol. 13, pp. 35621–35633, 2025, doi: 10.1109/ACCESS.2025.3544814.

[2] E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," in Proc. Int. Conf. Learning Representations (ICLR), 2022.

[3] O. Razuvayevskaya et al., "Comparison between parameter-efficient techniques and full fine-tuning: A case study on multilingual news article classification," PLOS ONE, vol. 19, no. 5, 2024, doi: 10.1371/journal.pone.0301738.

[4] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," in NIPS Deep Learning and Representation Learning Workshop, 2015.

[5] R. Piperno, L. Bacco, F. Dell'Orletta, M. Merone, and L. Pecchia, "Cross-lingual distillation for domain knowledge transfer with sentence transformers," Knowledge-Based Systems, vol. 311, art. no. 113079, 2025, doi: 10.1016/j.knosys.2025.113079.

[6] R. Azimi, R. Rishav, M. Teichmann, and S. Ebrahimi Kahou, "KD-LoRA: A hybrid approach to efficient fine-tuning with LoRA and knowledge distillation," in Proc. 4th NeurIPS Efficient Natural Language and Speech Processing Workshop, PMLR vol. 262, 2024, pp. 73–80.

[7] S. Nwaiwu, "Parameter-efficient fine-tuning for low-resource text classification: A comparative study of LoRA, IA3, and ReFT," Frontiers in Big Data, vol. 8, 2025, doi: 10.3389/fdata.2025.1677331.

[8] M. Ulčar et al., "Mono- and cross-lingual evaluation of representation language models on less-resourced languages," Computer Speech & Language, 2025, doi: 10.1016/j.csl.2025.101852.

[9] International Press Telecommunications Council (IPTC), "Media topics — IPTC NewsCodes," 2024. [Online]. Available: https://iptc.org/standards/media-topics/
