from cells import code, md

CELLS = [
md(r"""
<a id="4"></a>
## 4. Teacher

*Teacher* adalah pengklasifikasi IPTC multibahasa terbuka dari Kuzman & Ljubešić (2025),
`classla/multilingual-IPTC-news-topic-classifier`. Model ini adalah XLM-RoBERTa-large yang di-*fine-tune*
pada 15.000 artikel EMMediaTopic berlabel GPT-4o (ca, el, hr, sl). Menurut paper aslinya, macro-F1 model
ini pada test set manusia adalah 0,746. Input dipotong ke 512 token.

### 4.1 Profil teacher
"""),
code(r"""
from huggingface_hub import snapshot_download
from safetensors import safe_open
teacher_dir = Path(snapshot_download(config.TEACHER_MODEL_ID))
with safe_open(str(teacher_dir / "model.safetensors"), "np") as f:
    teacher_params = sum(int(np.prod(f.get_slice(k).get_shape())) for k in f.keys())
teacher_cfg = json.loads((teacher_dir / "config.json").read_text())
teacher_profile = pd.Series({
    "arsitektur": teacher_cfg["architectures"][0], "layer": teacher_cfg["num_hidden_layers"],
    "hidden": teacher_cfg["hidden_size"], "parameter (juta)": round(teacher_params / 1e6, 1),
    "ukuran file (MB)": round((teacher_dir / "model.safetensors").stat().st_size / 2**20, 0),
    "vocab": teacher_cfg["vocab_size"], "maks. token": 512})
NUM.add("teacher.params", teacher_params / 1e6, 1)
NUM.add("teacher.layers", teacher_cfg["num_hidden_layers"], 0)
teacher_profile.to_frame("teacher")
"""),
md(r"""
### 4.2 Pelabelan distilasi (*soft label*)

Untuk setiap dokumen $d$ di korpus distilasi, *teacher* menghasilkan vektor logit
$\mathbf{z}^{T}(d)\in\mathbb{R}^{17}$ yang disimpan utuh, bukan hanya label argmax-nya. Distribusi lengkap
ini adalah inti *knowledge distillation* (Hinton dkk.): probabilitas kelas selain kelas pemenang
membawa informasi kemiripan antarkelas, misalnya artikel tentang anggaran kesehatan yang dekat dengan
*economy* dan *politics*.

Pelabelan dijalankan sekali secara *offline* dalam fp16 di GPU. Batch diurutkan menurut panjang teks agar
*padding* minimal. GPU hanya mempercepat proses; logitnya tidak bergantung pada perangkat.

Waktu pelabelan diukur terpisah pada sampel tetap 5.000 dokumen data distilasi
(`label_with_teacher.py --time-sample 5000`), tanpa menimpa label yang sudah tersimpan. Pengukuran ini juga
memeriksa bahwa logit yang dihasilkan ulang sama dengan logit yang tersimpan.
"""),
code(r"""
if RUN_TEACHER_LABELS:
    run_script("label_with_teacher.py", "--corpora", "emmediatopic", "ccnews_train", "ccnews_heldout", "ccnews_train_extra")

label_time = json.loads((FINAL_DIR / "teacher_labeling_time.json").read_text())
NUM.add("teacher.label_dps", label_time["docs_per_sec"], 1); NUM.add("teacher.label_hours", label_time["estimated_pool_hours"], 2)
NUM.add("teacher.label_sample", label_time["n_docs"], 0)
NUM.add("teacher.label_agree", label_time["argmax_agreement_with_stored"], 0, percent=True)
print(f"pelabelan teacher: {label_time['docs_per_sec']} dok/detik di {label_time['device']} (fp16) -> "
      f"{label_time['estimated_pool_hours']} jam GPU untuk {label_time['pool_docs']} dokumen; "
      f"argmax sama dengan label tersimpan: {label_time['argmax_agreement_with_stored']:.1%}")
teacher_p = data.teacher_probs(pool)
pool_t = pool.assign(teacher_label=teacher_p.argmax(1), teacher_conf=teacher_p.max(1))
dist = pd.DataFrame({
    "EMMediaTopic (GPT-4o)": emm[emm.split == "train"].label_id.value_counts(normalize=True).reindex(range(NUM_LABELS), fill_value=0).values,
    "CC-News (teacher)": pool_t[pool_t.source != "emmediatopic"].teacher_label.value_counts(normalize=True).reindex(range(NUM_LABELS), fill_value=0).values,
}, index=SHORT_LABELS)
conf_lang = pool_t[pool_t.source != "emmediatopic"].groupby("lang").teacher_conf.mean()
NUM.add("teacher.conf.min", conf_lang.min(), 2); NUM.add("teacher.conf.max", conf_lang.max(), 2)
for lab in ("politics", "economy, business and finance", "sport", "weather"):
    NUM.add(f"teacher.share.{SHORT_LABELS[LABELS.index(lab)]}", dist.loc[SHORT_LABELS[LABELS.index(lab)], "CC-News (teacher)"], 1, percent=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.2, 1]})
dist.plot.bar(ax=axes[0], color=["#999999", viz.COLORS["teacher"]], width=0.8)
axes[0].set_ylabel("share of documents"); axes[0].set_title("Label distribution of the distillation data")
axes[1].hist(conf_lang.values, bins=20, color=viz.COLORS["teacher"])
axes[1].set_xlabel("mean teacher confidence per language"); axes[1].set_ylabel("languages")
axes[1].set_title("Teacher confidence (max softmax) per language")
viz.show_and_save(fig, "fig_teacher_labels")
"""),
md(r"""
<a id="5"></a>
## 5. Metode StaticKD langkah demi langkah

StaticKD mengubah *teacher* 560 juta parameter menjadi **satu tabel berisi 17 angka per token**.
Rumusnya:

$$
\mathbf{h}(d)=\frac{1}{n}\sum_{i=1}^{n}\mathbf{e}_{t_i},\qquad
\mathbf{z}^{S}(d)=W\,\mathbf{h}(d)+\mathbf{b}
\;\;\Longrightarrow\;\;
\mathbf{z}^{S}(d)=\frac{1}{n}\sum_{i=1}^{n}T_{t_i}+\mathbf{b},\quad T=EW^{\top}
$$

dengan $t_1,\dots,t_n$ token dokumen, $E\in\mathbb{R}^{|V|\times 256}$ matriks embedding statis,
$W\in\mathbb{R}^{17\times 256}$ kepala linear, dan $T\in\mathbb{R}^{|V|\times 17}$ tabel hasil pelipatan.
Sub-bagian berikut membahas setiap komponen, masing-masing dengan contoh kecil yang bisa diperiksa.

### 5.1 Tokenisasi

Tokenizer Unigram milik `potion-multilingual-128M` (500.353 token) memecah teks dari bahasa apa pun
menjadi potongan subword. Token khusus tidak dipakai, token tidak dikenal dibuang, dan maksimal 1.024
token dipakai per dokumen. Artikel 512 kata jarang melebihi batas ini, dan biayanya linear.
"""),
code(r"""
from statickd import student as st
tokenizer, E = st.load_base_embeddings()
examples = {
    "id": "Bank Indonesia menaikkan suku bunga acuan untuk menahan inflasi.",
    "en": "The central bank raised its benchmark interest rate to curb inflation.",
    "sw": "Benki kuu imepandisha kiwango cha riba ili kudhibiti mfumuko wa bei.",
    "hi": "केंद्रीय बैंक ने महंगाई रोकने के लिए ब्याज दर बढ़ाई।",
}
for lang, text in examples.items():
    toks = tokenizer.encode(text, add_special_tokens=False).tokens
    print(f"{lang}: {len(toks):2d} token | {' '.join(toks[:14])}")
print(f"\nvocab = {E.shape[0]:,} token, dimensi embedding = {E.shape[1]}")
NUM.add("student.vocab", E.shape[0], 0); NUM.add("student.dim", E.shape[1], 0)
"""),
md(r"""
### 5.2 Embedding statis yang selaras lintas bahasa

$E$ diinisialisasi dari `potion-multilingual-128M` (Model2Vec). Embedding ini adalah embedding token
statis yang didistilasi dari *sentence encoder* BGE-M3 untuk 101 bahasa. Karena gurunya multibahasa,
kalimat yang maknanya sama dalam bahasa berbeda mendapat rata-rata embedding yang berdekatan. Sifat inilah
yang memungkinkan transfer ke bahasa yang sedikit atau tidak punya data distilasi. Contoh di bawah
menghitung *cosine similarity* rata-rata embedding antara kalimat "suku bunga" di empat bahasa dan kalimat
olahraga.
"""),
code(r"""
def doc_vector(text):
    ids = st.encode(tokenizer, [text])[0]
    return E[ids].mean(axis=0)

sport = {"id-sport": "Tim nasional menang 2-1 di final piala dunia.", "en-sport": "The national team won the World Cup final 2-1."}
vecs = {k: doc_vector(v) for k, v in {**examples, **sport}.items()}
names = list(vecs)
M = np.array([[np.dot(vecs[a], vecs[b]) / np.linalg.norm(vecs[a]) / np.linalg.norm(vecs[b]) for b in names] for a in names])
pd.DataFrame(M, index=names, columns=names).round(2)
"""),
md(r"""
Kalimat ekonomi dalam empat bahasa saling mirip (nilai tinggi), dan lebih mirip satu sama lain daripada
dengan kalimat olahraga dalam bahasa yang sama. Ini adalah transfer lintas bahasa yang tidak dimiliki
fastText, karena embedding fastText dilatih dari nol per token.

### 5.3 Target distilasi dan *temperature*

*Student* dilatih meminimalkan divergensi Kullback–Leibler terhadap distribusi *teacher* yang dilunakkan
dengan *temperature* $\tau$:

$$\mathcal{L}_{\mathrm{KD}}=\tau^{2}\,\mathrm{KL}\!\left(\mathrm{softmax}(\mathbf{z}^{T}/\tau)\,\middle\|\,\mathrm{softmax}(\mathbf{z}^{S}/\tau)\right)$$

Faktor $\tau^2$ menjaga skala gradien. $\tau$ besar membuat distribusi lebih rata, sehingga *student*
dipaksa meniru "ekor" distribusi. $\tau=1$ memakai distribusi asli *teacher*. Hasil ablasi (Bagian 9)
menunjukkan $\tau=1$ lebih baik untuk *student* berkapasitas sangat rendah ini. Contoh satu artikel:
"""),
code(r"""
row = pool[(pool.source == "ccnews") & (pool.lang == "en")].iloc[40]
z = np.asarray(row.teacher_logits)
fig, ax = plt.subplots(figsize=(10, 2.8))
for k, tau in enumerate([0.5, 1, 2, 4]):
    ax.bar(np.arange(NUM_LABELS) + (k - 1.5) * 0.2, data.softmax(z, tau), width=0.2, label=f"τ = {tau}")
ax.set_xticks(range(NUM_LABELS)); ax.set_xticklabels(SHORT_LABELS, rotation=45, ha="right")
ax.set_ylabel("probability"); ax.legend(ncol=4); ax.set_yscale("log"); ax.set_ylim(1e-4, 1.2)
ax.set_title(f"Teacher distribution for one article ({row.lang}): {row.text[:70]}...")
viz.show_and_save(fig, "fig_temperature_example")
"""),
md(r"""
### 5.4 Implementasi training

Fungsi `statickd.student.train` mengimplementasikan langkah-langkah berikut. Kode aslinya ditampilkan di
bawah, sehingga yang dibaca di notebook sama dengan yang dijalankan.

1. **Data.** Muat 285.999 dokumen distilasi beserta logit *teacher*, tokenisasi (dengan *cache*), dan
   hitung $p^T=\mathrm{softmax}(\mathbf{z}^T/\tau)$.
2. **Split.** Sisihkan 3% dokumen acak sebagai validasi internal. Metriknya kesepakatan argmax dengan
   *teacher*, dipakai untuk memilih *epoch* terbaik.
3. **Model.**
   - `EmbeddingBag(mode="mean")` diinisialisasi dari potion. Ini menghitung $\mathbf{h}(d)$ langsung dari
     token id tanpa matriks padat.
   - Kepala: *dropout* 0,1 pada $\mathbf{h}$, lalu `Linear(256, 17)`.
4. **Optimasi.**
   - Embedding memakai **SparseAdam** ($\eta=10^{-3}$): hanya baris token yang muncul di *batch* yang
     diperbarui. Ini penting, karena 500 ribu baris × 256 dimensi terlalu mahal jika diperbarui padat.
   - Kepala memakai AdamW ($\eta=3\cdot10^{-3}$, *weight decay* $10^{-4}$).
   - *Batch* 64 dokumen, 6 *epoch*.
5. **Token dropout.** Setiap token dokumen latih dibuang dengan probabilitas 0,2 pada setiap langkah. Ini
   augmentasi murah agar model tidak bergantung pada beberapa kata kunci.
6. **Simpan.** Bobot dari *epoch* dengan kesepakatan validasi tertinggi dilipat menjadi tabel $T$.
"""),
code(r"""
import inspect, re
from IPython.display import Code

def show_code(obj, start=None, end=None):
    # Display the real source of `obj` (optionally between two marker strings).
    src = inspect.getsource(obj)
    if start:
        src = src[src.index(start):]
    if end:
        src = src[:src.index(end) + len(end)]
    display(Code(src, language="python"))

show_code(st.train, "    def forward(idx, augment=False):", "            total += loss.item()")
"""),
md(r"""
### 5.5 Pelipatan kepala linear ke tabel token

Kepala linear dan rata-rata dapat dipertukarkan:
$W\big(\tfrac{1}{n}\sum_i \mathbf{e}_{t_i}\big) = \tfrac{1}{n}\sum_i (W\mathbf{e}_{t_i})$. Karena itu
$T=EW^\top$ dihitung sekali setelah *training*. Inferensi hanya berupa pengambilan baris tabel,
rata-rata, dan penambahan bias, tanpa perkalian matriks dan tanpa PyTorch. Tabel 500.353 × 17 dalam fp16
berukuran 17 MB. Sel berikut memeriksa kesetaraan ini secara numerik dengan kepala acak.
"""),
code(r"""
rng = np.random.default_rng(0)
W, b = rng.normal(0, 0.1, (NUM_LABELS, E.shape[1])).astype(np.float32), rng.normal(0, 0.1, NUM_LABELS).astype(np.float32)
T = E @ W.T
ids = st.encode(tokenizer, [examples["id"]])[0]
unfolded = W @ E[ids].mean(axis=0) + b      # training form: pool the 256-d embeddings, then the head
folded = T[ids].mean(axis=0) + b            # deployment form: pool 17-d table rows
print("max |selisih| =", float(np.abs(unfolded - folded).max()))
print(f"ukuran tabel: {T.shape[0]:,} x {T.shape[1]} = {T.astype(np.float16).nbytes / 2**20:.1f} MB (fp16) "
      f"vs embedding {E.nbytes / 2**20:.0f} MB (fp32)")
"""),
md(r"""
### 5.6 *Ensemble* tanpa biaya inferensi

*Student* bersifat linear terhadap tabelnya. Rata-rata logit $K$ model dengan *seed* berbeda,
$\frac{1}{K}\sum_k\big(\frac{1}{n}\sum_i T^{(k)}_{t_i}+\mathbf{b}^{(k)}\big)$, sama dengan logit **satu**
model yang tabelnya $\bar T=\frac{1}{K}\sum_k T^{(k)}$ dan biasnya $\bar{\mathbf{b}}$. Jadi *ensemble*
disimpan sebagai satu tabel dengan ukuran dan kecepatan yang sama. StaticKD final adalah rata-rata tabel
tiga *seed* (0, 1, 2) dari resep final. Resep ini ditetapkan dari set *development* sebelum set uji
gabungan dibangun. Pemeriksaannya dilakukan di Bagian 6 setelah model dilatih.

### 5.7 Varian *compact*

Struktur internal tokenizer Unigram 500 ribu token memakan sekitar 490 MB RAM, sedangkan tabelnya hanya
17 MB. Varian *compact* membuang potongan token yang muncul kurang dari 10 kali di korpus distilasi
(karakter tunggal tetap dipertahankan). Kata langka lalu dipecah menjadi potongan yang lebih pendek oleh
algoritma Viterbi Unigram.
"""),
]
