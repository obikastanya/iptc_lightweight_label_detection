from cells import code, md

CELLS = [
md(r"""
<a id="6"></a>
## 6. Training StaticKD dan baseline

### 6.1 Daftar run

Semua run StaticKD didefinisikan di `statickd.experiments.ABLATIONS`. Setiap varian mengubah **satu
faktor** terhadap resep final, dan setiap konfigurasi diulang dengan beberapa *seed* supaya perbedaannya
bisa diuji secara statistik:
- resep final: 10 *seed*;
- ablasi dan *temperature*: 5 *seed*;
- varian hasil negatif: 3 *seed*.

Dengan GPU RTX 4050, satu run membutuhkan sekitar 3–7 menit.
"""),
code(r"""
from statickd import experiments as ex
registry = pd.DataFrame([{"kode": k, "varian": d, "perubahan": ", ".join(f"{a}={b}" for a, b in o.items()) or "-",
                          "seed": len(s)} for k, (d, o, s) in ex.ABLATIONS.items()])
NUM.add("runs.total", len(ex.all_runs()), 0)
show(registry, "Registry run StaticKD", digits=0)
import inspect as _inspect
from statickd import baselines as bl, transformer_student as ts_mod
cfg = ex.StaticKDConfig(name="final"); tcfg = ts_mod.TransformerKDConfig(name="x")
ft_def = {k: v.default for k, v in _inspect.signature(bl.train_fasttext).parameters.items() if v.default is not _inspect.Parameter.empty}
tf_def = {k: v.default for k, v in _inspect.signature(bl.train_tfidf).parameters.items() if v.default is not _inspect.Parameter.empty}
GRID_BEST_FILE = FINAL_DIR / "baseline_grid_best.json"   # chosen on dev (scripts/tune_baselines.py)
GRID_BEST = json.loads(GRID_BEST_FILE.read_text()) if GRID_BEST_FILE.exists() else {}
if not GRID_BEST:
    print("PERINGATAN: grid baseline belum selesai; tabel memakai hiperparameter default.")
def chosen(name):   # say "chosen on dev" only when the grid result exists
    return "chosen on dev" if name in GRID_BEST else "fixed"
def ft_cfg(name):
    b = GRID_BEST.get(name, {"dim": ft_def["dim"], "lr": ft_def["lr"], "epoch": ft_def["epoch"]})
    return rf"dim {b['dim']}, lr {fmt(b['lr'], 1)}, {b['epoch']} \eng{{epoch}}"
def sci(x):
    e = int(np.floor(np.log10(x))); m = x / 10 ** e
    return f"$10^{{{e}}}$" if abs(m - 1) < 1e-9 else rf"${fmt(m, 0)}\cdot 10^{{{e}}}$"
SEC = lambda title: (rf"\multicolumn{{2}}{{@{{}}l}}{{\textit{{{title}}}}} \\", None)
hp = [SEC("StaticKD"),
      (r"\quad initial embedding", rf"potion-multilingual-128M ({fmt(E.shape[0], 0)} $\times$ {E.shape[1]})"),
      (r"\quad max. tokens / vector \eng{dropout}", f"1,024 / {fmt(cfg.dropout, 1)}"),
      (r"\quad \eng{loss}, $\tau$", rf"KL, $\tau={fmt(cfg.temperature, 0)}$"),
      (r"\quad \eng{token dropout}", fmt(cfg.token_dropout, 1)),
      (r"\quad optimizer embedding / head", rf"SparseAdam {sci(cfg.lr_emb)} / AdamW {sci(cfg.lr_head)}, wd $10^{{-4}}$"),
      (r"\quad \eng{batch}, \eng{epoch}, validation", rf"{cfg.batch_size}, {cfg.epochs}, {fmt(cfg.val_fraction, 0, percent=True)} of data (selects the \eng{{epoch}})"),
      (r"\quad \eng{ensemble}", rf"mean of {len(ex.V2_MEMBERS)} \eng{{seed}} tables"),
      SEC("fastText"),
      (rf"\quad word tokens ({chosen('ft_kd_word')})", ft_cfg("ft_kd_word")),
      (rf"\quad subword tokens ({chosen('ft_kd_subword')})", ft_cfg("ft_kd_subword")),
      (r"\quad GPT-4o labels", rf"dim {ft_def['dim']}, lr {fmt(ft_def['lr'], 1)}, 25 \eng{{epoch}}"),
      (r"\quad word n-grams, \eng{loss}", rf"{ft_def['word_ngrams']}, \eng{{softmax}}, 2M \eng{{buckets}}"),
      (r"\quad tokens", "words (+ character 2--5-grams) or XLM-R subwords"),
      (r"\quad quantization", r"\texttt{quantize}, \eng{cutoff} 300k, \eng{retrain}"),
      SEC("TF-IDF + SVM"),
      (r"\quad features", rf"XLM-R subword uni+bigrams, min.\ df 3, max.\ {fmt(tf_def['max_features'], 0)}, \eng{{sublinear tf}}"),
      (rf"\quad classifier ({chosen('tfidf_svm_kd')})", f"LinearSVC, $C={fmt(GRID_BEST.get('tfidf_svm_kd', {'c': tf_def['c']})['c'], 2)}$"),
      SEC("MiniLM-L6 / DistilmBERT + KD"),
      (r"\quad optimizer", rf"AdamW {sci(tcfg.lr)}, wd 0.01, \eng{{warmup}} 6\%, linear, fp16"),
      (r"\quad \eng{batch}, \eng{epoch}, max. tokens", f"{tcfg.batch_size}, {tcfg.epochs}, {tcfg.max_length}"),
      (r"\quad $\tau$", r"$\{1, 2\}$, chosen on dev")]
write_table("hparams", [a if b is None else rf"{a} & {b} \\" for a, b in hp])
show(pd.Series(ex.StaticKDConfig(name="final").__dict__).drop(["extra", "name"]).astype(str).to_frame("nilai"),
     "Hiperparameter resep final")
"""),
code(r"""
if RUN_TRAINING:
    run_script("train_statickd.py")   # skips runs that are already complete

seeds = an.seed_table()
meta0 = json.loads((st.STUDENTS_DIR / "full_s0" / "meta.json").read_text())
hist = pd.DataFrame([h for s in ex.MAIN_SEEDS for h in json.loads((st.STUDENTS_DIR / f"full_s{s}" / "meta.json").read_text())["history"]])
NUM.add("train.seconds", np.mean([json.loads((st.STUDENTS_DIR / f"full_s{s}" / "meta.json").read_text())["train_seconds"] for s in ex.MAIN_SEEDS]) / 60, 1)
NUM.add("train.docs", meta0["train_docs"], 0); NUM.add("train.val_docs", meta0["val_docs"], 0)
best_epochs = [int(np.argmax([h["val_agreement"] for h in json.loads((st.STUDENTS_DIR / f"full_s{s}" / "meta.json").read_text())["history"]]) + 1) for s in ex.MAIN_SEEDS]
print("run selesai:", len(seeds), "| epoch terbaik (10 seed resep final):", best_epochs)
fig, ax = plt.subplots(figsize=(5, 3))
for s, g in hist.groupby(np.repeat(ex.MAIN_SEEDS, meta0["config"]["epochs"])):
    ax.plot(g.epoch, g.val_agreement, color=viz.COLORS["statickd"], alpha=0.5)
ax.set_xlabel("epoch"); ax.set_ylabel("agreement with the teacher (validation)")
ax.set_title("Training curves of the final recipe (10 seeds)")
viz.show_and_save(fig, "fig_training_curves")
"""),
md(r"""
### 6.2 Model final: *ensemble* tabel tiga *seed* dan varian *compact*

StaticKD final adalah rata-rata tabel *seed* 0, 1, dan 2 dari resep final. Sel berikut membangun model
final, memeriksa bahwa logitnya sama dengan rata-rata logit ketiga model (Bagian 5.6), lalu membangun
varian *compact*.
"""),
code(r"""
FINAL = an.FINAL_MODELS_DIR
members = [st.STUDENTS_DIR / ex.run_name("full", s) for s in ex.V2_MEMBERS]
if not (FINAL / "statickd" / "model.npz").exists():
    st.ensemble(members, FINAL / "statickd")
if not (FINAL / "statickd_compact" / "model.npz").exists():
    st.compact(FINAL / "statickd", FINAL / "statickd_compact", min_count=10)

final_model = st.StaticKDModel(FINAL / "statickd")
sample = ev.eval_set("test")["text"].tolist()[:500]
mean_of_logits = np.mean([st.StaticKDModel(m).logits(sample) for m in members], axis=0)
diff = np.abs(final_model.logits(sample) - mean_of_logits)
same_pred = (final_model.logits(sample).argmax(1) == mean_of_logits.argmax(1)).mean()
print(f"ensemble: max |logit tabel rata-rata - rata-rata logit| = {diff.max():.4f} (pembulatan fp16), "
      f"prediksi sama: {same_pred:.3f}")
compact_model = st.StaticKDModel(FINAL / "statickd_compact")
size_table = pd.DataFrame({
    "model": ["StaticKD", "StaticKD-compact"],
    "token": [final_model.table.shape[0], compact_model.table.shape[0]],
    "parameter": [final_model.num_parameters, compact_model.num_parameters],
    "file (MB)": [final_model.size_mb, compact_model.size_mb]})
NUM.add("final.params", final_model.num_parameters / 1e6, 1); NUM.add("final.size", final_model.size_mb, 0)
NUM.add("compact.vocab", compact_model.table.shape[0], 0); NUM.add("compact.size", compact_model.size_mb, 0)
NUM.add("final.table_mb", final_model.table.nbytes / 2**20, 0)
for name, m in (("statickd", final_model), ("statickd_compact", compact_model)):
    ev.evaluate_model(name, m)
show(size_table, "Ukuran model final", digits=1)
"""),
md(r"""
### 6.3 Baseline

Semua *baseline* memakai **data distilasi yang sama** (285.999 dokumen, 69 bahasa) dan target *teacher* yang
sama. Satu-satunya perbedaan adalah arsitektur dan jenis label (keras atau lunak).

| Baseline | Deskripsi | Label |
|---|---|---|
| fastText + KD | fastText *supervised* (bigram kata), token kata (+ n-gram karakter 2–5) atau subword XLM-R; lr, dim, dan *epoch* dipilih pada dev (grid di bawah); varian terkuantisasi `.ftz` | argmax *teacher* |
| fastText, label GPT-4o | seperti Kuzman: hanya EMMediaTopic *train* (4 bahasa); dim 100, lr 0,5, 25 *epoch* | GPT-4o |
| TF-IDF + SVM + KD | TF-IDF subword XLM-R uni+bigram (maks. 1 juta fitur, *sublinear tf*) + LinearSVC (C dipilih pada dev); versi *baseline* non-neural Kuzman yang dilatih dengan label *teacher* | argmax *teacher* |
| MiniLM-L6 + KD | mMiniLMv2-L6-H384 (didistilasi dari XLM-R-large), KD dengan *loss* yang sama, 2 *epoch*, lr 5·10⁻⁵, 256 token; τ ∈ {1, 2} dipilih di dev | *soft* |
| DistilmBERT + KD | distilbert-base-multilingual-cased, resep sama dengan MiniLM | *soft* |
| *Teacher* ONNX int8 | ekspor ONNX *dynamic int8* dari onnx-community, konfigurasi CPU tercepat untuk *teacher* | – |

**Pemilihan hiperparameter baseline.** Supaya *budget* tuning tidak hanya berpihak ke StaticKD (72 run),
*baseline* non-transformer dipilih dengan grid yang ditetapkan sebelum dijalankan
(`scripts/tune_baselines.py`): fastText lr ∈ {0,1; 0,5; 1,0} × dim ∈ {100; 300} × *epoch* ∈ {5; 15; 25}
per jenis token, dan TF-IDF + SVM C ∈ {0,1; 0,25; 0,5; 1; 2}. Kriterianya macro-F1 dev, sama dengan
kriteria resep StaticKD; set uji tidak dipakai. Transformer memakai dua nilai τ (dipilih pada dev).
"""),
code(r"""
GRID_FILE = FINAL_DIR / "baseline_grid.jsonl"
grid = pd.read_json(GRID_FILE, lines=True) if GRID_FILE.exists() else pd.DataFrame(columns=["baseline", "lr", "dim", "epoch", "c", "dev_macro_f1"])
for k in ("lr", "dim", "epoch", "c"):
    if k not in grid:
        grid[k] = np.nan
grid = grid.drop_duplicates(["baseline", "lr", "dim", "epoch", "c"], keep="last")
show(grid.sort_values(["baseline", "dev_macro_f1"], ascending=[True, False]), "Grid baseline (macro-F1 dev)")
for name, g in grid.groupby("baseline"):
    if name not in GRID_BEST:   # this part of the grid is not finished yet
        continue
    b = GRID_BEST[name]
    NUM.add(f"grid.{name}.n", len(g), 0); NUM.add(f"grid.{name}.dev", b["dev_macro_f1"], 3)
    default = g[(g.c == 0.5)] if name == "tfidf_svm_kd" else g[(g.lr == 0.5) & (g.dim == 100) & (g.epoch == 15)]
    if len(default):
        NUM.add(f"grid.{name}.default_dev", default.dev_macro_f1.iloc[0], 3)
    for k in ("lr", "dim", "epoch", "c"):
        if k in b:
            NUM.add(f"grid.{name}.{k}", b[k], 2 if k == "c" else (1 if k == "lr" else 0))
NUM.add("grid.total", len(grid), 0)
print("konfigurasi terpilih:", GRID_BEST)
"""),
code(r"""
if RUN_BASELINES:
    run_script("train_baselines.py", "--threads", "8")
if RUN_TRANSFORMERS:
    run_script("train_transformers.py")
if RUN_EVAL_TEACHER:
    run_script("evaluate_model.py", "--name", "teacher", "--spec", "teacher-gpu", "--sets", "dev", "heldout", "test")
    run_script("evaluate_model.py", "--name", "teacher_onnx_int8", "--spec", "teacher-onnx-int8", "--sets", "dev", "test",
               "--threads", "8", "--batch-size", "8")

# MiniLM: the temperature is chosen on the development set only
minilm_dev = {t: ev.macro_f1(ev.eval_view("dev").y, ev.pred_view(f"minilm_kd_t{t}", "dev"))
              for t in (1, 2) if ev.has_predictions(f"minilm_kd_t{t}", "dev")}
MINILM = f"minilm_kd_t{max(minilm_dev, key=minilm_dev.get)}" if minilm_dev else None
print("MiniLM macro-F1 dev per temperature:", {t: round(v, 4) for t, v in minilm_dev.items()}, "-> dipakai:", MINILM)
MODEL_KEYS = {"minilm_kd": MINILM}
def key(m):  # registry key -> prediction name
    return MODEL_KEYS.get(m, m)
transformer_meta = {n: json.loads((an.BASELINES_DIR / key(n) / "train_meta.json").read_text())
                    for n in ("minilm_kd", "distilmbert_kd") if key(n) and (an.BASELINES_DIR / key(n) / "train_meta.json").exists()}
for n, m in transformer_meta.items():
    print(f"{n}: τ={m['temperature']}, {m['epochs']} epoch, {m['train_seconds'] / 60:.0f} menit, "
          f"ONNX vs PyTorch (200 dok dev) = {m.get('onnx_vs_pytorch_agreement_dev200')}")
"""),
md(r"""
**Panjang masukan.** Semua model menerima 512 kata pertama, tetapi batas token tiap model berbeda: *teacher*
512 token XLM-R (batas arsitektur), MiniLM dan DistilmBERT 256 token (resep *baseline*), StaticKD 1.024 token.
Sel berikut menghitung berapa artikel set uji yang terpotong oleh batas itu, dengan tokenizer masing-masing model.
"""),
code(r"""
from transformers import AutoTokenizer
test_texts = ev.eval_set("test").text.tolist()
test_words = np.array([len(t.split()) for t in test_texts])
TOKEN_LIMITS = {"teacher": (config.TEACHER_MODEL_ID, 512), "minilm_kd": (an.BASELINES_DIR / key("minilm_kd"), 256),
                "distilmbert_kd": (an.BASELINES_DIR / key("distilmbert_kd"), 256)}
for m, (path, limit) in TOKEN_LIMITS.items():
    tok_m = AutoTokenizer.from_pretrained(str(path))
    n_tok = np.array([len(tok_m(t, add_special_tokens=True)["input_ids"]) for t in test_texts])
    cut = n_tok > limit
    NUM.add(f"trunc.{m}.pct", cut.mean(), 1, percent=True)
    NUM.add(f"trunc.{m}.words_kept", float(np.median(test_words[cut] * limit / n_tok[cut])), 0)
    NUM.add(f"trunc.{m}.words_total", float(np.median(test_words[cut])), 0)
    NUM.add(f"trunc.{m}.tokens_mean", float(n_tok.mean()), 0)
    print(f"{m}: batas {limit} token, {cut.mean():.1%} artikel terpotong; median kata terbaca "
          f"{np.median(test_words[cut] * limit / n_tok[cut]):.0f} dari {np.median(test_words[cut]):.0f}")
static_tok = [len(ids) for ids in st.encode(final_model.tokenizer, test_texts, 10**6)] if "final_model" in globals() else []
if static_tok:
    NUM.add("trunc.statickd.pct", float(np.mean(np.array(static_tok) > 1024)), 1, percent=True)
    print(f"StaticKD: {np.mean(np.array(static_tok) > 1024):.1%} artikel melewati 1.024 token")
"""),
md(r"""
<a id="7"></a>
## 7. Protokol evaluasi dan uji statistik

**Metrik.**
- **Macro-F1** atas label yang hadir di referensi (sama dengan kode evaluasi Kuzman & Ljubešić). Ini
  metrik utama karena distribusi label tidak seimbang.
- **Akurasi** (= micro-F1).
- **Macro-F1 per bahasa rata-rata** (hanya di set uji): macro-F1 dihitung di setiap bahasa atas label
  yang ada di bahasa itu, lalu dirata-rata antarbahasa. Dengan begitu bahasa besar tidak mendominasi.
- **Kesepakatan** (*fidelity*): akurasi terhadap prediksi *teacher* di CC-News *held-out*. Hanya artikel
  dari situs yang tidak pernah muncul di data distilasi (di bahasa apa pun) yang dinilai.

**Interval kepercayaan.**
- Set uji: *site-cluster bootstrap* 95% (1.000×). Yang di-*resample* adalah situs, atau dataset sumber
  bila dataset tidak memiliki URL, bukan artikel, karena artikel dari situs yang sama tidak independen.
- Dev: *bootstrap* dokumen.

**Perbandingan dua model** pada dokumen yang sama:
- *paired (cluster) bootstrap* 2.000× untuk Δmacro-F1;
- uji McNemar eksak untuk akurasi;
- koreksi Holm–Bonferroni per keluarga uji.

**Variasi antar-*seed*.** Setiap varian ablasi dibandingkan dengan resep final memakai uji Welch atas skor
per *seed* (5 vs 10 *seed*).

**Aturan.** Resep final dan semua hiperparameter ditetapkan dari set *development*. Set uji gabungan
dibangun setelah resep ditetapkan dan tidak dipakai untuk memilih apa pun.
"""),
code(r"""
show_code(ev.bootstrap_samples)
show_code(ev.paired_bootstrap)
"""),
]
