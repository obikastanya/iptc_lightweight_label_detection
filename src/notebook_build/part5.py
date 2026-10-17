from cells import code, md

CELLS = [
md(r"""
<a id="10"></a>
## 10. Efisiensi CPU (RQ1) dan analisis generalisasi (RQ4)

### 10.1 Protokol *benchmark*

Protokol ini dirancang untuk mensimulasikan server kecil tanpa GPU (`statickd.benchmark`):
- Setiap model dijalankan di **proses baru**, dengan GPU disembunyikan. Jumlah *thread* dikunci untuk
  semua pustaka (OMP, MKL, OpenBLAS, ONNX Runtime, `tokenizers`).
- **Pinning.** Proses dipin ke *core* performa fisik (P-core). CPU Intel Core Ultra 7 155H bersifat
  *hybrid*: 6 P-core, 8 E-core yang ±2,5× lebih lambat, dan 2 LP E-core yang ±3,7× lebih lambat. Tanpa
  *pinning*, Windows memindahkan *thread* ke *core* lambat dan hasil *multi-thread* menjadi tidak stabil.
  - 1 *core*: CPU logis 0.
  - 4 *core*: CPU logis 0, 10, 12, 14.
- **Latensi** diukur dengan satu artikel per panggilan (skenario *streaming*: rata-rata, p50, p95).
  **Throughput** diukur dengan *batch* 64 (8 untuk transformer).
- **RAM model** adalah kenaikan RSS setelah model dimuat.
- **Sampel** adalah sampel acak tetap dari set uji gabungan (47 bahasa, median ±310 kata), sama untuk
  semua model: 100 dokumen latensi dan 96 dokumen *throughput* untuk *teacher* dan transformer, 500 dan
  3.000 untuk model ringan.
- Seluruh *benchmark* **diulang tiga kali** (`--run 1, 2, 3`) saat mesin idle. Angka di naskah adalah
  median ketiga pengulangan; rentang min–maks dilaporkan sebagai ukuran variasi.
"""),
code(r"""
BENCH = [m for m in ["teacher", "teacher_onnx_int8", "distilmbert_kd", "minilm_kd", "statickd", "statickd_compact",
                     "ft_kd_subword_q", "ft_kd_subword", "tfidf_svm_kd"] if key(m)]
BENCH_FILE = FINAL_DIR / "cpu_benchmark.jsonl"

if RUN_BENCHMARK:   # three repetitions; one fresh process per (model, threads) in every repetition
    for rep in (1, 2, 3):
        run_script("run_benchmarks.py", "--minilm", key("minilm_kd"), "--run", str(rep))

bench_runs = pd.read_json(BENCH_FILE, lines=True)
if "run" not in bench_runs:   # older single-session file (before the 3x repetition protocol)
    bench_runs["run"] = 0
bench_runs = bench_runs.drop_duplicates(["model", "threads", "run"], keep="last")
METRICS = ["latency_ms_p50", "latency_ms_p95", "docs_per_sec", "rss_model_mb", "size_mb", "load_seconds"]
bench = bench_runs.groupby(["model", "threads"])[METRICS].median().reset_index()   # median over repetitions
bench = bench.merge(bench_runs.groupby(["model", "threads"]).agg(num_parameters=("num_parameters", "first"),
                    n_runs=("run", "nunique"), n_latency=("n_latency", "first"), n_throughput=("n_throughput", "first")).reset_index())
spread = bench_runs[bench_runs.threads == 1].groupby("model").agg(dps_min=("docs_per_sec", "min"), dps_max=("docs_per_sec", "max"),
                                                                    lat_min=("latency_ms_p50", "min"), lat_max=("latency_ms_p50", "max"))
eff = bench.pivot(index="model", columns="threads", values=["latency_ms_p50", "latency_ms_p95", "docs_per_sec"])
eff.columns = [f"{a}_{b}c" for a, b in eff.columns]
eff = eff.join(bench[bench.threads == 1].set_index("model")[["rss_model_mb", "size_mb", "load_seconds", "num_parameters", "n_runs", "n_latency", "n_throughput"]])
eff = eff.join(spread)
eff = eff.reindex([m for m in BENCH if m in eff.index])
t_onnx, t_pt = eff.loc["teacher_onnx_int8"], eff.loc["teacher"]
eff["speedup_vs_onnx"] = eff["docs_per_sec_1c"] / t_onnx["docs_per_sec_1c"]
eff["speedup_vs_pytorch"] = eff["docs_per_sec_1c"] / t_pt["docs_per_sec_1c"]
eff["cores_for_1M_per_day"] = (1e6 / 86400) / eff["docs_per_sec_1c"]
show(eff, "Efisiensi CPU (P-core terpin)", digits=2)
"""),
code(r"""
def n(v):
    return fmt(v, 2 if v < 10 else (1 if v < 100 else 0))
for m, r in eff.iterrows():
    for c in ["latency_ms_p50_1c", "latency_ms_p95_1c", "latency_ms_p50_4c", "docs_per_sec_1c", "docs_per_sec_4c"]:
        v = r[c]
        NUM.add(f"eff.{m}.{c}", v, 2 if v < 10 else (1 if v < 100 else 0))
    NUM.add(f"eff.{m}.ram", r["rss_model_mb"], 0); NUM.add(f"eff.{m}.size", r["size_mb"], 0)
    NUM.add(f"eff.{m}.speedup_onnx", r["speedup_vs_onnx"], 0 if r["speedup_vs_onnx"] > 10 else 1)
    NUM.add(f"eff.{m}.speedup_pt", r["speedup_vs_pytorch"], 0 if r["speedup_vs_pytorch"] > 10 else 1)
    NUM.add(f"eff.{m}.cores1M", r["cores_for_1M_per_day"], 2 if r["cores_for_1M_per_day"] < 1 else 1)
    NUM.add(f"eff.{m}.dps_1c.range", f"{n(r.dps_min)}--{n(r.dps_max)}")
    NUM.add(f"eff.{m}.lat_1c.range", f"{n(r.lat_min)}--{n(r.lat_max)}")
    NUM.add(f"eff.{m}.n_latency", r["n_latency"], 0); NUM.add(f"eff.{m}.n_throughput", r["n_throughput"], 0)
    # largest relative deviation of a repetition from the median (1 core throughput)
    NUM.add(f"eff.{m}.dps_1c.maxdev", max(abs(r.dps_max - r.docs_per_sec_1c), abs(r.docs_per_sec_1c - r.dps_min)) / r.docs_per_sec_1c, 0, percent=True)
NUM.add("bench.runs", int(eff["n_runs"].min()), 0)
NUM.add("bench.maxdev", max(max(abs(r.dps_max - r.docs_per_sec_1c), abs(r.docs_per_sec_1c - r.dps_min)) / r.docs_per_sec_1c for _, r in eff.iterrows()), 0, percent=True)
NUM.add("eff.speed_vs_minilm", eff.loc["statickd", "docs_per_sec_1c"] / eff.loc["minilm_kd", "docs_per_sec_1c"], 0)
NUM.add("eff.speed_vs_distilmbert", eff.loc["statickd", "docs_per_sec_1c"] / eff.loc["distilmbert_kd", "docs_per_sec_1c"], 0)
NUM.add("eff.size_ratio_onnx", eff.loc["teacher_onnx_int8", "size_mb"] / eff.loc["statickd", "size_mb"], 0)
NUM.add("eff.size_ratio_pt", eff.loc["teacher", "size_mb"] / eff.loc["statickd", "size_mb"], 0)
DEPS = {"teacher": "torch, transformers", "teacher_onnx_int8": "onnxruntime, transformers", "distilmbert_kd": "onnxruntime, transformers",
        "minilm_kd": "onnxruntime, transformers", "statickd": "numpy, tokenizers", "statickd_compact": "numpy, tokenizers",
        "ft_kd_subword_q": "fasttext, tokenizers", "ft_kd_subword": "fasttext, tokenizers", "tfidf_svm_kd": "scikit-learn, tokenizers"}
lines = []
for m, r in eff.iterrows():
    name = an.MODELS[m][0].replace(" (proposed)", "")
    b = (lambda s: f"\\textbf{{{s}}}") if m == "statickd" else (lambda s: s)
    lines.append(" & ".join([b(name), b(f"{n(r.latency_ms_p50_1c)} / {n(r.latency_ms_p95_1c)}"), b(n(r.docs_per_sec_1c)), b(n(r.docs_per_sec_4c)),
                             fmt(r.rss_model_mb, 0), b(fmt(r.size_mb, 0)), DEPS[m]]) + " \\\\")
write_table("eff", lines)
"""),
code(r"""
prof_runs = pd.read_json(FINAL_DIR / "cpu_profile.jsonl", lines=True)
if "run" not in prof_runs:
    prof_runs["run"] = 0
prof_runs = prof_runs.drop_duplicates(["model", "run"], keep="last")
prof = prof_runs.groupby("model")[["mean_tokens", "tokenize_docs_per_sec", "pool_docs_per_sec", "tokenize_share"]].median()   # median over repetitions
for m, r in prof.iterrows():
    NUM.add(f"prof.{m}.tokshare", r.tokenize_share, 0, percent=True)
    NUM.add(f"prof.{m}.pool_dps", r.pool_docs_per_sec, 0); NUM.add(f"prof.{m}.tok_dps", r.tokenize_docs_per_sec, 0)
    NUM.add(f"prof.{m}.tokens", r.mean_tokens, 0)
show(prof[["mean_tokens", "tokenize_docs_per_sec", "pool_docs_per_sec", "tokenize_share"]],
     "Rincian biaya StaticKD (1 thread): tokenisasi vs pooling tabel", digits=2)

acc_col = main_table["test_macro"]
from matplotlib.ticker import NullFormatter, ScalarFormatter
MARKERS = {"teacher": "s", "teacher_onnx_int8": "D", "distilmbert_kd": "^", "minilm_kd": "v", "statickd": "*",
           "statickd_compact": "P", "ft_kd_subword_q": "o", "ft_kd_subword": "X", "tfidf_svm_kd": "h"}
fig, ax = plt.subplots(figsize=(6.8, 4.0))
for m, r in eff.iterrows():   # a legend instead of point labels: several models overlap on the log axis
    if m not in acc_col:
        continue
    ax.scatter(r.docs_per_sec_1c, acc_col[m], s=150 if m == "statickd" else 60, marker=MARKERS[m], color=viz.color(m),
               zorder=3, edgecolor="white", linewidth=0.6, label=an.MODELS[m][0].replace(" (proposed)", ""))
ax.set_xscale("log"); ax.set_xlabel("throughput, 1 CPU core (documents/s, log)"); ax.set_ylabel("macro-F1, unified test set")
ax.legend(fontsize=7, loc="lower left", ncol=2, frameon=False)
ax.set_title("Accuracy–cost trade-off")
viz.show_and_save(fig, "fig_tradeoff")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
order = eff.sort_values("docs_per_sec_1c").index
for ax, col, title in zip(axes, ["latency_ms_p50_1c", "rss_model_mb", "size_mb"], ["p50 latency, 1 core (ms)", "model RAM (MB)", "file size (MB)"]):
    ax.barh([an.MODELS[m][0].replace(" (proposed)", "") for m in order], eff.loc[order, col], color=[viz.color(m) for m in order])
    ax.set_xscale("log"); ax.set_title(title); ax.tick_params(axis="y", labelsize=7)
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
for ax in axes[1:]:
    ax.set_yticklabels([])
viz.show_and_save(fig, "fig_latency_ram_size")
"""),
md(r"""
### 10.2 Analisis per bahasa

Dua sumber analisis dipakai:
- **set uji** (akurasi terhadap label independen, 47 bahasa);
- **CC-News *held-out*** (kesepakatan dengan *teacher*, 63 bahasa). Set ini dipakai untuk menguji faktor
  penentu, karena jumlah dokumen per bahasanya seragam.

Faktor yang diuji:
- jumlah data distilasi (log);
- jumlah situs latih;
- keyakinan rata-rata *teacher*;
- sistem tulisan.
"""),
code(r"""
from scipy import stats as sstats
lang_rows = []
for lang, g in test.groupby("lang"):
    idx = g.index.to_numpy(); y = g.y.to_numpy()
    row = {"lang": lang, "n": len(g), "labels": g.label.nunique(), **lang_info.loc[lang, ["region", "script", "distil_tier", "diversity_tier", "distil_docs"]].to_dict()}
    for m in GROUP_MODELS:
        p = ev.load_pred(key(m), "test")[idx]
        row[f"{m}_acc"], row[f"{m}_macro"] = (y == p).mean(), ev.macro_f1(y, p)
    lang_rows.append(row)
per_lang = pd.DataFrame(lang_rows).set_index("lang")
per_lang["gap_acc"] = per_lang["statickd_acc"] - per_lang["teacher_acc"]
worst = per_lang.sort_values("gap_acc").head(8)
show(per_lang.sort_values("statickd_acc")[["n", "labels", "region", "distil_docs", "teacher_acc", "statickd_acc", "ft_kd_subword_q_acc", "gap_acc"]],
     "Akurasi per bahasa di set uji")
NUM.add("lang.static_better_ft", int((per_lang.statickd_acc > per_lang.ft_kd_subword_q_acc).sum()), 0)
# paired test over languages (Demsar, 2006): is the per-language difference systematic?
# Reported for accuracy and for macro-F1, the main metric.
for m in ("ft_kd_subword_q", "tfidf_svm_kd", "distilmbert_kd", "teacher"):
    if f"{m}_acc" in per_lang:
        w = sstats.wilcoxon(per_lang.statickd_acc, per_lang[f"{m}_acc"])
        NUM.add(f"lang.wilcoxon.{m}.p", fmt_p(w.pvalue)); NUM.add(f"lang.wilcoxon.{m}.W", w.statistic, 0)
        NUM.add(f"lang.static_better.{m}", int((per_lang.statickd_acc > per_lang[f"{m}_acc"]).sum()), 0)
        NUM.add(f"lang.static_worse.{m}", int((per_lang.statickd_acc < per_lang[f"{m}_acc"]).sum()), 0)
        wm = sstats.wilcoxon(per_lang.statickd_macro, per_lang[f"{m}_macro"])
        NUM.add(f"lang.wilcoxon_macro.{m}.p", fmt_p(wm.pvalue))
        NUM.add(f"lang.static_better_macro.{m}", int((per_lang.statickd_macro > per_lang[f"{m}_macro"]).sum()), 0)
        NUM.add(f"lang.static_worse_macro.{m}", int((per_lang.statickd_macro < per_lang[f"{m}_macro"]).sum()), 0)
        print(f"StaticKD vs {m}: akurasi lebih baik di {(per_lang.statickd_acc > per_lang[f'{m}_acc']).sum()}/{len(per_lang)} bahasa "
              f"(Wilcoxon p = {w.pvalue:.2g}); macro-F1 lebih baik di "
              f"{(per_lang.statickd_macro > per_lang[f'{m}_macro']).sum()}/{len(per_lang)} bahasa (Wilcoxon p = {wm.pvalue:.2g})")
NUM.add("lang.n", len(per_lang), 0)
NUM.add("lang.gap.median", per_lang.gap_acc.median(), 3, sign=True)
rho_gap = sstats.spearmanr(np.log1p(per_lang.distil_docs), per_lang.gap_acc)
NUM.add("lang.gap_vs_data.rho", rho_gap.statistic, 2); NUM.add("lang.gap_vs_data.p", fmt_p(rho_gap.pvalue))
"""),
md(r"""
**Cakupan token.** Embedding sebuah token hanya diperbarui saat *training* jika token itu muncul di data
distilasi. Token lain tetap memakai embedding awal potion. Untuk setiap bahasa uji dihitung porsi
kemunculan token yang muncul minimal 10 kali di korpus distilasi. Pertanyaannya: apakah bahasa dengan
cakupan rendah tetap bisa diklasifikasikan berkat embedding awal yang selaras lintas bahasa, atau justru
gagal?
"""),
code(r"""
counts = st.token_counts(len(E))
seen = counts >= 10
cov = {}
for lang, g in test.groupby("lang"):
    ids = st.encode(tokenizer, g.text.tolist())
    cov[lang] = float(np.mean(np.concatenate([seen[i] for i in ids])))
per_lang["token_coverage"] = pd.Series(cov)
per_lang["xlmr"] = lang_info["xlmr"]
rho_cov = sstats.spearmanr(per_lang.token_coverage, per_lang.statickd_acc)
rho_cov_gap = sstats.spearmanr(per_lang.token_coverage, per_lang.gap_acc)
NUM.add("lang.cov_vs_acc.rho", rho_cov.statistic, 2); NUM.add("lang.cov_vs_acc.p", fmt_p(rho_cov.pvalue))
NUM.add("lang.cov_vs_gap.rho", rho_cov_gap.statistic, 2); NUM.add("lang.cov_vs_gap.p", fmt_p(rho_cov_gap.pvalue))
low = per_lang[per_lang.token_coverage < 0.9]
NUM.add("lang.lowcov.langs", ", ".join(sorted(low.index))); NUM.add("lang.lowcov.n", len(low), 0)
NUM.add("lang.lowcov.acc", low.statickd_acc.mean()); NUM.add("lang.lowcov.teacher_acc", low.teacher_acc.mean())
hi = per_lang[per_lang.token_coverage >= 0.9]
NUM.add("lang.hicov.acc", hi.statickd_acc.mean()); NUM.add("lang.hicov.teacher_acc", hi.teacher_acc.mean())
out_x = per_lang[per_lang.xlmr.str.startswith("di luar")]
NUM.add("lang.outxlmr.langs", ", ".join(sorted(out_x.index)))
NUM.add("lang.outxlmr.teacher_acc", out_x.teacher_acc.mean()); NUM.add("lang.outxlmr.static_acc", out_x.statickd_acc.mean())
print(f"Spearman cakupan token vs akurasi StaticKD: rho={rho_cov.statistic:.2f} (p={rho_cov.pvalue:.3g}); "
      f"vs selisih terhadap teacher: rho={rho_cov_gap.statistic:.2f} (p={rho_cov_gap.pvalue:.3g})")
per_lang["compact_acc"] = pd.Series({l: float((ev.load_pred("statickd_compact", "test")[g.index] == g.y).mean())
                                     for l, g in test.groupby("lang")})
very_low = per_lang[per_lang.token_coverage < 0.5]
NUM.add("lang.vlowcov.langs", ", ".join(sorted(very_low.index))); NUM.add("lang.vlowcov.n", len(very_low), 0)
NUM.add("lang.vlowcov.cov_max", very_low.token_coverage.max(), 0, percent=True)
NUM.add("lang.vlowcov.static_acc", very_low.statickd_acc.mean()); NUM.add("lang.vlowcov.compact_acc", very_low.compact_acc.mean())
NUM.add("lang.vlowcov.teacher_acc", very_low.teacher_acc.mean())
rest = per_lang[per_lang.token_coverage >= 0.5]
NUM.add("lang.restcov.static_acc", rest.statickd_acc.mean()); NUM.add("lang.restcov.compact_acc", rest.compact_acc.mean())
show(per_lang.sort_values("token_coverage")[["token_coverage", "xlmr", "distil_docs", "teacher_acc", "statickd_acc", "compact_acc", "gap_acc"]].head(15),
     "Bahasa dengan cakupan token terendah (termasuk varian compact)")
"""),
code(r"""
# factors on the CC-News held-out set (agreement with the teacher per language, >= 20 documents)
hv = ev.eval_view("heldout")
hp = ev.pred_view("statickd", "heldout"); hf = ev.pred_view("ft_kd_subword_q", "heldout")
hconf = data.softmax(np.stack(hv.teacher_logits.to_numpy())).max(1)
sites_per_lang = pool[pool.source != "emmediatopic"].groupby("lang").domain.nunique()
fac = pd.DataFrame({"agree": pd.Series(hp == hv.y.to_numpy()).groupby(hv.lang.values).mean(),
                    "agree_ft": pd.Series(hf == hv.y.to_numpy()).groupby(hv.lang.values).mean(),
                    "n": hv.groupby("lang").size(), "teacher_conf": pd.Series(hconf).groupby(hv.lang.values).mean()})
fac = fac[fac.n >= 20]
fac["distil_docs"] = pool_langs.reindex(fac.index).fillna(0); fac["sites"] = sites_per_lang.reindex(fac.index).fillna(0)
from statickd.corpus.ccnews import NO_SPACE_LANGS
SCRIPT_H = {**an.SCRIPT}
cyr = {"be", "bg", "kk", "ky", "mk", "mn", "ru", "sr", "uk"}; arab = {"ar", "fa", "ps", "ur"}; indic = {"bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "ta", "te"}
cjk = {"ja", "ko", "my", "th", "zh"}; other = {"am", "hy", "ka", "he", "el"}
fac["script"] = ["Sirilik" if l in cyr else "Arab" if l in arab else "Indik" if l in indic else "CJK/Thai/Myanmar" if l in cjk
                 else "lainnya" if l in other else "Latin" for l in fac.index]
tests_f = {"log(data distilasi)": np.log1p(fac.distil_docs), "jumlah situs latih": fac.sites, "keyakinan teacher": fac.teacher_conf}
corr = pd.DataFrame({k: sstats.spearmanr(v, fac.agree) for k, v in tests_f.items()}, index=["rho", "p"]).T
kw = sstats.kruskal(*[g.agree.values for _, g in fac.groupby("script")])
for k, r in corr.iterrows():
    kk = {"log(data distilasi)": "data", "jumlah situs latih": "sites", "keyakinan teacher": "conf"}[k]
    NUM.add(f"fac.{kk}.rho", r.rho, 2); NUM.add(f"fac.{kk}.p", fmt_p(r.p))
NUM.add("fac.kw.H", kw.statistic, 2); NUM.add("fac.kw.p", fmt_p(kw.pvalue)); NUM.add("fac.langs", len(fac), 0)
NUM.add("fac.static_better_ft", int((fac.agree > fac.agree_ft).sum()), 0)
NUM.add("fac.static_better_ft.pct", (fac.agree > fac.agree_ft).mean(), 0, percent=True)
display(corr.round(3)); print(f"Kruskal–Wallis antarsistem tulisan: H={kw.statistic:.2f}, p={kw.pvalue:.3f}")
display(fac.groupby("script").agree.agg(["mean", "count"]).round(3))
lines = [f"Spearman: agreement vs log(distillation data) & $\\rho={fmt(corr.loc['log(data distilasi)', 'rho'], 2)}$ & {fmt_p(corr.loc['log(data distilasi)', 'p'])} \\\\",
         f"Spearman: agreement vs number of training sites & $\\rho={fmt(corr.loc['jumlah situs latih', 'rho'], 2)}$ & {fmt_p(corr.loc['jumlah situs latih', 'p'])} \\\\",
         f"Spearman: agreement vs \\eng{{teacher}} confidence & $\\rho={fmt(corr.loc['keyakinan teacher', 'rho'], 2)}$ & {fmt_p(corr.loc['keyakinan teacher', 'p'])} \\\\",
         f"Kruskal--Wallis: across scripts & $H={fmt(kw.statistic, 2)}$ & {fmt_p(kw.pvalue)} \\\\",
         f"Spearman (test): accuracy gap vs log(data) & $\\rho={fmt(rho_gap.statistic, 2)}$ & {fmt_p(rho_gap.pvalue)} \\\\",
         f"Spearman (test): StaticKD accuracy vs token coverage & $\\rho={fmt(rho_cov.statistic, 2)}$ & {fmt_p(rho_cov.pvalue)} \\\\",
         f"Spearman (test): accuracy gap vs token coverage & $\\rho={fmt(rho_cov_gap.statistic, 2)}$ & {fmt_p(rho_cov_gap.pvalue)} \\\\"]
write_table("factors", lines)
"""),
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
tier_color = {"tidak ada (0)": "#D55E00", "sedikit (<500)": "#E69F00", "cukup (≥500)": "#0072B2"}
ax = axes[0]
for tier, g in per_lang.groupby("distil_tier"):
    ax.scatter(g.teacher_acc, g.statickd_acc, s=18 + g.n / 12, color=tier_color[tier], label=f"distillation data {an.en(tier)}", alpha=0.85)
for lang, r in per_lang.iterrows():
    ax.annotate(lang, (r.teacher_acc, r.statickd_acc), fontsize=6, xytext=(2, 2), textcoords="offset points")
ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--"); ax.set_xlim(0.2, 1); ax.set_ylim(0.2, 1)
ax.set_xlabel("teacher accuracy"); ax.set_ylabel("StaticKD accuracy"); ax.legend(fontsize=7, loc="upper left")
ax.set_title("Test set: StaticKD vs teacher per language")
ax = axes[1]
ax.scatter(np.log10(fac.distil_docs + 1), fac.agree, color=viz.COLORS["statickd"], s=16, label="StaticKD")
ax.scatter(np.log10(fac.distil_docs + 1), fac.agree_ft, color=viz.COLORS["fasttext"], s=12, marker="x", label="fastText + KD (q)")
ax.set_xlabel("log10(distillation documents + 1)"); ax.set_ylabel("agreement with the teacher (held-out)")
ax.legend(fontsize=7); ax.set_title("Held-out: agreement per language vs data")
viz.show_and_save(fig, "fig_per_language")
"""),
md(r"""
### 10.3 Analisis per label
"""),
code(r"""
y_test = test.y.to_numpy()
label_f1 = pd.DataFrame({m: ev.per_label_f1(y_test, ev.load_pred(key(m), "test")) for m in ["teacher", "statickd", "ft_kd_subword_q"]}, index=SHORT_LABELS)
label_f1["n"] = pd.Series(y_test).value_counts().reindex(range(NUM_LABELS), fill_value=0).values
label_f1["gap"] = label_f1.statickd - label_f1.teacher
dev_view = ev.eval_view("dev")
label_f1_dev = pd.DataFrame({m: ev.per_label_f1(dev_view.y, ev.pred_view(key(m), "dev")) for m in ["teacher", "statickd"]}, index=SHORT_LABELS)
for lab, r in label_f1.iterrows():
    NUM.add(f"label.{lab.replace(' ', '').replace('.', '')}.static", r.statickd, 2); NUM.add(f"label.{lab.replace(' ', '').replace('.', '')}.teacher", r.teacher, 2)
def join_and(items):  # English list with a serial comma: "a, b, and c" (IEEE style)
    return items[0] if len(items) == 1 else " and ".join(items) if len(items) == 2 else ", ".join(items[:-1]) + ", and " + items[-1]
hardest = label_f1.sort_values("statickd").head(4).index.tolist(); easiest = label_f1.sort_values("statickd", ascending=False).head(3).index.tolist()
NUM.add("label.hardest", join_and([f"\\eng{{{h}}} ({fmt(label_f1.loc[h, 'statickd'], 2)})" for h in hardest]))
NUM.add("label.easiest", join_and([f"\\eng{{{h}}} ({fmt(label_f1.loc[h, 'statickd'], 2)})" for h in easiest]))
biggest = label_f1.sort_values("gap").head(3).index.tolist()
NUM.add("label.biggest_gap", join_and([f"\\eng{{{h}}} ({fmt(label_f1.loc[h, 'gap'], 2, sign=True)})" for h in biggest]))
show(label_f1.sort_values("statickd"), "F1 per label di set uji")
cm = ev.confusion(y_test, ev.load_pred("statickd", "test"), normalise=True)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
xx = np.arange(NUM_LABELS)
for k, m in enumerate(["teacher", "statickd", "ft_kd_subword_q"]):
    axes[0].barh(xx + (k - 1) * 0.27, label_f1[m], 0.27, color=viz.color(m), label=an.MODELS[m][0].replace(" (proposed)", ""))
axes[0].set_yticks(xx); axes[0].set_yticklabels(SHORT_LABELS); axes[0].invert_yaxis(); axes[0].set_xlabel("F1 (test set)"); axes[0].legend(fontsize=7)
im = axes[1].imshow(cm, cmap="Blues", vmin=0, vmax=1)
axes[1].set_xticks(xx); axes[1].set_xticklabels(SHORT_LABELS, rotation=60, ha="right", fontsize=7)
axes[1].set_yticks(xx); axes[1].set_yticklabels(SHORT_LABELS, fontsize=7); axes[1].grid(False)
axes[1].set_xlabel("StaticKD prediction"); axes[1].set_ylabel("reference label"); fig.colorbar(im, ax=axes[1], shrink=0.7)
viz.show_and_save(fig, "fig_per_label")
off = cm.copy(); np.fill_diagonal(off, 0)
top_conf = sorted(((off[i, j], SHORT_LABELS[i], SHORT_LABELS[j]) for i in range(NUM_LABELS) for j in range(NUM_LABELS)), reverse=True)[:5]
print("kebingungan terbesar (referensi -> prediksi):", [(a, b, round(v, 2)) for v, a, b in top_conf])
NUM.add("label.top_confusions", "; ".join(f"\\eng{{{a}}} $\\rightarrow$ \\eng{{{b}}} ({fmt(v, 0, percent=True)})" for v, a, b in top_conf[:3]))
"""),
md(r"""
### 10.4 Kalibrasi dan mode *cascade*

Probabilitas softmax StaticKD dapat dipakai sebagai sinyal keyakinan. Dalam mode *cascade*, artikel
dengan keyakinan ≥ θ diputuskan oleh StaticKD, dan sisanya dikirim ke *teacher* ONNX int8. *Throughput*
efektif satu *core* dihitung sebagai $1/(1/r_S + (1-c)/r_T)$, dengan $r_S$ dan $r_T$ *throughput*
StaticKD dan *teacher*, serta $c$ porsi artikel yang diputuskan StaticKD.
"""),
code(r"""
calib = {}
for m in ["teacher", "statickd"]:
    for s in ("dev", "test"):
        calib[(m, s)] = ev.ece(data.softmax(ev.logits_view(key(m), s)), ev.eval_view(s).y.to_numpy())
calib = pd.Series(calib).unstack()
for (m, s), v in pd.Series({(m, s): calib.loc[m, s] for m in calib.index for s in calib.columns}).items():
    NUM.add(f"ece.{m}.{s}", v, 3)
display(calib.round(3))
fig, ax = plt.subplots(figsize=(4, 3.6))
for s, c in (("dev", "#0072B2"), ("test", "#D55E00")):
    rel = ev.reliability(data.softmax(ev.logits_view("statickd", s)), ev.eval_view(s).y.to_numpy())
    ax.plot(rel.confidence, rel.accuracy, marker="o", color=c, label=f"{s} (ECE {calib.loc['statickd', s]:.3f})")
ax.plot([0, 1], [0, 1], ls="--", color="#999999"); ax.set_xlabel("keyakinan StaticKD"); ax.set_ylabel("akurasi")
ax.legend(fontsize=7); ax.set_title("StaticKD reliability diagram")
viz.show_and_save(fig, "fig_calibration")
"""),
code(r"""
p_s = data.softmax(ev.load_logits("statickd", "test")); conf = p_s.max(1); pred_s = p_s.argmax(1)
teacher_cpu = "teacher_onnx_int8" if ev.has_predictions("teacher_onnx_int8", "test") else "teacher"
pred_t = ev.load_pred(teacher_cpu, "test")
r_s, r_t = eff.loc["statickd", "docs_per_sec_1c"], eff.loc["teacher_onnx_int8", "docs_per_sec_1c"]
casc = []
for theta in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]:
    keep = conf >= theta
    pred = np.where(keep, pred_s, pred_t)
    casc.append({"theta": theta, "coverage": keep.mean(), "acc_covered": (pred_s[keep] == y_test[keep]).mean() if keep.any() else np.nan,
                 "macro_f1": ev.macro_f1(y_test, pred), "accuracy": (pred == y_test).mean(),
                 "docs_per_sec": 1 / (1 / r_s + (1 - keep.mean()) / r_t)})
casc = pd.DataFrame(casc)
for _, r in casc.iterrows():
    k = f"casc.{str(r.theta).replace('.', '')}"
    NUM.add(f"{k}.coverage", r.coverage, 1, percent=True)
    # same precision rule as the efficiency table (Table X), so the teacher-only throughput reads 0.77 in both
    NUM.add(f"{k}.docs_per_sec", r.docs_per_sec, 2 if r.docs_per_sec < 10 else (1 if r.docs_per_sec < 100 else 0))
    for c in ["acc_covered", "macro_f1", "accuracy"]:
        NUM.add(f"{k}.{c}", r[c], 3)
lines = []
for _, r in casc.iterrows():
    th = "0 (StaticKD only)" if r.theta == 0 else ("\\eng{Teacher} only" if r.theta > 1 else fmt(r.theta, 1))
    lines.append(" & ".join([th, fmt(r.coverage, 1, percent=True), fmt(r.acc_covered) if pd.notna(r.acc_covered) else "--",
                             fmt(r.macro_f1), fmt(r.accuracy), n(r.docs_per_sec)]) + " \\\\")
write_table("cascade", lines)
show(casc, f"Cascade StaticKD -> {teacher_cpu} di set uji")
"""),
md(r"""
### 10.5 Contoh kualitatif

Contoh buatan dalam beberapa bahasa, termasuk bahasa Jawa dan Sunda yang tidak ada di data distilasi,
diikuti beberapa kesalahan StaticKD di set uji yang diprediksi benar oleh *teacher*.
"""),
code(r"""
demo = {
    "id": "Pemerintah resmi menaikkan harga BBM bersubsidi mulai Sabtu. Menteri Keuangan menyebut kenaikan ini perlu untuk menjaga APBN.",
    "jv": "Tim Persebaya menang 3-1 nglawan Arema ing stadion Gelora Bung Tomo, pelatih ngendika para pemain main kanthi apik.",
    "su": "Hujan gede jeung angin puting beliung ngaruksak puluhan imah di Garut, BPBD ngungsikeun warga ka tempat anu aman.",
    "en": "Researchers at the university developed a new vaccine that reduced malaria infections by 70 percent in clinical trials.",
    "de": "Die Gewerkschaft ruft zu einem landesweiten Streik auf, um höhere Löhne für Pflegekräfte durchzusetzen.",
    "sw": "Mvua kubwa imesababisha mafuriko katika mji wa Mombasa na maelfu ya watu wamehamishwa.",
}
proba = final_model.predict_proba(list(demo.values()))
display(pd.DataFrame({"bahasa": list(demo), "teks": [t[:80] + "..." for t in demo.values()],
                      "prediksi StaticKD": [LABELS[i] for i in proba.argmax(1)], "keyakinan": proba.max(1).round(2)}))
wrong = test[(ev.load_pred("statickd", "test") != test.y) & (ev.load_pred("teacher", "test") == test.y)]
ex_rows = wrong.groupby("label_origin").head(2)
display(pd.DataFrame({"bahasa": ex_rows.lang, "asal": ex_rows.label_origin.map(an.ORIGIN_NAMES), "judul/teks": ex_rows.text.str[:90],
                      "label": ex_rows.label, "StaticKD": [LABELS[i] for i in ev.load_pred("statickd", "test")[ex_rows.index]]}))
"""),
md(r"""
### 10.6 Ringkasan temuan

Ringkasan berikut dibangkitkan dari angka yang terkumpul di `NUM`, bukan diketik manual.
"""),
code(r"""
V = NUM.values
summary_md = f'''
| Pertanyaan | Temuan utama |
|---|---|
| RQ1 biaya | Teacher PyTorch {V['eff.teacher.latency_ms_p50_1c']} ms/artikel, ONNX int8 {V['eff.teacher_onnx_int8.latency_ms_p50_1c']} ms. StaticKD {V['eff.statickd.latency_ms_p50_1c']} ms, {V['eff.statickd.docs_per_sec_1c']} artikel/detik/core (≈{V['eff.statickd.speedup_onnx']}× teacher ONNX int8), {V['final.size']} MB, 1 juta artikel/hari = {V['eff.statickd.cores1M']} core |
| RQ2 akurasi | Set uji {V['ts.langs']} bahasa: teacher {V['teacher.test.macro']}, StaticKD {V['statickd.test.macro']} ({V['statickd.test.pct_teacher']} teacher), MiniLM {V.get('minilm_kd.test.macro', '–')}, DistilmBERT {V.get('distilmbert_kd.test.macro', '–')}, fastText+KD {V['ft_kd_subword_q.test.macro']}, TF-IDF+SVM {V.get('tfidf_svm_kd.test.macro', '–')} |
| RQ3 komponen | Tanpa data multibahasa {V['abl.emm_only.test_macro.delta']}, init acak {V['abl.random_init.test_macro.delta']}, embedding beku {V['abl.frozen.test_macro.delta']}, label keras {V['abl.hard.test_macro.delta']}, τ=2 {V['abl.tau2.test_macro.delta']} (macro-F1 uji, rata-rata seed) |
| RQ4 generalisasi | % teacher: bahasa dengan data cukup {V['grp.distil_tier.cukup.pct']}, tanpa data distilasi {V['grp.distil_tier.tidakada.pct']}; bahasa di luar XLM-R: StaticKD {V['lang.outxlmr.static_acc']} vs teacher {V['lang.outxlmr.teacher_acc']} (akurasi) |
'''
display(Markdown(summary_md.replace("\\%", "%").replace("$-$", "−")))   # LaTeX notation -> plain text
"""),
md(r"""
<a id="11"></a>
## 11. Ekspor angka, tabel, dan gambar ke jurnal

Semua angka yang dikumpulkan di `NUM` ditulis ke `results/final/paper_generated/numbers.tex`, dengan salinan
mentah di `results/final/paper_numbers.json`. Tabel LaTeX ditulis ke `results/final/paper_generated/tab_*.tex`
dan gambar ke `results/final/figures/`. Sumber naskah di `src/paper/` memanggil angka dengan `\val{kunci}`;
`python scripts/build_paper.py` menyusun satu file `jurnal_final/main.tex` dengan angka literal dan hanya
menyalin gambar yang dipakai ke `jurnal_final/figures/`.
"""),
code(r"""
path = NUM.write()
from statickd.report import GENERATED_DIR
generated = sorted(p.name for p in GENERATED_DIR.glob("*.tex"))
figures = sorted(p.name for p in viz.FIG_DIR.glob("fig_*.pdf"))
print(f"{len(NUM.values)} angka -> {path}")
print("tabel:", generated)
print("gambar:", figures)
"""),
md(r"""
<a id="12"></a>
## 12. Keterbatasan dan catatan reproduktibilitas

**Ancaman validitas**
1. **Label silver.** 68% set uji berlabel kategori/rubrik penerbit. Pemetaannya konservatif dan tervalidasi
   terhadap pohon IPTC, tetapi tidak dianotasi ulang oleh manusia. Bagian 8.5 memberi pemeriksaan tidak
   langsung.
2. **Label "manusia" MasakhaNEWS** memakai skema MasakhaNEWS yang dipetakan ke IPTC. Hanya MN-DS yang
   dianotasi langsung dengan IPTC.
3. **Cakupan label per bahasa timpang.** Bahasa Afrika hanya mencakup 4–5 topik besar. Macro-F1 dihitung
   atas label yang hadir.
4. **Keragaman situs.** 27 bahasa praktis berasal dari satu penerbit (BBC/VOA untuk bahasa Afrika,
   L3Cube untuk bahasa India). Klaim untuk bahasa-bahasa ini terbatas pada penerbit tersebut.
5. **Batas atas *student* adalah *teacher*.** Kesalahan dan bias *teacher* ikut tertiru.
6. **Satu mesin *benchmark*.** Angka absolut di server akan berbeda, tetapi rasio antarmodel tetap
   informatif.
7. **Test set manusia Kuzman & Ljubešić tidak diakses** karena tidak publik.

**Reproduktibilitas**
- Seluruh langkah dapat diulang dengan saklar `RUN_*`. *Seed* dicatat di setiap `meta.json`.
- Versi pustaka ada di `results/final/environment.json`.
- Set uji dirilis sebagai manifest: ID/URL, label, dan asal label, tanpa teks yang berhak cipta, beserta
  skrip rekonstruksinya.
"""),
]
