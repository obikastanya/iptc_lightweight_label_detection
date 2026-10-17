from cells import code, md

CELLS = [
md(r"""
<a id="8"></a>
## 8. Hasil akurasi (RQ2)

### 8.1 Tabel utama

Semua model dinilai pada tiga set:
- **dev**: label GPT-4o, 4 bahasa;
- **held-out**: kesepakatan dengan *teacher*, 63 bahasa, situs tak terlihat;
- **uji gabungan**: label manusia/penerbit, 47 bahasa.

Interval kepercayaan 95% ditulis dalam kurung siku.
"""),
code(r"""
MAIN = ["teacher", "teacher_onnx_int8", "distilmbert_kd", "minilm_kd", "statickd", "statickd_compact",
        "ft_kd_subword_q", "ft_kd_subword", "ft_kd_word", "tfidf_svm_kd", "ft_gpt_subword", "ft_gpt_word"]
MAIN = [m for m in MAIN if key(m) and ev.has_predictions(key(m), "dev")]
rows = []
for m in MAIN:
    r = {"model": m, "nama": an.MODELS[m][0]}
    for s in ("dev", "heldout", "test"):
        if not ev.has_predictions(key(m), s):
            continue
        sr = an.score_row(key(m), s, n_boot=N_BOOT)
        r.update({f"{s}_macro": sr["macro_f1"], f"{s}_lo": sr["ci_low"], f"{s}_hi": sr["ci_high"], f"{s}_acc": sr["accuracy"]})
        if s == "test":
            r["test_lang_macro"] = sr["lang_macro_f1"]
    rows.append(r)
main_table = pd.DataFrame(rows).set_index("model")
T_REF = main_table.loc["teacher"]
main_table["test_vs_teacher"] = main_table["test_macro"] / T_REF["test_macro"]
main_table["dev_vs_teacher"] = main_table["dev_macro"] / T_REF["dev_macro"]
show(main_table.drop(columns=["nama"]), "Tabel utama (macro-F1, CI95, akurasi)")
"""),
code(r"""
# one StaticKD student (no ensemble): mean ± SD over the 10 seeds of the final recipe
full = seeds[seeds.code == "full"]
single = {c: (full[c].mean(), full[c].std(ddof=1)) for c in ["dev_macro", "dev_acc", "heldout_acc", "test_macro", "test_acc"]}
for c, (mu, sd) in single.items():
    NUM.add(f"single.{c}.mean", mu, 3); NUM.add(f"single.{c}.sd", sd, 3)
pd.DataFrame(single, index=["mean", "sd"])
"""),
code(r"""
for m, r in main_table.iterrows():
    for s in ("dev", "heldout", "test"):
        if f"{s}_macro" in r and pd.notna(r.get(f"{s}_macro")):
            NUM.add(f"{m}.{s}.macro", r[f"{s}_macro"]); NUM.add(f"{m}.{s}.acc", r[f"{s}_acc"])
            NUM.add(f"{m}.{s}.ci", f"{fmt(r[f'{s}_lo'])}--{fmt(r[f'{s}_hi'])}")
    if pd.notna(r.get("test_lang_macro")):
        NUM.add(f"{m}.test.langmacro", r["test_lang_macro"])
        NUM.add(f"{m}.test.pct_teacher", r["test_vs_teacher"], 1, percent=True)
        NUM.add(f"{m}.dev.pct_teacher", r["dev_vs_teacher"], 1, percent=True)

def cell(r, s, bold=False):
    if pd.isna(r.get(f"{s}_macro")):
        return "--"
    v = f"{fmt(r[f'{s}_macro'])} [{fmt(r[f'{s}_lo'])}--{fmt(r[f'{s}_hi'])}]"
    return f"\\textbf{{{v}}}" if bold else v

lines = []
for m, r in main_table.iterrows():
    b = m == "statickd"
    name = an.MODELS[m][0].replace("StaticKD (proposed)", "\\textbf{StaticKD (proposed)}")
    agree = "--" if m == "teacher" or pd.isna(r.get("heldout_acc")) else fmt(r["heldout_acc"])
    vals = [name, cell(r, "dev", b), fmt(r["dev_acc"]), agree, cell(r, "test", b), fmt(r["test_acc"]),
            fmt(r["test_lang_macro"]) if pd.notna(r.get("test_lang_macro")) else "--",
            fmt(r["test_vs_teacher"], 1, percent=True) if pd.notna(r.get("test_vs_teacher")) else "--"]
    lines.append(" & ".join(vals) + " \\\\")
    if m in ("teacher_onnx_int8", "minilm_kd", "statickd_compact", "tfidf_svm_kd"):
        lines.append("\\midrule")
write_table("main", lines[:-1] if lines[-1] == "\\midrule" else lines)
"""),
md(r"""
### 8.2 Uji signifikansi berpasangan

StaticKD dibandingkan dengan setiap model lain pada dokumen yang sama. Di set uji dipakai *paired
site-cluster bootstrap*, dan di dev *paired document bootstrap*. Nilai $p$ dikoreksi Holm per set.
"""),
code(r"""
sig_rows = []
for s in ("dev", "test"):
    view = ev.eval_view(s)
    y = view.y.to_numpy(); clusters = view.cluster.to_numpy() if s == "test" else None
    p_s = ev.pred_view("statickd", s)
    block = []
    for m in [m for m in MAIN if m != "statickd" and ev.has_predictions(key(m), s)]:
        p_m = ev.pred_view(key(m), s)
        pb = ev.paired_bootstrap(y, p_s, p_m, clusters=clusters, n_boot=2 * N_BOOT)
        mc = ev.mcnemar(y, p_s, p_m)
        block.append({"set": s, "vs": m, **{f"f1_{k}": v for k, v in pb.items()}, "acc_delta": float((p_s == y).mean() - (p_m == y).mean()),
                      "only_static": mc["only_a"], "only_other": mc["only_b"], "p_mcnemar": mc["p"]})
    padj = ev.holm([b["f1_p"] for b in block]); madj = ev.holm([b["p_mcnemar"] for b in block])
    for b, p1, p2 in zip(block, padj, madj):
        b["p_boot_holm"], b["p_mcnemar_holm"] = p1, p2
    sig_rows += block
sig = pd.DataFrame(sig_rows)
for _, r in sig.iterrows():
    k = f"sig.{r['set']}.{r['vs']}"
    NUM.add(k + ".delta", r.f1_delta, 3, sign=True); NUM.add(k + ".ci", f"{fmt(r.f1_ci_low, 3, sign=True)}, {fmt(r.f1_ci_high, 3, sign=True)}")
    NUM.add(k + ".p", fmt_p(r.p_boot_holm)); NUM.add(k + ".pm", fmt_p(r.p_mcnemar_holm))
lines = []
for s in ("dev", "test"):
    part = sig[sig.set == s]
    for i, (_, r) in enumerate(part.iterrows()):
        first = f"\\multirow{{{len(part)}}}{{*}}{{{'dev' if s == 'dev' else 'test'}}}" if i == 0 else ""
        lines.append(" & ".join([first, an.MODELS[r.vs][0], f"{fmt(r.f1_delta, 3, sign=True)} [{fmt(r.f1_ci_low, 3, sign=True)}, {fmt(r.f1_ci_high, 3, sign=True)}]",
                                 fmt_p(r.p_boot_holm), fmt(r.acc_delta, 3, sign=True), f"{r.only_static}/{r.only_other}", fmt_p(r.p_mcnemar_holm)]) + " \\\\")
    if s == "dev":
        lines.append("\\midrule")
write_table("sig", lines)
show(sig.drop(columns=["f1_p", "p_mcnemar"]), "StaticKD vs model lain (Δ = StaticKD − model)")
"""),
md(r"""
### 8.3 Menurut asal label, wilayah, keragaman situs, dan data distilasi

Set uji menggabungkan tiga asal label. Jika StaticKD hanya bagus pada salah satunya, misalnya rubrik
JSON-LD yang paling mirip data distilasi CC-News, klaim akurasinya lemah. Tabel berikut memecah skor per
subkelompok. Nilai "% teacher" adalah macro-F1 model dibagi macro-F1 *teacher* pada subkelompok yang sama.
"""),
code(r"""
import re
test = ev.eval_set("test")
test = test.join(lang_info[["region", "script", "distil_tier", "diversity_tier", "distil_docs", "xlmr"]], on="lang")
GROUP_MODELS = [m for m in ["teacher", "distilmbert_kd", "minilm_kd", "statickd", "ft_kd_subword_q", "tfidf_svm_kd", "ft_gpt_subword"]
                if key(m) and ev.has_predictions(key(m), "test")]
group_tables = {}
for col in ["label_origin", "region", "distil_tier", "diversity_tier", "script", "xlmr"]:
    g = an.subgroup_scores([key(m) for m in GROUP_MODELS], col, test)
    g["model"] = g["model"].map({key(m): m for m in GROUP_MODELS})
    wide = g.pivot(index=col, columns="model", values="macro_f1")[GROUP_MODELS]
    wide.insert(0, "n", g.groupby(col).n.first())
    wide["StaticKD % teacher"] = wide["statickd"] / wide["teacher"]
    group_tables[col] = wide
    show(wide, f"Macro-F1 menurut {col}")
for col, wide in group_tables.items():
    for grp, r in wide.iterrows():
        k = f"grp.{col}.{re.sub('[^A-Za-z]', '', str(grp))[:16]}"
        for m in GROUP_MODELS:
            NUM.add(f"{k}.{m}", r[m])
        NUM.add(f"{k}.pct", r["StaticKD % teacher"], 1, percent=True); NUM.add(f"{k}.n", int(r["n"]), 0)
"""),
code(r"""
lines = []
order = [("label_origin", an.EN), ("distil_tier", an.EN), ("region", an.EN), ("xlmr", an.EN), ("diversity_tier", an.EN)]
titles = {"label_origin": "Label origin", "distil_tier": "Distillation data of the language", "region": "Region", "xlmr": "Teacher pre-training coverage", "diversity_tier": "Site diversity"}
for col, names in order:
    wide = group_tables[col]
    lines.append(f"\\multicolumn{{{len(GROUP_MODELS) + 3}}}{{@{{}}l}}{{\\textit{{{titles[col]}}}}} \\\\")
    for grp, r in wide.iterrows():
        label = names.get(grp, grp) if names else grp
        label = (label.replace("&", r"\&").replace("≥", r"$\ge$").replace("<", r"$<$").replace("–", "--")
                 .replace("pra-latih XLM-R", r"pra-latih XLM-R"))
        lines.append(" & ".join([f"\\quad {label}", fmt(int(r['n']), 0)] + [fmt(r[m]) for m in GROUP_MODELS] + [fmt(r['StaticKD % teacher'], 0, percent=True)]) + " \\\\")
GROUP_HEAD = {"teacher": r"\eng{Teacher}", "distilmbert_kd": "DistilmBERT", "minilm_kd": "MiniLM", "statickd": r"\textbf{StaticKD}",
              "ft_kd_subword_q": "fastText (q)", "tfidf_svm_kd": "TF-IDF+SVM", "ft_gpt_subword": "fastText GPT"}
# the whole tabular is generated, so that the header always matches the model columns
head = [r"\begin{tabular}{@{}lr" + "c" * (len(GROUP_MODELS) + 1) + "@{}}", r"\toprule",
        " & ".join(["Subgroup", "$n$"] + [GROUP_HEAD[m] for m in GROUP_MODELS] + [r"\% \eng{teacher}"]) + r" \\", r"\midrule"]
write_table("groups", head + lines + [r"\bottomrule", r"\end{tabular}"])

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, col in zip(axes, ["label_origin", "distil_tier"]):
    wide = group_tables[col]
    x = np.arange(len(wide)); w = 0.8 / len(GROUP_MODELS)
    for k, m in enumerate(GROUP_MODELS):
        ax.bar(x + (k - len(GROUP_MODELS) / 2 + 0.5) * w, wide[m], w, color=viz.color(m), label=an.MODELS[m][0])
    ax.set_xticks(x); ax.set_xticklabels([an.en(i) for i in wide.index], fontsize=8)
    ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1)
axes[0].set_title("By label origin"); axes[1].set_title("By amount of distillation data of the language")
axes[1].legend(fontsize=7, loc="upper left", ncol=2)
viz.show_and_save(fig, "fig_groups")
"""),
md(r"""
**Menurut sumber set uji.** MN-DS adalah satu-satunya sumber yang dianotasi manusia langsung dengan topik
IPTC. MasakhaNEWS juga berlabel manusia, tetapi dengan skemanya sendiri yang dipetakan ke IPTC. Tabel berikut
melaporkan setiap sumber terpisah.
"""),
code(r"""
SRC_MODELS = [m for m in ["teacher", "distilmbert_kd", "minilm_kd", "statickd", "ft_kd_subword_q", "tfidf_svm_kd"] if m in GROUP_MODELS]
g = an.subgroup_scores([key(m) for m in SRC_MODELS], "source", test)
g["model"] = g["model"].map({key(m): m for m in SRC_MODELS})
by_src = g.pivot(index="source", columns="model", values="macro_f1")[SRC_MODELS]
by_src.insert(0, "n", g.groupby("source").n.first())
by_src.insert(1, "langs", test.groupby("source").lang.nunique())
by_src["pct"] = by_src["statickd"] / by_src["teacher"]
by_src = by_src.sort_values("n", ascending=False)
origin_of = test.groupby("source").label_origin.first()
for s, r in by_src.iterrows():
    for m in SRC_MODELS:
        NUM.add(f"src.{s}.{m}", r[m])
    NUM.add(f"src.{s}.pct", r["pct"], 1, percent=True); NUM.add(f"src.{s}.n", int(r["n"]), 0); NUM.add(f"src.{s}.langs", int(r["langs"]), 0)
SRC_HEAD = {"teacher": "Teacher", "distilmbert_kd": "DistilmBERT", "minilm_kd": "MiniLM", "statickd": r"\textbf{StaticKD}",
            "ft_kd_subword_q": "fastText (q)", "tfidf_svm_kd": "TF-IDF+SVM"}
head = [r"\begin{tabular}{@{}llrr" + "c" * (len(SRC_MODELS) + 1) + "@{}}", r"\toprule",
        " & ".join(["Source", "Label origin", "$n$", "Lang."] + [SRC_HEAD[m] for m in SRC_MODELS] + [r"\% teacher"]) + r" \\", r"\midrule"]
lines = [" & ".join([SOURCE_TEX[s], an.en(origin_of[s]), fmt(int(r["n"]), 0), fmt(int(r["langs"]), 0)] + [fmt(r[m]) for m in SRC_MODELS]
                    + [fmt(r["pct"], 0, percent=True)]) + r" \\" for s, r in by_src.iterrows()]
write_table("sources", head + lines + [r"\bottomrule", r"\end{tabular}"])
show(by_src, "Macro-F1 per sumber set uji")
"""),
md(r"""
### 8.4 Uji sensitivitas: batas 5 artikel per situs

Sebagai pembanding aturan sampling, skor dihitung ulang pada subset dengan maksimal 5 artikel per situs di
setiap sel bahasa × label (aturan versi awal set uji). Jika peringkat model tidak berubah, kesimpulan tidak
bergantung pada aturan sampling.
"""),
code(r"""
cap_idx = an.cap_per_site(test, 5)
sens = []
for m in GROUP_MODELS:
    y, p = test.y.to_numpy(), ev.load_pred(key(m), "test")
    sens.append({"model": m, "full_macro": ev.macro_f1(y, p), "cap5_macro": ev.macro_f1(y[cap_idx], p[cap_idx]),
                 "full_acc": (y == p).mean(), "cap5_acc": (y[cap_idx] == p[cap_idx]).mean()})
sens = pd.DataFrame(sens).set_index("model")
sens["rank_full"] = sens.full_macro.rank(ascending=False).astype(int); sens["rank_cap5"] = sens.cap5_macro.rank(ascending=False).astype(int)
NUM.add("sens.n", len(cap_idx), 0)
NUM.add("sens.rank_same", "unchanged" if (sens.rank_full == sens.rank_cap5).all() else "changed")
NUM.add("sens.statickd", sens.loc["statickd", "cap5_macro"]); NUM.add("sens.teacher", sens.loc["teacher", "cap5_macro"])
show(sens, f"Sensitivitas: set uji penuh vs subset maks. 5 artikel/situs/sel ({len(cap_idx)} artikel)")
"""),
md(r"""
### 8.5 Validitas label silver

Pemeriksaan tidak langsung terhadap label silver (kategori penerbit dan rubrik JSON-LD):
1. **Bahasa Inggris** memiliki label manusia (MN-DS, MasakhaNEWS) sekaligus label silver (CC-NEWS JSON-LD).
   Jika label silver jauh lebih bising, akurasi *teacher* pada label silver akan jauh lebih rendah daripada
   pada label manusia untuk bahasa yang sama.
2. Akurasi *teacher* per asal label untuk seluruh bahasa.
"""),
code(r"""
t_pred = ev.load_pred("teacher", "test")
val_rows = []
for (lang, origin), g in test[test.lang.isin(["en", "fr"])].groupby(["lang", "label_origin"]):
    idx = g.index.to_numpy()
    val_rows.append({"bahasa": lang, "asal label": an.ORIGIN_NAMES[origin], "n": len(g), "label": g.label.nunique(),
                     "teacher macro-F1": ev.macro_f1(g.y, t_pred[idx]), "teacher akurasi": (g.y.to_numpy() == t_pred[idx]).mean()})
validity = pd.DataFrame(val_rows)
for _, r in validity.iterrows():
    NUM.add(f"valid.{r['bahasa']}.{r['asal label'].split()[0]}.acc", r["teacher akurasi"])
    NUM.add(f"valid.{r['bahasa']}.{r['asal label'].split()[0]}.n", int(r["n"]), 0)
show(validity, "Akurasi teacher per asal label (bahasa dengan label manusia dan silver)")
"""),
md(r"""
Semua sel bahasa Inggris di set uji terisi sumber berlabel manusia (MN-DS, MasakhaNEWS), karena sumber
manusia diprioritaskan saat sampling. Untuk perbandingan dalam bahasa yang sama, diambil sampel artikel
CC-NEWS berbahasa Inggris berlabel rubrik JSON-LD dari *pool* yang **tidak** masuk set uji (maksimal 50 per
label, `scripts/validity_silver.py`). Sampel itu dinilai oleh *teacher* dan StaticKD dengan cara yang sama.
"""),
code(r"""
if RUN_EVAL_TEACHER:
    run_script("validity_silver.py")
from scipy import stats as sstats
v_en = pd.read_json(data.TESTSET_DIR / "validity_en_silver.jsonl", lines=True)
en = test[test.lang == "en"]
tp_s, tp_h = ev.load_pred("teacher", "valid_en_silver"), ev.load_pred("teacher", "test")[en.index]
sp_s, sp_h = ev.load_pred("statickd", "valid_en_silver"), ev.load_pred("statickd", "test")[en.index]
per_label_valid = pd.DataFrame({
    "silver (rubrik)": pd.Series(tp_s == v_en.label_id.values).groupby(v_en.label.values).mean(),
    "manusia (MN-DS/MasakhaNEWS)": pd.Series(tp_h == en.y.values).groupby(en.label.values).mean()})
per_label_valid.index = [SHORT_LABELS[LABELS.index(i)] for i in per_label_valid.index]
wil = sstats.wilcoxon(per_label_valid.iloc[:, 0], per_label_valid.iloc[:, 1])
NUM.add("valid.en.silver.n", len(v_en), 0); NUM.add("valid.en.silver.sites", v_en.domain.nunique(), 0)
NUM.add("valid.en.silver.acc", float((tp_s == v_en.label_id).mean())); NUM.add("valid.en.silver.macro", ev.macro_f1(v_en.label_id, tp_s))
NUM.add("valid.en.human.acc", float((tp_h == en.y).mean())); NUM.add("valid.en.human.macro", ev.macro_f1(en.y, tp_h))
NUM.add("valid.en.silver.static_acc", float((sp_s == v_en.label_id).mean())); NUM.add("valid.en.human.static_acc", float((sp_h == en.y).mean()))
NUM.add("valid.en.labels_higher", int((per_label_valid.iloc[:, 0] > per_label_valid.iloc[:, 1]).sum()), 0)
NUM.add("valid.en.wilcoxon.p", fmt_p(wil.pvalue))
rho_lab = sstats.spearmanr(per_label_valid.iloc[:, 0], per_label_valid.iloc[:, 1])
NUM.add("valid.en.label_rho", rho_lab.statistic, 2); NUM.add("valid.en.label_rho.p", fmt_p(rho_lab.pvalue))
print(f"teacher, bahasa Inggris: akurasi label silver {np.mean(tp_s == v_en.label_id):.3f} ({len(v_en)} artikel, {v_en.domain.nunique()} situs) "
      f"vs label manusia {np.mean(tp_h == en.y):.3f} ({len(en)} artikel); Wilcoxon per label p={wil.pvalue:.3f}; "
      f"korelasi Spearman per label rho={rho_lab.statistic:.2f}")
show(per_label_valid, "Akurasi teacher per label, bahasa Inggris: label silver vs label manusia")
"""),
md(r"""
**Pemetaan per kategori sumber.** Untuk setiap kategori asli sebuah sumber (misalnya MasakhaNEWS
*politics*), dihitung porsi artikel yang diprediksi *teacher* ke label IPTC hasil pemetaan, beserta label
alternatif yang paling sering diprediksi. Hanya bahasa yang tercakup pra-latih XLM-R yang dihitung, supaya
kegagalan bahasa tidak tercampur dengan ketidakcocokan skema. Kategori dengan kesepakatan rendah
menandakan skema sumber lebih luas daripada topik IPTC tujuannya.
"""),
code(r"""
covered = test[test.lang.map(lambda l: l in an.XLMR_LANGS)]
tp = ev.load_pred("teacher", "test")
map_rows = []
for (src, sl), g in covered.groupby(["source", "source_label"]):
    if len(g) < 40:
        continue
    pred = pd.Series([LABELS[i] for i in tp[g.index]])
    alt = pred[pred != g.label.iloc[0]].value_counts()
    map_rows.append({"sumber": src, "kategori asli": sl, "label IPTC": SHORT_LABELS[LABELS.index(g.label.iloc[0])], "n": len(g),
                     "teacher setuju": float((pred.values == g.label.values).mean()),
                     "alternatif utama": f"{SHORT_LABELS[LABELS.index(alt.index[0])]} ({alt.iloc[0] / len(g):.0%})" if len(alt) else "-"})
mapping_check = pd.DataFrame(map_rows).sort_values("teacher setuju")
for src in ["masakhanews", "mnds", "l3cube_indic", "ccnews_jsonld"]:
    sub = mapping_check[mapping_check.sumber == src]
    if len(sub):
        NUM.add(f"mapcheck.{src}.median", sub["teacher setuju"].median(), 2)
m_pol = mapping_check[(mapping_check.sumber == "masakhanews") & (mapping_check["kategori asli"] == "politics")]
if len(m_pol):
    NUM.add("mapcheck.masakha_politics", m_pol["teacher setuju"].iloc[0], 2); NUM.add("mapcheck.masakha_politics.alt", m_pol["alternatif utama"].iloc[0].replace("%", "\\%"))
show(mapping_check.head(15), "Kategori sumber dengan kesepakatan teacher terendah (bahasa tercakup XLM-R)")
"""),
md(r"""
<a id="9"></a>
## 9. Ablasi, *temperature*, dan *ensemble* (RQ3)

### 9.1 Ablasi komponen

Setiap baris mengubah satu faktor terhadap resep final (satu *seed*, tanpa *ensemble*). Skor adalah
rata-rata ± simpangan baku antar-*seed*. Nilai $p$ berasal dari uji Welch terhadap 10 *seed* resep final,
dikoreksi Holm.
"""),
code(r"""
agg = seeds.groupby("code").agg(n=("seed", "size"), **{f"{c}_{f}": (c, f) for c in ["dev_macro", "heldout_acc", "test_macro", "test_acc"] for f in ["mean", "std"]})
agg = agg.reindex([c for c in ex.ABLATIONS if c in agg.index])
agg.insert(0, "varian", [ex.ABLATIONS[c][0] for c in agg.index])
base = seeds[seeds.code == "full"]
tests = {}
for c in agg.index.drop("full"):
    other = seeds[seeds.code == c]
    tests[c] = {f"p_{metric}": ev.welch(other[metric], base[metric])["p"] for metric in ["dev_macro", "heldout_acc", "test_macro"]}
tests = pd.DataFrame(tests).T
for metric in ["dev_macro", "heldout_acc", "test_macro"]:
    tests[f"p_{metric}_holm"] = ev.holm(tests[f"p_{metric}"])
ablation = agg.join(tests)
for c, r in ablation.iterrows():
    for metric in ["dev_macro", "heldout_acc", "test_macro", "test_acc"]:
        NUM.add(f"abl.{c}.{metric}", r[f"{metric}_mean"]); NUM.add(f"abl.{c}.{metric}.sd", r[f"{metric}_std"])
        if c != "full" and metric != "test_acc":
            NUM.add(f"abl.{c}.{metric}.delta", r[f"{metric}_mean"] - ablation.loc["full", f"{metric}_mean"], 3, sign=True)
            NUM.add(f"abl.{c}.{metric}.p", fmt_p(r[f"p_{metric}_holm"]))
show(ablation, "Ablasi (rata-rata ± SD antar-seed; p Welch vs resep final, Holm)")
"""),
code(r"""
ABL_TEX = {  # variant descriptions for the paper (English terms in \eng{})
    "full": r"Final recipe (one \eng{seed})",
    "hard": r"Hard \eng{teacher} labels (\eng{cross-entropy})",
    "emm_only": r"EMMediaTopic only (4 languages, 20k)",
    "data130k": r"130k documents (no extra CC-News)",
    "random_init": r"Randomly initialized embedding",
    "frozen": r"Frozen embedding (linear head only)",
    "no_tdrop": r"No \eng{token dropout}",
    "mlp": r"512-unit MLP head (cannot be folded)",
    "tau0.5": r"$\tau=0.5$", "tau2": r"$\tau=2$", "tau4": r"$\tau=4$",
    "bigram": r"+ bigram \eng{hashing} $2^{21}$", "idf": r"+ idf$^{0.5}$-weighted \eng{pooling}",
    "conf": r"+ document weight = \eng{teacher} confidence", "gpt_ce": r"+ \eng{cross-entropy} on GPT-4o labels",
}

def pm(r, metric):
    return f"{fmt(r[metric + '_mean'])}$\\pm${fmt(r[metric + '_std'])}"

def star(r, metric):
    if r.name == "full":
        return ""
    p = r[f"p_{metric}_holm"]
    return "$^{*}$" if p < 0.05 else ""

groups = [("Final recipe", ["full"]), ("Component ablation", ["hard", "emm_only", "data130k", "random_init", "frozen", "no_tdrop", "mlp"]),
          ("Temperature", ["tau0.5", "tau2", "tau4"]), ("Negative results (exploration)", ["bigram", "idf", "conf", "gpt_ce"])]
lines = []
for title, codes in groups:
    lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
    for c in [c for c in codes if c in ablation.index]:
        r = ablation.loc[c]
        lines.append(" & ".join([f"\\quad {ABL_TEX[c]}", str(int(r['n'])), pm(r, 'dev_macro') + star(r, 'dev_macro'),
                                 pm(r, 'heldout_acc') + star(r, 'heldout_acc'), pm(r, 'test_macro') + star(r, 'test_macro'),
                                 fmt(r['test_acc_mean'])]) + " \\\\")
write_table("ablation", lines)
"""),
md(r"""
**Ablasi per subkelompok bahasa.** Agregat set uji dapat menyembunyikan efek yang berlawanan arah di
bahasa yang berbeda. Tabel berikut menghitung macro-F1 setiap varian (rata-rata antar-*seed*) secara
terpisah untuk bahasa dengan data distilasi cukup, sedikit, dan tidak ada.
"""),
code(r"""
ABL_GROUP_CODES = ["full", "frozen", "random_init", "emm_only", "hard", "mlp", "tau2"]
y_all, tiers = test.y.to_numpy(), test.distil_tier.to_numpy()
abl_group = {}
for c in ABL_GROUP_CODES:
    seeds_c = [s for s in ex.ABLATIONS[c][2] if ev.has_predictions(ex.run_name(c, s), "test")]
    per_seed = []
    for s in seeds_c:
        p = ev.load_pred(ex.run_name(c, s), "test")
        per_seed.append({t: ev.macro_f1(y_all[tiers == t], p[tiers == t]) for t in np.unique(tiers)} | {"dev": ev.macro_f1(ev.eval_view("dev").y, ev.pred_view(ex.run_name(c, s), "dev"))})
    abl_group[c] = pd.DataFrame(per_seed).mean()
abl_group = pd.DataFrame(abl_group).T[["dev", "cukup (≥500)", "sedikit (<500)", "tidak ada (0)"]]
for c, r in abl_group.iterrows():
    for col, k in [("dev", "dev"), ("cukup (≥500)", "cukup"), ("sedikit (<500)", "sedikit"), ("tidak ada (0)", "tidakada")]:
        NUM.add(f"ablgrp.{c}.{k}", r[col])
        if c != "full":
            NUM.add(f"ablgrp.{c}.{k}.delta", r[col] - abl_group.loc["full", col], 3, sign=True)
lines = [" & ".join([ABL_TEX[c]] + [fmt(v) for v in r.values]) + " \\\\" for c, r in abl_group.iterrows()]
write_table("ablation_groups", lines)
show(abl_group, "Macro-F1 per varian menurut data distilasi bahasa (rata-rata seed)")
"""),
md(r"""
### 9.2 *Temperature*

*Student* ini hanya punya satu lapisan linear di atas rata-rata embedding, sehingga kapasitasnya sangat
rendah. Literatur *capacity gap* memprediksi bahwa *student* seperti ini tidak mampu meniru ekor distribusi
*teacher* yang dilunakkan. Gambar berikut menunjukkan skor tiap *seed* terhadap τ.
"""),
code(r"""
tau_codes = {"tau0.5": 0.5, "full": 1.0, "tau2": 2.0, "tau4": 4.0}
tau_df = seeds[seeds.code.isin(tau_codes)].assign(tau=lambda d: d.code.map(tau_codes))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
for ax, (metric, title) in zip(axes, [("dev_macro", "dev macro-F1"), ("heldout_acc", "held-out agreement"), ("test_macro", "test macro-F1")]):
    ax.scatter(tau_df.tau, tau_df[metric], color=viz.COLORS["statickd"], alpha=0.5, s=14)
    mean = tau_df.groupby("tau")[metric].mean()
    ax.plot(mean.index, mean.values, color=viz.COLORS["statickd"], marker="o")
    ax.set_xscale("log", base=2); ax.set_xticks(list(tau_codes.values())); ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.set_xlabel("τ"); ax.set_title(title)
viz.show_and_save(fig, "fig_temperature")
tau_tab = tau_df.groupby("tau")[["dev_macro", "heldout_acc", "test_macro"]].agg(["mean", "std"])
tau_tab
"""),
md(r"""
### 9.3 Ukuran *ensemble*

*Ensemble* $K$ *seed* dihitung dengan merata-ratakan logit tersimpan. Hasilnya setara dengan merata-ratakan
tabel, hingga pembulatan fp16. Untuk setiap $K$ diambil sampai 30 kombinasi acak dari 10 *seed* resep
final. Model final memakai $K=3$ (*seed* 0–2), yang ditetapkan sebelum set uji dibangun.
"""),
code(r"""
import itertools
rng_e = np.random.default_rng(0)
ens_rows = []
logits_cache = {(s, sn): ev.logits_view(ex.run_name("full", s), sn) for s in ex.MAIN_SEEDS for sn in ("dev", "heldout", "test")}
for k in (1, 2, 3, 5, 10):
    combos = list(itertools.combinations(ex.MAIN_SEEDS, k))
    if len(combos) > 30:
        combos = [combos[i] for i in rng_e.choice(len(combos), 30, replace=False)]
    for combo in combos:
        row = {"k": k, "seeds": combo}
        for sn in ("dev", "heldout", "test"):
            y = ev.eval_view(sn).y.to_numpy()
            pred = np.mean([logits_cache[(s, sn)] for s in combo], axis=0).argmax(1)
            row[f"{sn}_macro"], row[f"{sn}_acc"] = ev.macro_f1(y, pred), float((y == pred).mean())
        ens_rows.append(row)
ens = pd.DataFrame(ens_rows)
ens_tab = ens.groupby("k")[["dev_macro", "heldout_acc", "test_macro", "test_acc"]].agg(["mean", "std"])
for k, r in ens_tab.iterrows():
    for metric in ["dev_macro", "heldout_acc", "test_macro"]:
        NUM.add(f"ens.k{k}.{metric}", r[(metric, "mean")])
show(ens_tab, "Ukuran ensemble (rata-rata ± SD atas kombinasi seed)")
fig, ax = plt.subplots(figsize=(4.8, 3))
for metric, lab in [("dev_macro", "dev macro-F1"), ("heldout_acc", "held-out agreement"), ("test_macro", "test macro-F1")]:
    m = ens.groupby("k")[metric].mean(); s = ens.groupby("k")[metric].std().fillna(0)
    ax.errorbar(m.index, m.values, yerr=s.values, marker="o", capsize=3, label=lab)
ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 10]); ax.set_xticklabels([1, 2, 3, 5, 10])
ax.set_xlabel("number of seeds in the ensemble (K)"); ax.legend(fontsize=7)
viz.show_and_save(fig, "fig_ensemble_size")
"""),
]
