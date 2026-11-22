"""Review round 3: facts and re-analyses of stored data and predictions (no training).

Items (see results/review_experiments/r3_facts.md):
  1 site overlap between test articles and the distillation / held-out data, and the test scores without them
  2 per-source test macro-F1 with 95% site-cluster (and document) bootstrap intervals
  3 TOST margin sensitivity (+-0.005, +-0.01, +-0.02) for every equivalence claim
  4 MLP-head ablation held-out agreement gain (read from paper_numbers.json)
  6 composition of the 285,999 distillation documents
  7 external dataset category -> IPTC mapping as applied by the code
  8 StaticKD search budget (v1 exploration + v2 registry)
  9 test-set languages, names and regions
 10 released manifest content per source

Run from the project root:  env/Scripts/python.exe scripts/review_experiments/r3_reviewer_checks.py
Output: results/review_experiments/r3_reviewer_checks.json
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from statickd import analysis as an  # noqa: E402
from statickd import evaluation as ev  # noqa: E402
from statickd import experiments as ex  # noqa: E402
from statickd.data import load_ccnews_heldout, load_distillation_pool, softmax  # noqa: E402

OUT = ROOT / "results" / "review_experiments" / "r3_reviewer_checks.json"
N_BOOT = 1000  # as in the notebook (part1.py): 1000 for CIs, 2 * N_BOOT for paired tests
MARGINS = (0.005, 0.01, 0.02)
FOCUS = {"teacher": "teacher", "statickd": "statickd", "model2vec_clf": "model2vec_clf", "minilm_kd": "minilm_kd_t1"}
MAIN = ["teacher", "teacher_onnx_int8", "distilmbert_kd", "minilm_kd", "statickd", "statickd_compact",
        "ft_kd_subword_q", "ft_kd_subword", "ft_kd_word", "tfidf_svm_kd", "model2vec_clf", "sdstyle_small", "sdstyle_large",
        "ft_gpt_subword", "ft_gpt_word"]
KEYS = {"minilm_kd": "minilm_kd_t1", "sdstyle_small": "sdstyle_small_s0", "sdstyle_large": "sdstyle_large_s0"}
LANG_NAMES = {
    "am": "Amharic", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "ca": "Catalan", "cs": "Czech", "de": "German",
    "el": "Greek", "en": "English", "es": "Spanish", "fa": "Persian", "fr": "French", "gu": "Gujarati", "ha": "Hausa",
    "hi": "Hindi", "id": "Indonesian", "ig": "Igbo", "it": "Italian", "kn": "Kannada", "ko": "Korean", "lg": "Luganda",
    "ln": "Lingala", "ml": "Malayalam", "mr": "Marathi", "nl": "Dutch", "om": "Oromo", "or": "Odia", "pa": "Punjabi",
    "pcm": "Nigerian Pidgin", "pl": "Polish", "pt": "Portuguese", "rn": "Kirundi", "ro": "Romanian", "ru": "Russian",
    "sn": "Shona", "so": "Somali", "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai", "ti": "Tigrinya",
    "tr": "Turkish", "uk": "Ukrainian", "vi": "Vietnamese", "xh": "isiXhosa", "yo": "Yoruba", "zh": "Chinese"}
SECOND_LEVEL = {"co", "com", "org", "net", "gov", "ac", "edu", "or", "ne", "go", "gob", "nic", "info"}


def key(m: str) -> str:
    return KEYS.get(m, m)


def host(d: str) -> str:
    d = (d or "").lower().strip()
    return d[4:] if d.startswith("www.") else d


def reg_domain(h: str) -> str:
    """Rough registrable domain (no public-suffix list available): last two labels, three for co.uk-style."""
    parts = h.split(".")
    if len(parts) >= 3 and parts[-2] in SECOND_LEVEL and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ============================================================================ item 1: site overlap
def item1(test: pd.DataFrame) -> dict:
    pool = load_distillation_pool()
    cc = pool[pool.source != "emmediatopic"]
    ho = load_ccnews_heldout()
    train_hosts, ho_hosts = set(cc.domain.map(host)), set(ho.domain.map(host))
    train_pairs = set(zip(cc.domain.map(host), cc.lang))
    ho_pairs = set(zip(ho.domain.map(host), ho.lang))
    train_reg, ho_reg = {reg_domain(h) for h in train_hosts}, {reg_domain(h) for h in ho_hosts}
    urls = set(pd.read_parquet(ROOT / "data/processed/teacher_labels/ccnews_train.parquet", columns=["url"]).url) \
        | set(pd.read_parquet(ROOT / "data/processed/teacher_labels/ccnews_train_extra.parquet", columns=["url"]).url) \
        | set(ho.url)

    t = test.copy()
    t["h"] = t.domain.map(host)
    known = t.h.ne("")
    t["known"] = known
    t["in_train"] = known & t.h.isin(train_hosts)
    t["in_heldout"] = known & t.h.isin(ho_hosts)
    t["in_any"] = t.in_train | t.in_heldout
    t["in_any_samelang"] = known & pd.Series([(h, l) in train_pairs or (h, l) in ho_pairs for h, l in zip(t.h, t.lang)], index=t.index)
    t["in_train_samelang"] = known & pd.Series([(h, l) in train_pairs for h, l in zip(t.h, t.lang)], index=t.index)
    t["in_any_reg"] = known & t.h.map(reg_domain).isin(train_reg | ho_reg)
    t["url_in_data"] = t.url.isin(urls) & t.url.notna() & t.url.ne("not available")

    by_source = {}
    for s, g in t.groupby("source"):
        by_source[s] = {
            "n": len(g), "domain_known": int(g.known.sum()),
            "domain_known_note": None,
            "in_distillation_train_exact": int(g.in_train.sum()), "in_heldout_exact": int(g.in_heldout.sum()),
            "in_distillation_or_heldout_exact": int(g.in_any.sum()),
            "same_site_and_same_language_in_distillation": int(g.in_train_samelang.sum()),
            "same_site_and_same_language_in_distillation_or_heldout": int(g.in_any_samelang.sum()),
            "in_distillation_or_heldout_registrable_domain": int(g.in_any_reg.sum()),
            "url_exactly_in_distillation_or_heldout": int(g.url_in_data.sum()),
            "overlapping_sites": g[g.in_any].h.value_counts().to_dict(),
            "overlapping_sites_registrable_only": g[g.in_any_reg & ~g.in_any].h.value_counts().head(20).to_dict(),
        }
    notes = {"ccnews_jsonld": "URL from the WARC record; filter 1 of the test-set construction already removed every "
                              "site of ccnews_train/extra/heldout (testset/jsonld.py excluded_domains, exact host +- www.)",
             "masakhanews": "URL field of the dataset; 295 articles (xh 200, en 95) have url = 'not available'",
             "mnds": "URL field of the dataset (feedproxy.google.com = 45 feed-proxy URLs, real site unknown)",
             "l3cube_indic": "no URL or site in the dataset (CSV with text and label only)",
             "bangla_plos": "no URL or site in the dataset", "indonews": "no URL or site in the dataset"}
    for s, n in notes.items():
        if s in by_source:
            by_source[s]["domain_known_note"] = n
    # MasakhaNEWS English rows with a relative URL ("/sport/...", "/news/...") carry BBC-style paths but no host
    t["relative_url"] = (t.source == "masakhanews") & t.url.fillna("").str.startswith("/")
    bbc = t[t.h == "bbc.com"]
    lang_detail = {l: {"test": int((bbc.lang == l).sum()),
                       "bbc.com docs in distillation (same lang)": int(((cc.domain.map(host) == "bbc.com") & (cc.lang == l)).sum()),
                       "bbc.com docs in heldout (same lang)": int(((ho.domain.map(host) == "bbc.com") & (ho.lang == l)).sum())}
                   for l in sorted(bbc.lang.unique())}
    summary = {
        "n_test": len(t), "domain_known": int(known.sum()), "domain_unknown": int((~known).sum()),
        "in_distillation_or_heldout_exact": int(t.in_any.sum()), "share_exact": float(t.in_any.mean()),
        "in_distillation_train_exact": int(t.in_train.sum()),
        "same_site_same_language_any": int(t.in_any_samelang.sum()),
        "in_any_registrable_domain": int(t.in_any_reg.sum()), "share_registrable": float(t.in_any_reg.mean()),
        "url_exact_matches": int(t.url_in_data.sum()),
        "emmediatopic_note": "EMMediaTopic train documents have no URL/site (CLASSLA-web document_id only); overlap "
                             "with them cannot be checked. Test languages shared with EMMediaTopic: "
                             + ", ".join(f"{l}={int((t.lang == l).sum())}" for l in ("ca", "el", "hr", "sl") if (t.lang == l).any()),
        "bbc_com_by_language": lang_detail,
        "train_overlap_8gram_bbc_test_max": float(bbc.train_overlap.max()) if len(bbc) else None,
        "train_overlap_8gram_bbc_test_mean": float(bbc.train_overlap.mean()) if len(bbc) else None,
        "masakhanews_relative_url_rows": {l: int(n) for l, n in t[t.relative_url].lang.value_counts().items()},
        "masakhanews_url_not_available_rows": {l: int(n) for l, n in t[(t.source == "masakhanews") & (t.url == "not available")].lang.value_counts().items()},
    }
    # scores on the overlapping articles themselves (does the overlap favour StaticKD?)
    # (macro-F1 over a mixed subset is misleading: the 30 MN-DS rows add many rare labels, so MasakhaNEWS-only here)
    masak = t.source == "masakhanews"
    for name, mask in (("scores_masakhanews_bbc_overlap", t.in_any & masak),
                       ("scores_masakhanews_other_sites", ~t.in_any & masak),
                       ("scores_masakhanews_same_site_same_language", t.in_any_samelang & masak)):
        sub = t[mask]
        y = sub.y.to_numpy()
        summary[name] = {"n": int(len(sub))}
        for m, k in FOCUS.items():
            p = ev.load_pred(k, "test")[sub.index.to_numpy()]
            summary[name][m] = {"macro_f1": ev.macro_f1(y, p), "accuracy": float((y == p).mean())}

    # ---- re-score without the overlapping articles
    def rescore(mask: np.ndarray, label: str) -> dict:
        sub = t[mask]
        idx = sub.index.to_numpy()
        y = sub.y.to_numpy()
        cl = sub.cluster.to_numpy()
        models = [m for m in MAIN if ev.has_predictions(key(m), "test")]
        preds = {m: ev.load_pred(key(m), "test")[idx] for m in models}
        scores = {m: ev.macro_f1(y, preds[m]) for m in models}
        res = {"n": int(len(sub)), "n_clusters": int(len(np.unique(cl))), "effective_clusters": ev.effective_clusters(cl),
               "macro_f1": {m: scores[m] for m in models},
               "ranking": sorted(models, key=lambda m: -scores[m]), "focus": {}}
        comps = {}
        for m in [m for m in models if m != "statickd"]:
            pb = ev.paired_bootstrap_seeds(y, [preds["statickd"]], [preds[m]], clusters=cl, n_boot=2 * N_BOOT)
            comps[m] = {k: v for k, v in pb.items() if k != "diffs"}
            if m == "model2vec_clf":
                comps[m]["tost"] = {str(mg): ev.tost(pb["diffs"], mg) for mg in MARGINS}
        holm = ev.holm([comps[m]["p"] for m in comps])
        for m, ph in zip(comps, holm):
            comps[m]["p_holm_all_models"] = float(ph)
        res["statickd_minus"] = comps
        for m in FOCUS:
            lo, hi = ev.bootstrap_ci(y, preds[m], clusters=cl, n_boot=N_BOOT)
            res["focus"][m] = {"macro_f1": scores[m], "ci95": [lo, hi], "accuracy": float((y == preds[m]).mean())}
        print(f"[1] {label}: n={len(sub)}", {m: round(scores[m], 3) for m in FOCUS},
              {m: (round(comps[m]["delta"], 3), round(comps[m]["ci_low"], 3), round(comps[m]["ci_high"], 3), round(comps[m]["p_holm_all_models"], 3))
               for m in ("teacher", "model2vec_clf", "minilm_kd")}, flush=True)
        return res

    full = np.ones(len(t), bool)
    rescored = {
        "full_test": rescore(full, "full"),
        "without_exact_site_overlap": rescore(~t.in_any.to_numpy(), "no exact overlap"),
        "without_exact_overlap_and_relative_bbc_urls": rescore(~(t.in_any | t.relative_url).to_numpy(), "no overlap + rel"),
        "without_registrable_domain_overlap": rescore(~(t.in_any_reg | t.in_any).to_numpy(), "no reg overlap"),
        "without_overlap_and_without_unknown_sites": rescore((t.known & ~t.in_any_reg & ~t.in_any).to_numpy(), "verified unseen only"),
    }
    return {"summary": summary, "by_source": by_source, "rescored": rescored}


# ============================================================================ item 2: per-source CIs
def item2(test: pd.DataFrame) -> dict:
    out = {}
    for s, g in test.groupby("source"):
        idx, y, cl = g.index.to_numpy(), g.y.to_numpy(), g.cluster.to_numpy()
        row = {"n": len(g), "n_clusters": int(len(np.unique(cl))), "effective_clusters": ev.effective_clusters(cl), "models": {}}
        for m, k in FOCUS.items():
            p = ev.load_pred(k, "test")[idx]
            lo, hi = ev.bootstrap_ci(y, p, clusters=cl, n_boot=N_BOOT)
            dlo, dhi = ev.bootstrap_ci(y, p, clusters=None, n_boot=N_BOOT)
            row["models"][m] = {"macro_f1": ev.macro_f1(y, p), "ci95_site_cluster": [lo, hi], "ci95_document": [dlo, dhi]}
        out[s] = row
        print(f"[2] {s}", {m: round(v['macro_f1'], 3) for m, v in row["models"].items()}, flush=True)
    return out


# ============================================================================ item 3: TOST margins
def item3(test: pd.DataFrame, numbers: dict) -> dict:
    y, cl = test.y.to_numpy(), test.cluster.to_numpy()
    out = {}

    def record(name, diffs, delta, paper_key=None):
        res = {"delta": delta, "ci90": [float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))],
               "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
        for mg in MARGINS:
            tt = ev.tost(diffs, mg)
            res[f"margin_{mg}"] = {"equivalent": tt["equivalent"], "tost_p": tt["p"]}
        if paper_key:
            res["paper_ci90"] = numbers.get(paper_key)
        out[name] = res
        print(f"[3] {name}: d={delta:+.4f} ci90=[{res['ci90'][0]:+.4f},{res['ci90'][1]:+.4f}]",
              {m: res[f'margin_{m}']['equivalent'] for m in MARGINS}, flush=True)

    p_static = ev.pred_view("statickd", "test")
    for m in ("model2vec_clf", "distilmbert_kd", "sdstyle_small", "sdstyle_large"):
        pb = ev.paired_bootstrap_seeds(y, [p_static], [ev.pred_view(key(m), "test")], clusters=cl, n_boot=2 * N_BOOT)
        record(f"statickd_vs_{m}", pb["diffs"], pb["delta"], f"eqv.{m}.ci90" if m in ("model2vec_clf", "distilmbert_kd") else None)
    for m, p in {"minilm_kd": "minilm_kd_t1_len512", "distilmbert_kd": "distilmbert_kd_len512"}.items():
        pb = ev.paired_bootstrap_seeds(y, [p_static], [ev.pred_view(p, "test")], clusters=cl, n_boot=2 * N_BOOT)
        record(f"statickd_vs_{p}", pb["diffs"], pb["delta"])
    # cascade (theta chosen on dev with the notebook rule) vs the CPU teacher
    p_dev = softmax(ev.load_logits("statickd", "dev"))
    y_dev = ev.eval_view("dev").y.to_numpy()
    pt_dev = ev.load_pred("teacher_onnx_int8", "dev")
    f1_t_dev = ev.macro_f1(y_dev, pt_dev)
    grid = {th: ev.macro_f1(y_dev, np.where(p_dev.max(1) >= th, p_dev.argmax(1), pt_dev)) for th in (0.5, 0.6, 0.7, 0.8, 0.9)}
    reach = [th for th, f in grid.items() if f >= f1_t_dev]
    theta = reach[0] if reach else max(grid, key=lambda th: (grid[th], -th))
    ps = softmax(ev.load_logits("statickd", "test"))
    pred_c = np.where(ps.max(1) >= theta, ps.argmax(1), ev.load_pred("teacher_onnx_int8", "test"))
    pb = ev.paired_bootstrap_seeds(y, [pred_c], [ev.load_pred("teacher_onnx_int8", "test")], clusters=cl, n_boot=2 * N_BOOT, seed=0)
    record(f"cascade_theta{theta}_vs_teacher_onnx_int8", pb["diffs"], pb["delta"], "casc.eq.ci")
    out[f"cascade_theta{theta}_vs_teacher_onnx_int8"]["theta_dev"] = theta
    # ablations: seeds of the variant vs the 10 seeds of the final recipe
    pb_full = [ev.pred_view(ex.run_name("full", s), "test") for s in ex.MAIN_SEEDS]
    for c in [c for c in ex.ABLATIONS if c != "full"]:
        pa = [ev.pred_view(ex.run_name(c, s), "test") for s in ex.ABLATIONS[c][2]]
        r = ev.paired_bootstrap_seeds(y, pa, pb_full, clusters=cl, n_boot=2 * N_BOOT)
        record(f"ablation_{c}_vs_final", r["diffs"], r["delta"], f"abltost.{c}.ci90")
    return out


# ============================================================================ item 6: distillation data
def item6() -> dict:
    pool = load_distillation_pool()
    cc = pool[pool.source == "ccnews"]
    ccx = pool[pool.source == "ccnews_extra"]
    allcc = pool[pool.source != "emmediatopic"]
    return {
        "counts": pool.source.value_counts().to_dict(), "total": len(pool),
        "base_130k_variant_sources": list(ex.ABLATIONS["data130k"][1]["sources"]),
        "base_130k_total": int(len(pool[pool.source.isin(["emmediatopic", "ccnews"])])),
        "languages": {"all": int(pool.lang.nunique()), "ccnews": int(cc.lang.nunique()), "ccnews_extra": int(ccx.lang.nunique()),
                      "emmediatopic": sorted(pool[pool.source == "emmediatopic"].lang.unique().tolist())},
        "sites": {"ccnews_train_plus_extra": int(allcc.domain.nunique()), "ccnews_base": int(cc.domain.nunique()),
                  "extra_sites_not_in_base": int(len(set(ccx.domain) - set(cc.domain)))},
        "max_docs_per_site": {"ccnews_base": int(cc.domain.value_counts().max()),
                              "ccnews_train_plus_extra": int(allcc.domain.value_counts().max())},
        "max_docs_per_site_and_language": {"ccnews_base": int(cc.groupby(["domain", "lang"]).size().max()),
                                           "ccnews_train_plus_extra": int(allcc.groupby(["domain", "lang"]).size().max())},
        "extra_max_docs_per_language": int(ccx.lang.value_counts().max()),
        "median_words_all": float(pool.text.str.split().str.len().median()),
    }


# ============================================================================ item 7: external mapping
def item7(test: pd.DataFrame) -> dict:
    from statickd.testset.unified import iptc_label, load_external
    ext = load_external()
    ext["iptc"] = [iptc_label(s, l, g) for s, l, g in zip(ext.source, ext.source_label, ext.lang)]
    ntest = test[test.source != "ccnews_jsonld"].groupby(["source", "source_label"]).size()
    out = {}
    for (s, lab), g in ext.groupby(["source", "source_label"]):
        labels = sorted({str(x) for x in g.iptc})
        out.setdefault(s, {})[lab] = {"iptc": labels[0] if len(labels) == 1 else labels, "n_raw": len(g),
                                     "languages": sorted(g.lang.unique().tolist()),
                                     "n_test": int(ntest.get((s, lab), 0))}
    d = pd.read_csv(ROOT / "data/raw/testset_external/mnds/MN-DS-news-classification.csv")
    multi = d.groupby("id").category_level_1.nunique()
    multi_ids = set(multi[multi > 1].index)
    multi_urls = set(d[d.id.isin(multi_ids)].url)
    tm = test[test.source == "mnds"]
    out["_mnds_structure"] = {"rows": len(d), "unique_ids": int(d.id.nunique()), "unique_urls": int(d.url.nunique()),
                              "ids_with_more_than_one_level1": len(multi_ids),
                              "level1_categories": int(d.category_level_1.nunique()),
                              "level2_categories": int(d.category_level_2.nunique()),
                              "test_mnds_articles_whose_url_has_several_level1_rows": int(tm.url.isin(multi_urls).sum())}
    return out


# ============================================================================ item 8: search budget
def item8() -> dict:
    v1 = []
    for dpath in sorted(glob.glob(str(ROOT / "models/_archive_v1/static/*"))):
        n = os.path.basename(dpath)
        if "table" in n:
            continue
        m = json.load(open(Path(dpath) / "meta.json", encoding="utf8"))
        v1.append({"name": n, "sources": m.get("sources"), "epochs": m.get("epochs"), "hidden": m.get("hidden"),
                   "token_dropout": m.get("token_dropout"), "temperature": m.get("temperature"),
                   "gpt_weight": m.get("gpt_weight"), "conf_weight": m.get("conf_weight"), "bigram_bits": m.get("bigram_bits"),
                   "max_tokens": m.get("max_tokens"), "token_weight": m.get("token_weight"), "seed": m.get("seed"),
                   "pruned_min_count": m.get("pruned_min_count"),
                   "val_agreement": m.get("val_agreement_with_teacher")})
    trained = [r for r in v1 if r["pruned_min_count"] is None]
    phase2 = pd.read_csv(ROOT / "results/_archive_v1/exploration/phase2_comparison.csv")
    reg = {c: len(seeds) for c, (_, _, seeds) in ex.ABLATIONS.items()}
    return {"v1_exploration_runs": v1, "v1_trained_runs": len(trained), "v1_posthoc_pruned": len(v1) - len(trained),
            "v1_phase2_comparison": phase2.to_dict(orient="records"),
            "v2_registry_runs_per_code": reg, "v2_registry_codes": len(reg), "v2_registry_runs": sum(reg.values()),
            "v2_student_dirs": len(os.listdir(ROOT / "models/students"))}


# ============================================================================ item 9 / 10
def item9(test: pd.DataFrame) -> list:
    rows = []
    for l, g in test.groupby("lang"):
        rows.append({"code": l, "name": LANG_NAMES.get(l, "?"), "region_code": an.REGION.get(l, "?"),
                     "region_en": an.en(an.REGION.get(l, "?")), "script": an.en(an.SCRIPT.get(l, "?")), "n": len(g),
                     "sources": g.source.value_counts().to_dict()})
    return rows


def item10() -> dict:
    p = ROOT / "dist/static_kd_iptc_classifier/data/testset/unified_test_manifest.csv"
    m = pd.read_csv(p, dtype=str)
    out = {"columns": m.columns.tolist()}
    for s, g in m.groupby("source"):
        out[s] = {"n": len(g), "url": int(g.url.notna().sum()), "url_not_available": int((g.url == "not available").sum()),
                  "domain": int(g.domain.notna().sum()), "warc_file": int(g.warc_file.notna().sum()),
                  "source_row_id": int(g.source_row_id.notna().sum()), "text_sha1": int(g.text_sha1.notna().sum()),
                  "label": int(g.label.notna().sum()), "source_label": int(g.source_label.notna().sum()),
                  "example_source_row_id": g.source_row_id.dropna().iloc[0] if g.source_row_id.notna().any() else None}
    return out


def main() -> None:
    numbers = json.load(open(ROOT / "results/final/paper_numbers.json", encoding="utf8"))
    test = ev.eval_view("test")
    res = {}
    res["item1_site_overlap"] = item1(test)
    res["item2_per_source"] = item2(test)
    res["item3_tost_margins"] = item3(test, numbers)
    res["item4_mlp_head"] = {k: v for k, v in numbers.items() if k.startswith(("abl.mlp.", "ablboot.mlp.", "abltost.mlp."))}
    res["item6_distillation"] = item6()
    res["item7_mapping"] = item7(test)
    res["item8_search_budget"] = item8()
    res["item9_languages"] = item9(test)
    res["item10_manifest"] = item10()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(jsonable(res), indent=1, ensure_ascii=False), encoding="utf8")
    print("written", OUT)


if __name__ == "__main__":
    main()
