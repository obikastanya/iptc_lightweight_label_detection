"""Review round 5, part A: re-analyses of stored predictions (no training).

Items (see results/review_experiments/r5_facts.md):
  A1 source-stratified site-cluster bootstrap (CIs, paired tests with Holm, TOST of the eleven equivalence claims)
  A2 mean of per-source macro-F1 and restricted-inventory macro-F1 with the stratified bootstrap
  A3 ECE with bootstrap CIs (StaticKD, teacher; dev and test) and the paired difference
  A4 audited labels (300) and blind human labels (100): accuracies, gains, paired differences
  A5 StaticKD with input truncated to 512 tokens
  A6 teacher entropy / max probability on the distillation data
  A7 per-label precision/recall/F1 and confidence-threshold filtering
  A8 operating regime from the stored CPU benchmark
  A9 test-set manifest fields (release check, no code change)

Bootstrap conventions as r4 (notebook part1.py N_BOOT = 1000): CIs = first 1,000 resamples, paired tests 2,000
resamples, seed 0. Unstratified = the paper's site-cluster bootstrap (ev.bootstrap_samples). Stratified = clusters
are resampled with replacement WITHIN each of the six test sources (the six sources and their sizes in clusters stay
fixed); a source that has only one cluster (no site information: l3cube_indic, bangla_plos, indonews) is resampled
by documents within the source.

Run from the project root:  env/Scripts/python.exe scripts/review_experiments/r5_a_analyses.py
Output: results/review_experiments/r5_a.json, results/review_experiments/r5_statickd_len512/statickd_len512__<set>.npz
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "review_experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from statickd import analysis as an  # noqa: E402
from statickd import evaluation as ev  # noqa: E402
from statickd import experiments as ex  # noqa: E402
from statickd.config import LABELS, NUM_LABELS  # noqa: E402
from statickd.data import load_distillation_pool, softmax  # noqa: E402

from r4_reviewer_checks import (KEYS, MAIN, N_BOOT, N_PAIRED, MARGIN, acc, ci_from, jsonable,  # noqa: E402
                                paired_from, restricted_logits, wilson)

OUT_DIR = ROOT / "results" / "review_experiments"
OUT_JSON = OUT_DIR / "r5_a.json"
LEN512_DIR = OUT_DIR / "r5_statickd_len512"
ENS_DIR = OUT_DIR / "r4_frozen_ensemble"
FAMILY14 = [m for m in MAIN if m != "statickd"]          # family of the paper's significance table (Holm over 14)
EXTRA = {"minilm_kd_len512": "minilm_kd_t1_len512", "distilmbert_kd_len512": "distilmbert_kd_len512",
         "frozen_ens": None}                               # other rows of the main table
DISPLAY = {m: an.MODELS[m][0] for m in an.MODELS}
DISPLAY.update({"minilm_kd_len512": "MiniLM-L6 + KD, 512 tokens", "distilmbert_kd_len512": "DistilmBERT + KD, 512 tokens",
                "frozen_ens": "StaticKD, frozen embeddings (3-seed ens.)", "cascade": "Cascade (theta 0.8)"})


def key(m: str) -> str:
    return KEYS.get(m, m)


def test_logits(m: str) -> np.ndarray:
    if m == "frozen_ens":
        return np.load(ENS_DIR / "frozen_ens012__test.npz")["logits"]
    return ev.load_logits(EXTRA.get(m) or key(m), "test")


# ----------------------------------------------------------------------------- stratified sampler
def strat_groups(src: np.ndarray, clusters: np.ndarray) -> list:
    out = []
    for s in sorted(np.unique(src)):
        idx = np.flatnonzero(src == s)
        cl = clusters[idx]
        if len(np.unique(cl)) > 1:
            out.append((s, "clusters", [idx[g] for g in ev._cluster_index(cl)]))
        else:
            out.append((s, "documents", idx))
    return out


def strat_samples(groups, n_boot: int = N_PAIRED, seed: int = 0):
    """Yield a list of per-source index arrays per resample (sources fixed, clusters resampled within source)."""
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        parts = []
        for _, kind, g in groups:
            if kind == "clusters":
                pick = rng.integers(0, len(g), len(g))
                parts.append(np.concatenate([g[k] for k in pick]))
            else:
                parts.append(g[rng.integers(0, len(g), len(g))])
        yield parts


def p_two(diffs) -> float:
    return float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))


# ============================================================================ A1 + A2
def a1_a2(test: pd.DataFrame, numbers: dict, r4: dict) -> dict:
    y, src, cl = test.y.to_numpy(), test.source.to_numpy(), test.cluster.to_numpy()
    sources = sorted(np.unique(src))
    rows = list(MAIN) + list(EXTRA)
    L = {m: test_logits(m) for m in rows}
    # restricted-inventory mask (gold labels per source, as r4 item 1)
    mask = np.zeros((len(test), NUM_LABELS), bool)
    for s in sources:
        mask[np.ix_(src == s, np.unique(y[src == s]))] = True
    pu = {m: L[m].argmax(1) for m in rows}
    pr = {m: restricted_logits(L[m], mask).argmax(1) for m in rows}
    # eleven equivalence claims (r4 item 4 family A): cascade theta from dev, ablation seed runs
    p_dev = softmax(ev.load_logits("statickd", "dev"))
    y_dev = ev.eval_view("dev").y.to_numpy()
    pt_dev = ev.load_pred("teacher_onnx_int8", "dev")
    f1_t_dev = ev.macro_f1(y_dev, pt_dev)
    grid = {th: ev.macro_f1(y_dev, np.where(p_dev.max(1) >= th, p_dev.argmax(1), pt_dev)) for th in (0.5, 0.6, 0.7, 0.8, 0.9)}
    reach = [th for th, f in grid.items() if f >= f1_t_dev]
    theta = reach[0] if reach else max(grid, key=lambda th: (grid[th], -th))
    ps = softmax(ev.load_logits("statickd", "test"))
    cascade = np.where(ps.max(1) >= theta, ps.argmax(1), pu["teacher_onnx_int8"])
    claims = r4["item4"]["families"]["claimed_equivalences"]
    abl_codes = [k.removeprefix("ablation_") for k in claims if k.startswith("ablation_")]
    runs = {}
    for c in ["full", *abl_codes]:
        for s in ex.ABLATIONS[c][2]:
            runs[ex.run_name(c, s)] = ev.load_pred(ex.run_name(c, s), "test")

    pooled = {**{("u", m): pu[m] for m in rows}, **{("r", m): pr[m] for m in rows}, ("u", "cascade"): cascade,
              **{("run", k): v for k, v in runs.items()}}
    full_val = {k: ev.macro_f1(y, p) for k, p in pooled.items()}
    per_src_full = {m: {s: ev.macro_f1(y[src == s], pu[m][src == s]) for s in sources} for m in rows}
    for m in rows:
        full_val[("mps", m)] = float(np.mean(list(per_src_full[m].values())))

    res = {}
    groups = strat_groups(src, cl)
    res["strata"] = {s: {"kind": k, "n_units": len(g), "n": int((src == s).sum())} for s, k, g in groups}
    for cond in ("unstratified", "stratified"):
        t0 = time.time()
        vals = {k: np.empty(N_PAIRED) for k in pooled}
        if cond == "stratified":
            for m in rows:
                vals[("mps", m)] = np.empty(N_PAIRED)
            it = strat_samples(groups)
        else:
            it = ([i] for i in ev.bootstrap_samples(len(y), cl, N_PAIRED, 0))
        for b, parts in enumerate(it):
            i = np.concatenate(parts) if len(parts) > 1 else parts[0]
            yi = y[i]
            for k, p in pooled.items():
                vals[k][b] = ev.macro_f1(yi, p[i])
            if cond == "stratified":
                ys = [y[q] for q in parts]
                for m in rows:
                    vals[("mps", m)][b] = np.mean([ev.macro_f1(yq, pu[m][q]) for yq, q in zip(ys, parts)])
        print(f"[A1] {cond} bootstrap {time.time() - t0:.0f}s", flush=True)

        out = {"ci_macro_f1": {}, "ci_macro_f1_restricted": {}}
        for m in rows:
            out["ci_macro_f1"][m] = {"macro_f1": full_val[("u", m)], "ci95": ci_from(vals[("u", m)])}
            out["ci_macro_f1_restricted"][m] = {"macro_f1": full_val[("r", m)], "ci95": ci_from(vals[("r", m)])}
        # (b) paired, StaticKD - model
        for tag, kind in (("paired", "u"), ("paired_restricted", "r")):
            block = {m: paired_from([full_val[(kind, "statickd")]], [full_val[(kind, m)]], [vals[(kind, "statickd")]], [vals[(kind, m)]])
                     for m in FAMILY14}
            for b_, ph in zip(block.values(), ev.holm([b_["p"] for b_ in block.values()])):
                b_["p_holm14"] = float(ph)
            extra = {m: paired_from([full_val[(kind, "statickd")]], [full_val[(kind, m)]], [vals[(kind, "statickd")]], [vals[(kind, m)]])
                     for m in EXTRA}
            h2 = ev.holm([extra["minilm_kd_len512"]["p"], extra["distilmbert_kd_len512"]["p"]])
            extra["minilm_kd_len512"]["p_holm2"], extra["distilmbert_kd_len512"]["p_holm2"] = float(h2[0]), float(h2[1])
            out[tag] = {"family14": block, "extra": extra}
        # (c) TOST of the eleven equivalence claims
        full_runs = [ex.run_name("full", s) for s in ex.MAIN_SEEDS]
        tests = {}
        for k in claims:
            if k == "model2vec_clf_vs_statickd":
                a, b_ = [("u", "statickd")], [("u", "model2vec_clf")]
            elif k.startswith("cascade"):
                a, b_ = [("u", "cascade")], [("u", "teacher_onnx_int8")]
            else:
                c = k.removeprefix("ablation_")
                a, b_ = [("run", ex.run_name(c, s)) for s in ex.ABLATIONS[c][2]], [("run", r_) for r_ in full_runs]
            tests[k] = paired_from([full_val[x] for x in a], [full_val[x] for x in b_], [vals[x] for x in a], [vals[x] for x in b_])
        names = list(tests)
        raw = [tests[k]["tost_p"] for k in names]
        cons = [(tests[k]["tost_tail_count"] + 1) / (N_PAIRED + 1) for k in names]
        for k, h, hc, c_ in zip(names, ev.holm(raw), ev.holm(cons), cons):
            tests[k].update({"tost_p_conservative": c_, "tost_holm11": float(h), "tost_holm11_conservative": float(hc),
                             "survives_holm": bool(h < 0.05), "survives_holm_conservative": bool(hc < 0.05)})
        out["tost_eleven"] = tests
        if cond == "stratified":
            # A2: mean of per-source macro-F1
            mps = {}
            for m in rows:
                mps[m] = {"per_source": per_src_full[m], "mean_per_source": full_val[("mps", m)], "ci95": ci_from(vals[("mps", m)])}
            blk = {m: paired_from([full_val[("mps", "statickd")]], [full_val[("mps", m)]], [vals[("mps", "statickd")]], [vals[("mps", m)]])
                   for m in FAMILY14}
            for b_, ph in zip(blk.values(), ev.holm([b_["p"] for b_ in blk.values()])):
                b_["p_holm14"] = float(ph)
            ext = {m: paired_from([full_val[("mps", "statickd")]], [full_val[("mps", m)]], [vals[("mps", "statickd")]], [vals[("mps", m)]])
                   for m in EXTRA}
            out["mean_per_source"] = {"scores": mps, "paired_family14": blk, "paired_extra": ext}
        res[cond] = out
    res["theta_dev"] = theta

    # conclusion changes stratified vs unstratified (Holm 0.05 / sign / TOST)
    changes = []
    for tag in ("paired", "paired_restricted"):
        for m in FAMILY14:
            u, s = res["unstratified"][tag]["family14"][m], res["stratified"][tag]["family14"][m]
            if (u["p_holm14"] < 0.05) != (s["p_holm14"] < 0.05):
                changes.append({"test": f"{tag} StaticKD vs {m}", "unstratified_p_holm": u["p_holm14"], "stratified_p_holm": s["p_holm14"],
                                "delta": u["delta"]})
        for m in EXTRA:
            u, s = res["unstratified"][tag]["extra"][m], res["stratified"][tag]["extra"][m]
            pk = "p_holm2" if "p_holm2" in u else "p"
            if (u[pk] < 0.05) != (s[pk] < 0.05):
                changes.append({"test": f"{tag} StaticKD vs {m} ({pk})", "unstratified_p": u[pk], "stratified_p": s[pk], "delta": u["delta"]})
    for k in claims:
        u, s = res["unstratified"]["tost_eleven"][k], res["stratified"]["tost_eleven"][k]
        for f in ("tost_equivalent", "survives_holm"):
            if u[f] != s[f]:
                changes.append({"test": f"TOST {k} ({f})", "unstratified": [u["tost_p"], u["tost_holm11"], u[f]],
                                "stratified": [s["tost_p"], s["tost_holm11"], s[f]]})
    res["conclusion_changes"] = changes
    # sanity: unstratified must reproduce r4/paper
    ru = res["unstratified"]["paired"]["family14"]
    res["sanity_vs_r4"] = {
        "m2v_delta_ci": [ru["model2vec_clf"]["delta"], ru["model2vec_clf"]["ci95"]],
        "r4_m2v_delta_ci": [r4["item1"]["c_statickd_minus"]["unrestricted"]["model2vec_clf"]["delta"],
                            r4["item1"]["c_statickd_minus"]["unrestricted"]["model2vec_clf"]["ci95"]],
        "minilm_p_holm": ru["minilm_kd"]["p_holm14"], "paper_minilm_p": numbers.get("sig.test.minilm_kd.p"),
        "teacher_ci": res["unstratified"]["ci_macro_f1"]["teacher"]["ci95"], "paper_teacher_ci": numbers.get("teacher.test.ci"),
        "tost_m2v_p": res["unstratified"]["tost_eleven"]["model2vec_clf_vs_statickd"]["tost_p"],
        "r4_tost_m2v_p": r4["item4"]["tests"]["model2vec_clf_vs_statickd"]["tost_p"]}
    return res


# ============================================================================ A3
def a3(test: pd.DataFrame) -> dict:
    out = {}
    for s in ("dev", "test"):
        view = ev.eval_view(s)
        y = view.y.to_numpy()
        P = {m: softmax(ev.logits_view(m, s)) for m in ("statickd", "teacher")}
        full = {m: ev.ece(P[m], y) for m in P}
        conds = [("document", None)] if s == "dev" else [("site-cluster", "u"), ("stratified", "s")]
        for cname, c in conds:
            if c is None:
                it = ev.bootstrap_samples(len(y), None, N_PAIRED, 0)
            elif c == "u":
                it = ev.bootstrap_samples(len(y), view.cluster.to_numpy(), N_PAIRED, 0)
            else:
                it = (np.concatenate(p) for p in strat_samples(strat_groups(view.source.to_numpy(), view.cluster.to_numpy())))
            vals = {m: np.empty(N_PAIRED) for m in P}
            for b, i in enumerate(it):
                for m in P:
                    vals[m][b] = ev.ece(P[m][i], y[i])
            d = vals["statickd"] - vals["teacher"]
            out[f"{s}_{cname}"] = {
                "statickd": {"ece": full["statickd"], "ci95": ci_from(vals["statickd"])},
                "teacher": {"ece": full["teacher"], "ci95": ci_from(vals["teacher"])},
                "diff_statickd_minus_teacher": {"delta": full["statickd"] - full["teacher"],
                                                "ci95": [float(v) for v in np.percentile(d, [2.5, 97.5])], "p": p_two(d)}}
            print(f"[A3] {s} {cname}", {m: round(full[m], 4) for m in full}, flush=True)
    out["definition"] = "ev.ece: top-1 confidence, 15 equal-width bins (Guo et al., 2017), as notebook part5 (ece.* numbers)"
    return out


# ============================================================================ A4
AUDIT_MODELS = {"teacher": "teacher", "StaticKD (final)": "statickd", "Model2Vec classifier + KD": "model2vec_clf",
                "MiniLM-L6 + KD": "minilm_kd_t1", "DistilmBERT + KD": "distilmbert_kd",
                "TF-IDF + SVM + KD": "tfidf_svm_kd", "fastText + KD": "ft_kd_subword"}
TEACHER_LIKE = ["teacher", "MiniLM-L6 + KD", "DistilmBERT + KD", "StaticKD (final)", "Model2Vec classifier + KD"]
OTHERS = ["fastText + KD", "TF-IDF + SVM + KD"]


def gain_block(correct_a: dict, correct_b: dict, focus_pairs, n_boot=N_PAIRED, seed=0) -> dict:
    """Paired article bootstrap. correct_a = correct vs reference A (editorial), correct_b = vs reference B
    (audited / human). gains = acc_B - acc_A per model; Δ StaticKD - model under A and B; group gain difference."""
    models = list(correct_a)
    n = len(next(iter(correct_a.values())))
    A = np.stack([correct_a[m] for m in models]).astype(float)
    B = np.stack([correct_b[m] for m in models]).astype(float)
    rng = np.random.default_rng(seed)
    bA, bB = np.empty((n_boot, len(models))), np.empty((n_boot, len(models)))
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        bA[b], bB[b] = A[:, i].mean(1), B[:, i].mean(1)
    fa, fb = A.mean(1), B.mean(1)
    ix = {m: k for k, m in enumerate(models)}
    out = {"gain": {}, "delta": {}, "group": {}}
    for m in models:
        g, gb = fb[ix[m]] - fa[ix[m]], bB[:, ix[m]] - bA[:, ix[m]]
        out["gain"][m] = {"acc_ref_a": float(fa[ix[m]]), "acc_ref_b": float(fb[ix[m]]), "gain": float(g),
                          "ci95": [float(v) for v in np.percentile(gb[:N_BOOT], [2.5, 97.5])], "p": p_two(gb)}
    s = ix["StaticKD (final)"]
    for m in focus_pairs:
        k = ix[m]
        da, db = bA[:, s] - bA[:, k], bB[:, s] - bB[:, k]
        out["delta"][m] = {"ref_a": {"delta": float(fa[s] - fa[k]), "ci95": [float(v) for v in np.percentile(da, [2.5, 97.5])], "p": p_two(da)},
                           "ref_b": {"delta": float(fb[s] - fb[k]), "ci95": [float(v) for v in np.percentile(db, [2.5, 97.5])], "p": p_two(db)},
                           "diff_in_diff": {"delta": float((fb[s] - fb[k]) - (fa[s] - fa[k])),
                                            "ci95": [float(v) for v in np.percentile(db - da, [2.5, 97.5])], "p": p_two(db - da)}}
    tl = [ix[m] for m in TEACHER_LIKE]
    ot = [ix[m] for m in OTHERS]
    gfull = (fb - fa)
    gboot = bB - bA
    d_full = gfull[tl].mean() - gfull[ot].mean()
    d_boot = gboot[:, tl].mean(1) - gboot[:, ot].mean(1)
    out["group"] = {"teacher_like": TEACHER_LIKE, "others": OTHERS, "mean_gain_teacher_like": float(gfull[tl].mean()),
                    "mean_gain_others": float(gfull[ot].mean()), "difference": float(d_full),
                    "ci95": [float(v) for v in np.percentile(d_boot, [2.5, 97.5])], "p": p_two(d_boot)}
    return out


def a4(numbers: dict) -> dict:
    out = {}
    r4 = json.load(open(OUT_DIR / "r4_checks.json", encoding="utf8"))
    out["copied_from_r4_item5"] = r4["item5"]
    # ---- 300 audited articles
    sys.path.insert(0, str(ROOT / "scripts" / "label_audit"))
    import analyze  # noqa: E402  (scripts/label_audit/analyze.py; only build_items is used)
    df = analyze.build_items().set_index("item_id")
    # final verified labels exactly as scripts/label_audit/final_status.py (recheck + the authors' three decisions),
    # which wrote results/label_audit/final_model_check.json (not imported: that script writes files on import)
    rc = pd.read_csv(analyze.OUT / "recheck_disagreements.csv", encoding="utf-8-sig").set_index("item_id")
    df["final"] = df["status"]
    df.loc[rc.index, "final"] = rc["recheck_status"]
    for k, v in {"A027": "both_acceptable", "A032": "correct", "A222": "both_acceptable"}.items():
        df.loc[k, "final"] = v
    df = df.reset_index()

    def final_set(r):
        v = list(r["verified"]) if isinstance(r["verified"], list) else []
        if r["final"] == "correct":
            return [r["label"]]
        if r["final"] == "both_acceptable":
            return sorted(set(v) | {r["label"]})
        return v
    df["verified"] = df.apply(final_set, axis=1)
    test = ev.eval_set("test")
    pos = pd.Series(np.arange(len(test)), index=test["doc_id"].astype(str))
    rows = pos.loc[df["doc_id"].astype(str)].to_numpy()
    y_ed = df["label"].map(LABELS.index).to_numpy()
    assert all(isinstance(v, list) and v for v in df["verified"]), "pending / unusable audit items"
    ca, cb = {}, {}
    check = {}
    for name, f in AUDIT_MODELS.items():
        pred = ev.load_logits(f, "test")[rows].argmax(1)
        ca[name] = pred == y_ed
        cb[name] = np.array([LABELS[p] in v for p, v in zip(pred, df["verified"])])
        check[name] = {"acc_editorial": float(ca[name].mean()), "acc_audited": float(cb[name].mean()),
                       "r4_acc_editorial": r4["item5"]["models"][name]["acc_editorial"], "r4_acc_audited": r4["item5"]["models"][name]["acc_audited"]}
    out["audit_reproduces_r4"] = check
    out["audit"] = gain_block(ca, cb, ["teacher", "MiniLM-L6 + KD", "Model2Vec classifier + KD"])
    out["audit"]["n"] = int(len(df))
    out["audit"]["n_items_with_two_acceptable_labels"] = int(sum(len(v) > 1 for v in df["verified"]))
    # ---- 100 blind double-annotated articles
    key_ = pd.read_csv(ROOT / "human_annotation/_key/sample_key.csv", dtype=str)
    fin = pd.read_csv(ROOT / "human_annotation/_key/final_labels.csv", dtype=str)
    h = key_[["item_id", "doc_id"]].merge(fin, on="item_id")
    assert len(h) == 100 and h["final_main"].isin(LABELS).all()
    rows = pos.loc[h["doc_id"].astype(str)].to_numpy()
    assert (test["label"].to_numpy()[rows] == h["label"].to_numpy()).all()
    y_ed = h["label"].map(LABELS.index).to_numpy()
    y_hu = h["final_main"].map(LABELS.index).to_numpy()
    ca, cb, tab = {}, {}, {}
    for name, f in AUDIT_MODELS.items():
        pred = ev.load_logits(f, "test")[rows].argmax(1)
        ca[name], cb[name] = pred == y_ed, pred == y_hu
        ke, kh = int(ca[name].sum()), int(cb[name].sum())
        tab[name] = {"correct_editorial": ke, "acc_editorial": ke / 100, "wilson_editorial": wilson(ke, 100),
                     "correct_human": kh, "acc_human": kh / 100, "wilson_human": wilson(kh, 100)}
    out["human100"] = {"n": 100, "editorial_equals_human": float((y_ed == y_hu).mean()),
                       "sources": h["source"].value_counts().to_dict(), "overlap_with_audit_300": int(len(set(h.doc_id) & set(df.doc_id))),
                       "models": tab, **gain_block(ca, cb, ["teacher", "MiniLM-L6 + KD", "Model2Vec classifier + KD"])}
    out["note"] = ("ref_a = editorial label, ref_b = audited label (300; a prediction counts as correct when it is one of the "
                   "verified labels) or adjudicated human main topic (100). Paired bootstrap over articles, 2,000 resamples, "
                   "seed 0, CIs of gains from the first 1,000, CIs of Δ from all 2,000; p two-sided, unadjusted.")
    return out


# ============================================================================ A5
def a5(numbers: dict) -> dict:
    from statickd.student import StaticKDModel, encode

    model = StaticKDModel(ROOT / "models/final/statickd")
    assert model.max_tokens == 1024
    out = {"max_tokens_default": model.max_tokens}
    LEN512_DIR.mkdir(parents=True, exist_ok=True)
    logits = {}
    for s in ("dev", "heldout", "test"):
        texts = ev.eval_set(s)["text"].tolist()
        f = LEN512_DIR / f"statickd_len512__{s}.npz"
        model.max_tokens = 1024
        ids_full = encode(model.tokenizer, texts, 10**9)
        n_tok = np.array([len(i) for i in ids_full])
        model.max_tokens = 512
        t0 = time.time()
        L = np.concatenate([model.logits(texts[i:i + 512]) for i in range(0, len(texts), 512)]).astype(np.float32)
        np.savez_compressed(f, logits=L)
        logits[s] = L
        view_idx = ev.eval_view(s).index.to_numpy()
        out.setdefault("token_counts", {})[s] = {"median_tokens": float(np.median(n_tok[view_idx])),
                                                 "share_over_512": float((n_tok[view_idx] > 512).mean()),
                                                 "share_over_1024": float((n_tok[view_idx] > 1024).mean())}
        # procedure check: 1024-token logits reproduce the stored StaticKD logits
        if s == "dev":
            model.max_tokens = 1024
            chk = np.concatenate([model.logits(texts[i:i + 512]) for i in range(0, len(texts), 512)])
            out["check_1024_vs_stored_dev_max_abs"] = float(np.abs(chk - ev.load_logits("statickd", "dev")).max())
        print(f"[A5] {s} {time.time() - t0:.0f}s", flush=True)
    t_test = ev.macro_f1(ev.eval_view("test").y.to_numpy(), ev.pred_view("teacher", "test"))
    for s in ("dev", "heldout", "test"):
        view = ev.eval_view(s)
        y = view.y.to_numpy()
        cl = None if s == "dev" else view.cluster.to_numpy()
        p5 = logits[s][view.index.to_numpy()].argmax(1)
        comp = {"statickd": ev.pred_view("statickd", s), "minilm_kd_len512": ev.pred_view("minilm_kd_t1_len512", s),
                "distilmbert_kd_len512": ev.pred_view("distilmbert_kd_len512", s)}
        r = {"macro_f1": ev.macro_f1(y, p5), "accuracy": acc(y, p5), "statickd_1024_macro_f1": ev.macro_f1(y, comp["statickd"]),
             "statickd_1024_accuracy": acc(y, comp["statickd"]), "argmax_agreement_with_1024": float((p5 == comp["statickd"]).mean())}
        for metric_name, metric in (("macro_f1", ev.macro_f1), ("accuracy", acc)):
            vals = {"s512": np.empty(N_PAIRED), **{k: np.empty(N_PAIRED) for k in comp}}
            for b, i in enumerate(ev.bootstrap_samples(len(y), cl, N_PAIRED, 0)):
                vals["s512"][b] = metric(y[i], p5[i])
                for k, p in comp.items():
                    vals[k][b] = metric(y[i], p[i])
            r[f"{metric_name}_ci95"] = ci_from(vals["s512"])
            blk = {k: paired_from([metric(y, p5)], [metric(y, comp[k])], [vals["s512"]], [vals[k]]) for k in comp}
            for b_, ph in zip(blk.values(), ev.holm([b_["p"] for b_ in blk.values()])):
                b_["p_holm3"] = float(ph)
            r[f"paired_{metric_name}_statickd512_minus"] = blk
        if s == "test":
            r["lang_macro_f1"] = ev.grouped_macro_f1(y, p5, view.lang.to_numpy())
            r["pct_teacher"] = r["macro_f1"] / t_test
        out[s] = r
    return out


# ============================================================================ A6
def a6() -> dict:
    pool = load_distillation_pool()
    P = softmax(np.stack(pool["teacher_logits"].to_numpy()))
    ent = -(P * np.log(np.clip(P, 1e-300, None))).sum(1)
    mx = P.max(1)
    out = {}
    parts = {"emmediatopic_train": pool.source.eq("emmediatopic").to_numpy(),
             "ccnews_all": pool.source.isin(["ccnews", "ccnews_extra"]).to_numpy(),
             "ccnews": pool.source.eq("ccnews").to_numpy(), "ccnews_extra": pool.source.eq("ccnews_extra").to_numpy(),
             "ccnews_in_emm_languages": (pool.source.isin(["ccnews", "ccnews_extra"]) & pool.lang.isin(["ca", "el", "hr", "sl"])).to_numpy(),
             "ccnews_other_languages": (pool.source.isin(["ccnews", "ccnews_extra"]) & ~pool.lang.isin(["ca", "el", "hr", "sl"])).to_numpy(),
             "all": np.ones(len(pool), bool)}
    for k, m in parts.items():
        out[k] = {"n": int(m.sum()), "entropy_mean_nats": float(ent[m].mean()), "entropy_median_nats": float(np.median(ent[m])),
                  "entropy_mean_normalised": float(ent[m].mean() / np.log(NUM_LABELS)),
                  "maxprob_mean": float(mx[m].mean()), "maxprob_median": float(np.median(mx[m])),
                  "share_maxprob_gt_0.9": float((mx[m] > 0.9).mean()), "share_maxprob_lt_0.5": float((mx[m] < 0.5).mean())}
    out["note"] = "teacher softmax at tau = 1 of the stored logits (data/processed/teacher_labels/*.parquet); entropy in nats, max log(17) = 2.833"
    return out


# ============================================================================ A7
def a7() -> dict:
    from sklearn.metrics import precision_recall_fscore_support

    view = ev.eval_view("test")
    y = view.y.to_numpy()
    out = {}
    for m, k in (("statickd", "statickd"), ("teacher", "teacher"), ("minilm_kd", "minilm_kd_t1")):
        P = softmax(ev.logits_view(k, "test"))
        pred, conf = P.argmax(1), P.max(1)
        pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=list(range(NUM_LABELS)), zero_division=0)
        r = {"per_label": {LABELS[i]: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i]),
                                      "n_predicted": int((pred == i).sum())} for i in range(NUM_LABELS)},
             "accuracy": acc(y, pred), "macro_f1": ev.macro_f1(y, pred), "thresholds": {}}
        for th in (0.5, 0.7, 0.9):
            keep = conf >= th
            t = {"coverage": float(keep.mean()), "n_kept": int(keep.sum()), "precision_kept": acc(y[keep], pred[keep]),
                 "macro_f1_kept": ev.macro_f1(y[keep], pred[keep]), "per_label": {}}
            for i in range(NUM_LABELS):
                pk = keep & (pred == i)
                t["per_label"][LABELS[i]] = {"n_kept_predicted": int(pk.sum()),
                                            "precision": float((y[pk] == i).mean()) if pk.any() else None,
                                            "coverage_of_predictions": float(pk.sum() / max(1, (pred == i).sum())),
                                            "recall": float((pk & (y == i)).sum() / max(1, (y == i).sum()))}
            r["thresholds"][str(th)] = t
        out[m] = r
    out["note"] = ("coverage = share of test articles with max probability >= threshold; precision_kept = accuracy of the kept "
                   "predictions; per label: precision of the kept predictions of that label, coverage_of_predictions = share "
                   "of that label's predictions that are kept, recall = kept and correct / gold articles of the label")
    return out


# ============================================================================ A8
def a8() -> dict:
    b = pd.read_json(ROOT / "results/final/cpu_benchmark.jsonl", lines=True)
    if "run" not in b:
        b["run"] = 0
    e1 = pd.read_json(OUT_DIR / "e1" / "cpu_benchmark.jsonl", lines=True).drop_duplicates(["model", "threads", "run"], keep="last")
    b = pd.concat([b, e1[e1.model.str.startswith("sdstyle")]], ignore_index=True).drop_duplicates(["model", "threads", "run"], keep="last")
    med = b.groupby(["model", "threads"])[["docs_per_sec", "rss_model_mb", "rss_peak_mb", "size_mb"]].median().reset_index()
    out = {}
    for m, g in med.groupby("model"):
        one = g[g.threads == 1].iloc[0]
        four = g[g.threads == 4].iloc[0] if (g.threads == 4).any() else None
        r = {"docs_per_sec_1c": float(one.docs_per_sec), "rss_model_mb_1c": float(one.rss_model_mb), "rss_peak_mb_1c": float(one.rss_peak_mb),
             "size_mb": float(one.size_mb)}
        for n, tag in ((1e6, "1M"), (1e8, "100M")):
            r[f"cores_{tag}_per_day"] = n / 86400 / r["docs_per_sec_1c"]
            r[f"processes_{tag}_ceil"] = int(np.ceil(r[f"cores_{tag}_per_day"]))
            r[f"ram_mb_{tag}_ceil_processes"] = r[f"processes_{tag}_ceil"] * r["rss_model_mb_1c"]
        if four is not None:
            r["docs_per_sec_4c"] = float(four.docs_per_sec)
            r["rss_model_mb_4c"] = float(four.rss_model_mb)
        out[m] = r
    return {"models": out, "note": ("median over the stored repetitions per (model, threads); SD-style from the later e1 session; "
                                    "cores = articles per day / 86,400 / single-core throughput (24 h at full load, no headroom); "
                                    "RAM per process = model RSS increase at 1 thread; one single-threaded process per core")}


# ============================================================================ A9
def a9() -> dict:
    out = {}
    paths = {"release_repo (dist/static_kd_iptc_classifier)": ROOT / "dist/static_kd_iptc_classifier/data/testset/unified_test_manifest.csv",
             "scripts/build_dist.py output (dist/github)": ROOT / "dist/github/data/testset/unified_test_manifest.csv"}
    for k, p in paths.items():
        if not p.exists():
            out[k] = "missing"
            continue
        m = pd.read_csv(p, dtype=str)
        cc = m[m.source == "ccnews_jsonld"]
        out[k] = {"fields": m.columns.tolist(), "n": len(m), "ccnews_n": len(cc),
                  "ccnews_with_warc_file": int(cc["warc_file"].notna().sum()) if "warc_file" in m else 0,
                  "has_byte_offset_field": any("offset" in c.lower() for c in m.columns),
                  "has_warc_record_id_field": any("record" in c.lower() for c in m.columns),
                  "example_ccnews": cc.iloc[0][[c for c in ("doc_id", "url", "warc_file", "text_sha1") if c in m]].to_dict()}
    t = pd.read_json(ROOT / "data/processed/testset_jsonld/unified_test.jsonl", lines=True, nrows=1)
    out["unified_test_jsonl_fields (src/statickd/testset/unified.py)"] = t.columns.tolist()
    pool = pd.read_parquet(ROOT / "data/processed/testset_jsonld/pool.parquet", columns=None).columns.tolist()
    out["pool_parquet_fields"] = pool
    out["extraction_record_fields_note"] = ("src/statickd/testset/jsonld.py stores per extracted page: url, canonical, domain, ..., "
                                            "warc_date (WARC-Date header), warc (WARC file name); it does not store the byte offset "
                                            "or the WARC-Record-ID of the record")
    return out


def main() -> None:
    t_all = time.time()
    numbers = json.load(open(ROOT / "results/final/paper_numbers.json", encoding="utf8"))
    r4 = json.load(open(OUT_DIR / "r4_checks.json", encoding="utf8"))
    test = ev.eval_view("test")
    only = set(sys.argv[1:])
    res = json.loads(OUT_JSON.read_text(encoding="utf8")) if (only and OUT_JSON.exists()) else {}
    timings = res.get("timings_s", {})
    for name, fn in (("A1_A2", lambda: a1_a2(test, numbers, r4)), ("A3", lambda: a3(test)), ("A4", lambda: a4(numbers)),
                     ("A5", lambda: a5(numbers)), ("A6", a6), ("A7", a7), ("A8", a8), ("A9", a9)):
        if only and name not in only:
            continue
        t0 = time.time()
        res[name] = fn()
        timings[name] = round(time.time() - t0, 1)
        print(f"== {name} done in {timings[name]}s", flush=True)
        OUT_JSON.write_text(json.dumps(jsonable(res), indent=1, ensure_ascii=False), encoding="utf8")
    res["timings_s"] = timings
    res["total_s" if not only else "last_partial_run_s"] = round(time.time() - t_all, 1)
    OUT_JSON.write_text(json.dumps(jsonable(res), indent=1, ensure_ascii=False), encoding="utf8")
    print("written", OUT_JSON)


if __name__ == "__main__":
    main()
