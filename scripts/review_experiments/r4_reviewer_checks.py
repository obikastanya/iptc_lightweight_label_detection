"""Review round 4: re-analyses of stored data, predictions and trained students (no training).

Items (see results/review_experiments/r4_facts.md):
  1 pooled macro-F1 artifact: label inventories per source, restricted-inventory evaluation, paired tests,
    decomposition of the pooled human-annotation subgroup, mean of per-source macro-F1
  2 frozen-embedding variant: folded three-seed ensemble (seeds 0, 1, 2) built in memory, evaluated like StaticKD
  3 source-level sensitivity: per-source and leave-one-source-out differences
  4 Holm adjustment of the TOST p-values of the equivalence family
  5 label-audit ranking with 95% Wilson intervals
  6 512-token transformers in the format of the main table

Bootstrap settings are the notebook's (part1.py N_BOOT = 1000): CIs use the first 1,000 resamples and paired
tests 2,000 resamples, seed 0, site clusters on test/held-out and documents on dev. All paired statistics are
computed from one pass over the resamples (identical draws to ev.paired_bootstrap_seeds / ev.bootstrap_ci).

Run from the project root:  env/Scripts/python.exe scripts/review_experiments/r4_reviewer_checks.py
Output: results/review_experiments/r4_checks.json, results/review_experiments/r4_facts.md,
        results/review_experiments/r4_frozen_ensemble/frozen_ens012__<set>.npz (logits of the frozen ensemble)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from statickd import analysis as an  # noqa: E402
from statickd import evaluation as ev  # noqa: E402
from statickd import experiments as ex  # noqa: E402
from statickd.config import LABELS, NUM_LABELS  # noqa: E402
from statickd.data import load_distillation_pool, softmax  # noqa: E402

OUT_DIR = ROOT / "results" / "review_experiments"
OUT_JSON = OUT_DIR / "r4_checks.json"
OUT_MD = OUT_DIR / "r4_facts.md"
ENS_DIR = OUT_DIR / "r4_frozen_ensemble"
N_BOOT = 1000
N_PAIRED = 2 * N_BOOT
MARGIN = 0.01
KEYS = {"minilm_kd": "minilm_kd_t1", "sdstyle_small": "sdstyle_small_s0", "sdstyle_large": "sdstyle_large_s0"}
MAIN = ["teacher", "teacher_onnx_int8", "distilmbert_kd", "minilm_kd", "statickd", "statickd_compact",
        "ft_kd_subword_q", "ft_kd_subword", "ft_kd_word", "tfidf_svm_kd", "model2vec_clf", "sdstyle_small", "sdstyle_large",
        "ft_gpt_subword", "ft_gpt_word"]
ITEM1 = ["teacher", "statickd", "model2vec_clf", "minilm_kd", "distilmbert_kd", "ft_kd_subword", "tfidf_svm_kd",
         "sdstyle_small"]
LOGIT_KIND = {
    "teacher": "real logits", "teacher_onnx_int8": "real logits", "distilmbert_kd": "real logits",
    "minilm_kd": "real logits", "statickd": "real logits (folded table)", "statickd_compact": "real logits (folded table)",
    "sdstyle_small": "real logits", "sdstyle_large": "real logits",
    "model2vec_clf": "pseudo-logits: log of the MLP-head probabilities, clipped at 1e-12",
    "tfidf_svm_kd": "pseudo-logits: LinearSVC decision_function scores (one-vs-rest margins)",
}
for _m in ("ft_kd_subword_q", "ft_kd_subword", "ft_kd_word", "ft_gpt_subword", "ft_gpt_word"):
    LOGIT_KIND[_m] = "pseudo-logits: log of fastText probabilities (fastText floors them at 1e-5, i.e. -11.513)"
DISPLAY = {m: an.MODELS[m][0] for m in an.MODELS}
DISPLAY["frozen_ens"] = "Frozen embedding, folded 3-seed ensemble"


def key(m: str) -> str:
    return KEYS.get(m, m)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def acc(y, p) -> float:
    return float((np.asarray(y) == np.asarray(p)).mean())


# ----------------------------------------------------------------------------- bootstrap helpers
def boot_values(y, preds: dict, clusters=None, n_boot: int = N_PAIRED, seed: int = 0, metric=ev.macro_f1) -> dict:
    """metric of every prediction array on the same bootstrap resamples (the draws of ev.bootstrap_samples)."""
    y = np.asarray(y)
    vals = {k: np.empty(n_boot) for k in preds}
    for b, i in enumerate(ev.bootstrap_samples(len(y), clusters, n_boot, seed)):
        yi = y[i]
        for k, p in preds.items():
            vals[k][b] = metric(yi, p[i])
    return vals


def ci_from(vals) -> list:
    """95% percentile interval from the first N_BOOT resamples (= ev.bootstrap_ci with n_boot=N_BOOT, seed 0)."""
    return [float(v) for v in np.percentile(vals[:N_BOOT], [2.5, 97.5])]


def paired_from(full_a, full_b, vals_a, vals_b, margin=MARGIN) -> dict:
    """Same statistic as ev.paired_bootstrap_seeds (+ ev.tost): lists of per-seed full-sample scores and
    per-seed bootstrap arrays of the two variants."""
    delta = float(np.mean(full_a) - np.mean(full_b))
    diffs = np.mean(vals_a, axis=0) - np.mean(vals_b, axis=0)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    t = ev.tost(diffs, margin)
    k_tail = max(int((diffs <= -margin).sum()), int((diffs >= margin).sum()))
    return {"delta": delta, "ci95": [float(lo), float(hi)], "p": float(min(1.0, p)),
            "tost_ci90": [t["ci90_low"], t["ci90_high"]], "tost_p": t["p"], "tost_equivalent": t["equivalent"],
            "tost_tail_count": k_tail, "n_boot": int(len(diffs))}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def f3(x, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:+.3f}" if sign else f"{x:.3f}"


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ============================================================================ item 1
def restricted_logits(L: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, L, -np.inf)


def ties_at_max(L: np.ndarray) -> int:
    mx = L.max(1, keepdims=True)
    return int(((L == mx).sum(1) > 1).sum())


def item1(test: pd.DataFrame, res_boot: dict) -> dict:
    y, src = test.y.to_numpy(), test.source.to_numpy()
    out = {}
    # (a) inventories
    inv = {}
    for s, g in test.groupby("source"):
        labs = sorted(g.y.unique())
        inv[s] = labs
        out.setdefault("a_inventories", {})[s] = {
            "n": len(g), "n_labels": len(labs), "labels": [LABELS[i] for i in labs],
            "label_counts": {LABELS[i]: int((g.y == i).sum()) for i in labs}}
    mask = np.zeros((len(test), NUM_LABELS), bool)
    for s, labs in inv.items():
        mask[np.ix_(src == s, labs)] = True

    # (b) restricted evaluation
    models = [m for m in MAIN if ev.has_predictions(key(m), "test")]
    preds_u, preds_r, rows = {}, {}, {}
    for m in models:
        L = ev.logits_view(key(m), "test")
        Lr = restricted_logits(L, mask)
        pu, pr = L.argmax(1), Lr.argmax(1)
        preds_u[m], preds_r[m] = pu, pr
        rows[m] = {"logits_stored": True, "logit_kind": LOGIT_KIND.get(m, "?"),
                   "macro_f1": ev.macro_f1(y, pu), "accuracy": acc(y, pu),
                   "macro_f1_restricted": ev.macro_f1(y, pr), "accuracy_restricted": acc(y, pr),
                   "predictions_outside_inventory": int((~mask[np.arange(len(y)), pu]).sum()),
                   "share_outside_inventory": float((~mask[np.arange(len(y)), pu]).mean()),
                   "ties_at_restricted_max": ties_at_max(Lr), "ties_at_unrestricted_max": ties_at_max(L),
                   "lang_macro_f1": ev.grouped_macro_f1(y, pu, test.lang.to_numpy()),
                   "lang_macro_f1_restricted": ev.grouped_macro_f1(y, pr, test.lang.to_numpy())}
    print("[1b] restricted scores done", flush=True)

    # (c) paired tests in both conditions: one pass over the 2,000 site-cluster resamples
    cl = test.cluster.to_numpy()
    t0 = time.time()
    vals = boot_values(y, {**{("u", m): p for m, p in preds_u.items()}, **{("r", m): p for m, p in preds_r.items()}}, cl)
    print(f"[1c] bootstrap {time.time() - t0:.0f}s", flush=True)
    for m in models:
        rows[m]["ci95"] = ci_from(vals[("u", m)])
        rows[m]["ci95_restricted"] = ci_from(vals[("r", m)])
    comps = {}
    for cond in ("u", "r"):
        block = {}
        for m in [m for m in models if m != "statickd"]:
            pa, pb = (preds_u if cond == "u" else preds_r)["statickd"], (preds_u if cond == "u" else preds_r)[m]
            block[m] = paired_from([ev.macro_f1(y, pa)], [ev.macro_f1(y, pb)], [vals[(cond, "statickd")]], [vals[(cond, m)]])
        holm = ev.holm([b["p"] for b in block.values()])
        for b, ph in zip(block.values(), holm):
            b["p_holm"] = float(ph)
        comps["unrestricted" if cond == "u" else "restricted"] = block
    changes = []
    for m in comps["restricted"]:
        u, r = comps["unrestricted"][m], comps["restricted"][m]
        sig_u, sig_r = u["p_holm"] < 0.05, r["p_holm"] < 0.05
        if sig_u != sig_r or np.sign(u["delta"]) != np.sign(r["delta"]):
            changes.append({"vs": m, "unrestricted": [u["delta"], u["p_holm"]], "restricted": [r["delta"], r["p_holm"]]})
    out["b_scores"] = {m: rows[m] for m in models}
    out["c_statickd_minus"] = comps
    out["c_conclusion_changes"] = changes
    res_boot["item1_vals"] = vals  # reused by items 3/4 sanity checks

    # (d) pooled human-annotation subgroup
    hum = test.label_origin.eq("human_annotation").to_numpy()
    yh, sh = y[hum], src[hum]
    masak_labels = inv["masakhanews"]
    other = [l for l in np.unique(yh) if l not in masak_labels]
    decomp = {}
    for m in ITEM1:
        p = preds_u[m][hum]
        pf1 = ev.per_label_f1(yh, p)
        present = np.unique(yh)
        mk, mn = sh == "masakhanews", sh == "mnds"
        f1_mk = ev.per_label_f1(yh[mk], p[mk])
        f1_mn = ev.per_label_f1(yh[mn], p[mn])
        fp_mk_into_other = int(np.isin(p[mk], other).sum())
        decomp[m] = {
            "pooled_macro_f1": ev.macro_f1(yh, p), "n_labels_pooled": int(len(present)),
            "masakhanews_macro_f1": ev.macro_f1(yh[mk], p[mk]), "mnds_macro_f1": ev.macro_f1(yh[mn], p[mn]),
            "pooled_mean_f1_masakhanews_labels": float(pf1[masak_labels].mean()),
            "pooled_mean_f1_mnds_only_labels": float(pf1[other].mean()),
            "mnds_within_mean_f1_mnds_only_labels": float(f1_mn[other].mean()),
            "mnds_within_mean_f1_masakhanews_labels": float(f1_mn[masak_labels].mean()),
            "masakhanews_articles_predicted_into_mnds_only_labels": fp_mk_into_other,
            "share_masakhanews_predicted_outside_inventory": fp_mk_into_other / int(mk.sum()),
            "pooled_macro_f1_restricted": ev.macro_f1(yh, preds_r[m][hum]),
        }
        if m == "teacher":
            table = []
            for l in present:
                tp = (yh == l) & (p == l)
                table.append({
                    "label": LABELS[l], "gold_masakhanews": int(((yh == l) & mk).sum()), "gold_mnds": int(((yh == l) & mn).sum()),
                    "tp": int(tp.sum()), "fp_from_masakhanews": int(((p == l) & (yh != l) & mk).sum()),
                    "fp_from_mnds": int(((p == l) & (yh != l) & mn).sum()), "fn": int(((yh == l) & (p != l)).sum()),
                    "f1_pooled": float(pf1[l]), "contribution_to_pooled_macro": float(pf1[l] / len(present)),
                    "f1_within_masakhanews": float(f1_mk[l]) if l in masak_labels else None,
                    "f1_within_mnds": float(f1_mn[l])})
            decomp["teacher_per_label"] = table
    out["d_human_subgroup"] = {"masakhanews_labels": [LABELS[l] for l in masak_labels],
                               "mnds_only_labels": [LABELS[l] for l in other], "n_masakhanews": int((sh == "masakhanews").sum()),
                               "n_mnds": int((sh == "mnds").sum()), "models": decomp}

    # (e) mean of per-source macro-F1
    sizes = test.source.value_counts()
    e = {}
    for m in models:
        per_u = {s: ev.macro_f1(y[src == s], preds_u[m][src == s]) for s in inv}
        per_r = {s: ev.macro_f1(y[src == s], preds_r[m][src == s]) for s in inv}
        w = np.array([sizes[s] for s in inv], float)
        e[m] = {"per_source": per_u, "per_source_restricted": per_r,
                "mean_unweighted": float(np.mean(list(per_u.values()))),
                "mean_article_weighted": float(np.average(list(per_u.values()), weights=w)),
                "mean_unweighted_restricted": float(np.mean(list(per_r.values()))),
                "mean_article_weighted_restricted": float(np.average(list(per_r.values()), weights=w))}
    out["e_mean_per_source"] = e
    res_boot["preds_u"], res_boot["preds_r"], res_boot["mask"] = preds_u, preds_r, mask
    return out


# ============================================================================ item 2
def build_ensemble_logits(member_dirs: list[Path], texts_by_set: dict) -> dict:
    """Folded ensemble = mean of the members' token tables (fp32 mean -> fp16) and biases, exactly as
    student.ensemble(), but kept in memory (nothing is written to models/)."""
    from statickd.student import StaticKDModel

    datas = [np.load(Path(d) / "model.npz") for d in member_dirs]
    for d in datas:
        if "token_table" not in d or "token_weight" in d or "bigram_table" in d:
            raise ValueError("not a plain folded student")
    model = StaticKDModel(member_dirs[0])
    model.table = np.mean([d["token_table"].astype(np.float32) for d in datas], axis=0).astype(np.float16)
    model.bias = np.mean([d["bias"].astype(np.float32) for d in datas], axis=0)
    out = {}
    for s, texts in texts_by_set.items():
        t0 = time.time()
        out[s] = np.concatenate([model.logits(texts[i:i + 512]) for i in range(0, len(texts), 512)]).astype(np.float32)
        print(f"[2] {Path(member_dirs[0]).name}.. {s}: {len(texts)} docs {time.time() - t0:.0f}s", flush=True)
    return out


def item2(test: pd.DataFrame, numbers: dict) -> dict:
    from statickd.config import MODELS_DIR

    students = MODELS_DIR / "students"
    members = [students / f"frozen_s{s}" for s in ex.V2_MEMBERS]
    out = {"members": [m.name for m in members], "members_exist": all((m / "model.npz").exists() for m in members)}
    if not out["members_exist"]:
        out["note"] = "frozen students not stored; ensemble not built"
        return out
    meta = json.loads((members[0] / "meta.json").read_text(encoding="utf8"))
    out["member_config"] = {k: meta["config"][k] for k in ("freeze_embeddings", "init", "hidden", "loss", "temperature",
                                                          "token_dropout", "epochs", "sources")}
    sets = {s: ev.eval_set(s)["text"].tolist() for s in ("dev", "heldout", "test")}

    # procedure check: the same in-memory fold of full_s0..2 must reproduce the stored StaticKD logits on dev
    chk = build_ensemble_logits([students / f"full_s{s}" for s in ex.V2_MEMBERS], {"dev": sets["dev"]})["dev"]
    stored = ev.load_logits("statickd", "dev")
    out["procedure_check_statickd_dev"] = {"max_abs_logit_diff": float(np.abs(chk - stored).max()),
                                           "argmax_agreement": float((chk.argmax(1) == stored.argmax(1)).mean())}

    logits = build_ensemble_logits(members, sets)
    ENS_DIR.mkdir(parents=True, exist_ok=True)
    for s, L in logits.items():
        np.savez_compressed(ENS_DIR / f"frozen_ens012__{s}.npz", logits=L)
    # cross-check: mean of the stored per-seed logits (the linear model makes the two nearly identical)
    out["check_vs_mean_of_stored_seed_logits"] = {}
    for s, L in logits.items():
        avg = np.mean([ev.load_logits(f"frozen_s{k}", s) for k in ex.V2_MEMBERS], axis=0)
        out["check_vs_mean_of_stored_seed_logits"][s] = {"max_abs_logit_diff": float(np.abs(avg - L).max()),
                                                         "argmax_agreement": float((avg.argmax(1) == L.argmax(1)).mean())}

    rows, paired = {}, {}
    for s in ("dev", "heldout", "test"):
        view = ev.eval_view(s)
        idx, y = view.index.to_numpy(), view.y.to_numpy()
        cl = None if s == "dev" else view.cluster.to_numpy()
        p_f = logits[s][idx].argmax(1)
        p_s, p_t = ev.pred_view("statickd", s), ev.pred_view("teacher", s)
        for metric_name, metric in (("macro_f1", ev.macro_f1), ("accuracy", acc)):
            vals = boot_values(y, {"f": p_f, "s": p_s}, cl, metric=metric)
            r = rows.setdefault(s, {})
            r[metric_name] = metric(y, p_f)
            r[f"{metric_name}_ci95"] = ci_from(vals["f"])
            r[f"statickd_{metric_name}"] = metric(y, p_s)
            paired.setdefault(s, {})[metric_name] = paired_from([metric(y, p_s)], [metric(y, p_f)], [vals["s"]], [vals["f"]])
        r["teacher_macro_f1"] = ev.macro_f1(y, p_t)
        r["pct_teacher"] = r["macro_f1"] / r["teacher_macro_f1"]
        if s == "test":
            r["lang_macro_f1"] = ev.grouped_macro_f1(y, p_f, view.lang.to_numpy())
        # per-seed frozen scores and the 10 final-recipe seeds (stored predictions)
        r["frozen_seeds_macro_f1"] = [ev.macro_f1(y, ev.pred_view(f"frozen_s{k}", s)) for k in ex.ABLATIONS["frozen"][2]]
        r["frozen_seeds_accuracy"] = [acc(y, ev.pred_view(f"frozen_s{k}", s)) for k in ex.ABLATIONS["frozen"][2]]
        r["full_seeds_macro_f1"] = [ev.macro_f1(y, ev.pred_view(f"full_s{k}", s)) for k in ex.MAIN_SEEDS]
        r["full_seeds_accuracy"] = [acc(y, ev.pred_view(f"full_s{k}", s)) for k in ex.MAIN_SEEDS]
        r["frozen_members_012_mean_macro_f1"] = float(np.mean(r["frozen_seeds_macro_f1"][:3]))
        r["full_members_012_mean_macro_f1"] = float(np.mean(r["full_seeds_macro_f1"][:3]))
    out["scores"] = rows
    out["paired_statickd_minus_frozen_ens"] = paired
    out["paired_note"] = "single comparison, no multiplicity correction; dev = document bootstrap, held-out/test = site-cluster bootstrap"

    # subgroups by distillation-data tier (test)
    tiers = test.distil_tier.to_numpy()
    y = test.y.to_numpy()
    p_f = logits["test"].argmax(1)
    sub = {}
    for t in ["cukup (≥500)", "sedikit (<500)", "tidak ada (0)"]:
        mk = tiers == t
        r = {"n": int(mk.sum()), "frozen_ens": ev.macro_f1(y[mk], p_f[mk]),
             "statickd": ev.macro_f1(y[mk], ev.load_pred("statickd", "test")[mk]),
             "teacher": ev.macro_f1(y[mk], ev.load_pred("teacher", "test")[mk]),
             "model2vec_clf": ev.macro_f1(y[mk], ev.load_pred("model2vec_clf", "test")[mk]),
             "frozen_seed_mean": float(np.mean([ev.macro_f1(y[mk], ev.load_pred(f"frozen_s{k}", "test")[mk]) for k in ex.ABLATIONS["frozen"][2]])),
             "full_seed_mean": float(np.mean([ev.macro_f1(y[mk], ev.load_pred(f"full_s{k}", "test")[mk]) for k in ex.MAIN_SEEDS]))}
        r["frozen_ens_pct_teacher"] = r["frozen_ens"] / r["teacher"]
        r["statickd_pct_teacher"] = r["statickd"] / r["teacher"]
        sub[an.en(t)] = r
    out["distil_tier"] = sub
    out["paper_frozen_ablation"] = {k: v for k, v in numbers.items() if k.startswith(("abl.frozen.", "ablboot.frozen.", "abltost.frozen.", "ablgrp.frozen."))}
    return out


# ============================================================================ item 3
def item3(test: pd.DataFrame, b: dict) -> dict:
    y, src, cl = test.y.to_numpy(), test.source.to_numpy(), test.cluster.to_numpy()
    preds = b["preds_u"]
    focus = ["model2vec_clf", "teacher", "minilm_kd"]
    sources = sorted(np.unique(src))
    per_src = {}
    for s in sources:
        mk = src == s
        n_cl = len(np.unique(cl[mk]))
        row = {"n": int(mk.sum()), "n_clusters": n_cl, "effective_clusters": ev.effective_clusters(cl[mk]),
               "bootstrap": "site-cluster" if n_cl > 1 else "document (source has no site information)",
               "statickd": ev.macro_f1(y[mk], preds["statickd"][mk]), "vs": {}}
        vals = boot_values(y[mk], {m: preds[m][mk] for m in ["statickd", *focus]}, cl[mk] if n_cl > 1 else None)
        for m in focus:
            r = paired_from([ev.macro_f1(y[mk], preds["statickd"][mk])], [ev.macro_f1(y[mk], preds[m][mk])],
                            [vals["statickd"]], [vals[m]])
            r["model_macro_f1"] = ev.macro_f1(y[mk], preds[m][mk])
            r["accuracy_delta"] = acc(y[mk], preds["statickd"][mk]) - acc(y[mk], preds[m][mk])
            a, c = preds["statickd"][mk] == y[mk], preds[m][mk] == y[mk]
            r["only_statickd_correct"], r["only_model_correct"] = int((a & ~c).sum()), int((~a & c).sum())
            row["vs"][m] = r
        per_src[s] = row
        print(f"[3] {s}", {m: round(row['vs'][m]['delta'], 3) for m in focus}, flush=True)
    signs = {m: {"positive": sum(per_src[s]["vs"][m]["delta"] > 0 for s in sources),
                 "negative": sum(per_src[s]["vs"][m]["delta"] < 0 for s in sources),
                 "accuracy_positive": sum(per_src[s]["vs"][m]["accuracy_delta"] > 0 for s in sources)} for m in focus}

    # leave-one-source-out
    loso_models = [m for m in ["teacher", "minilm_kd", "distilmbert_kd", "model2vec_clf", "tfidf_svm_kd", "ft_kd_subword_q",
                               "ft_kd_subword", "ft_kd_word", "sdstyle_small", "sdstyle_large"] if m in preds]
    loso = {}
    for s in sources:
        keep = src != s
        loso[s] = {m: ev.macro_f1(y[keep], preds["statickd"][keep]) - ev.macro_f1(y[keep], preds[m][keep]) for m in loso_models}
    loso_range = {m: {"min": min(loso[s][m] for s in sources), "max": max(loso[s][m] for s in sources),
                      "argmin": min(sources, key=lambda s: loso[s][m]), "argmax": max(sources, key=lambda s: loso[s][m])}
                  for m in loso_models}
    # paired tests after leaving one source out (focus models, Holm over the 3 focus comparisons per left-out source)
    loso_tests = {}
    for s in sources:
        keep = src != s
        vals = boot_values(y[keep], {m: preds[m][keep] for m in ["statickd", *focus]}, cl[keep])
        block = {m: paired_from([ev.macro_f1(y[keep], preds["statickd"][keep])], [ev.macro_f1(y[keep], preds[m][keep])],
                                [vals["statickd"]], [vals[m]]) for m in focus}
        for m, ph in zip(block, ev.holm([block[m]["p"] for m in block])):
            block[m]["p_holm3"] = float(ph)
        loso_tests[s] = {"n": int(keep.sum()), "effective_clusters": ev.effective_clusters(cl[keep]), **block}
        print(f"[3] without {s}", {m: (round(block[m]['delta'], 3), round(block[m]['p'], 3)) for m in focus}, flush=True)
    try:
        import statsmodels  # noqa: F401
        mixed = "statsmodels available but the mixed model was not run"
    except ImportError:
        mixed = ("skipped: statsmodels is not installed in env/ (no install was made). Within-article pairing makes the "
                 "site and language intercepts cancel for the model contrast; the discordant counts per source are given instead.")
    return {"per_source": per_src, "sign_consistency": signs, "loso_delta": loso, "loso_range": loso_range,
            "loso_paired_tests": loso_tests, "mixed_model": mixed}


# ============================================================================ item 4
def item4(test: pd.DataFrame, numbers: dict) -> dict:
    y, cl = test.y.to_numpy(), test.cluster.to_numpy()
    preds = {"statickd": ev.pred_view("statickd", "test"), "model2vec_clf": ev.pred_view("model2vec_clf", "test"),
             "distilmbert_kd": ev.pred_view("distilmbert_kd", "test"),
             "minilm_kd_t1_len512": ev.pred_view("minilm_kd_t1_len512", "test"),
             "distilmbert_kd_len512": ev.pred_view("distilmbert_kd_len512", "test"),
             "teacher_onnx_int8": ev.pred_view("teacher_onnx_int8", "test")}
    # cascade (theta chosen on dev with the notebook rule, part5.py)
    p_dev = softmax(ev.load_logits("statickd", "dev"))
    y_dev = ev.eval_view("dev").y.to_numpy()
    pt_dev = ev.load_pred("teacher_onnx_int8", "dev")
    f1_t_dev = ev.macro_f1(y_dev, pt_dev)
    grid = {th: ev.macro_f1(y_dev, np.where(p_dev.max(1) >= th, p_dev.argmax(1), pt_dev)) for th in (0.5, 0.6, 0.7, 0.8, 0.9)}
    reach = [th for th, f in grid.items() if f >= f1_t_dev]
    theta = reach[0] if reach else max(grid, key=lambda th: (grid[th], -th))
    ps = softmax(ev.load_logits("statickd", "test"))
    preds["cascade"] = np.where(ps.max(1) >= theta, ps.argmax(1), preds["teacher_onnx_int8"])
    runs = {}
    for c, (_, _, seeds) in ex.ABLATIONS.items():
        for s in seeds:
            runs[ex.run_name(c, s)] = ev.pred_view(ex.run_name(c, s), "test")
    t0 = time.time()
    vals = boot_values(y, {**preds, **runs}, cl)
    print(f"[4] bootstrap {time.time() - t0:.0f}s", flush=True)
    f = {k: ev.macro_f1(y, p) for k, p in {**preds, **runs}.items()}

    def pair(a: list, b: list):
        return paired_from([f[k] for k in a], [f[k] for k in b], [vals[k] for k in a], [vals[k] for k in b])

    tests = {"model2vec_clf_vs_statickd": pair(["statickd"], ["model2vec_clf"]),
             "distilmbert_kd_vs_statickd": pair(["statickd"], ["distilmbert_kd"]),
             "minilm_kd_t1_len512_vs_statickd": pair(["statickd"], ["minilm_kd_t1_len512"]),
             "distilmbert_kd_len512_vs_statickd": pair(["statickd"], ["distilmbert_kd_len512"]),
             f"cascade_theta{theta}_vs_teacher_onnx_int8": pair(["cascade"], ["teacher_onnx_int8"])}
    full = [ex.run_name("full", s) for s in ex.MAIN_SEEDS]
    for c, (_, _, seeds) in ex.ABLATIONS.items():
        if c != "full":
            tests[f"ablation_{c}"] = pair([ex.run_name(c, s) for s in seeds], full)
    paper_eq = {"model2vec_clf_vs_statickd": numbers.get("eqv.model2vec_clf.equivalent"),
                "distilmbert_kd_vs_statickd": numbers.get("eqv.distilmbert_kd.equivalent"),
                "minilm_kd_t1_len512_vs_statickd": numbers.get("len512.minilm_kd.equivalent"),
                "distilmbert_kd_len512_vs_statickd": numbers.get("len512.distilmbert_kd.equivalent"),
                f"cascade_theta{theta}_vs_teacher_onnx_int8": "yes (casc.eq.tost_p " + str(numbers.get("casc.eq.tost_p")) + ")"}
    for c in ex.ABLATIONS:
        if c != "full":
            paper_eq[f"ablation_{c}"] = numbers.get(f"abltost.{c}.equivalent")
    claimed = [k for k in tests if tests[k]["tost_equivalent"]]
    B = N_PAIRED
    fams = {}
    for name, members in (("claimed_equivalences", claimed), ("all_tost_tests_in_paper", list(tests))):
        raw = [tests[k]["tost_p"] for k in members]
        cons = [(tests[k]["tost_tail_count"] + 1) / (B + 1) for k in members]
        h_raw, h_cons = ev.holm(raw), ev.holm(cons)
        fams[name] = {k: {"tost_p": r, "tost_p_conservative": c, "holm": float(hr), "holm_conservative": float(hc),
                          "survives_holm": bool(hr < 0.05), "survives_holm_conservative": bool(hc < 0.05)}
                      for k, r, c, hr, hc in zip(members, raw, cons, h_raw, h_cons)}
    return {"theta_dev": theta, "tests": tests, "paper_equivalent_flag": paper_eq, "families": fams,
            "note": ("bootstrap TOST p = larger one-sided tail share of 2,000 paired site-cluster resamples; a tail count of 0 "
                     "gives p = 0, so a conservative (k+1)/(B+1) version is Holm-adjusted as well")}


# ============================================================================ item 5
def item5() -> dict:
    mc = json.loads((ROOT / "results/label_audit/final_model_check.json").read_text(encoding="utf8"))
    rows = {}
    for m, v in mc.items():
        n = int(v["n_section"])
        ks, kv = round(v["acc_vs_section"] * n), round(v["acc_vs_verified"] * v["n_verified"])
        rows[m] = {"n": n, "correct_editorial": ks, "acc_editorial": v["acc_vs_section"], "wilson_editorial": wilson(ks, n),
                   "correct_audited": kv, "acc_audited": v["acc_vs_verified"], "wilson_audited": wilson(kv, int(v["n_verified"]))}
    rank_e = sorted(rows, key=lambda m: -rows[m]["acc_editorial"])
    rank_a = sorted(rows, key=lambda m: -rows[m]["acc_audited"])
    ties_e = [[a, b] for i, a in enumerate(rank_e) for b in rank_e[i + 1:] if rows[a]["correct_editorial"] == rows[b]["correct_editorial"]]
    ties_a = [[a, b] for i, a in enumerate(rank_a) for b in rank_a[i + 1:] if rows[a]["correct_audited"] == rows[b]["correct_audited"]]
    ft, tf = rows["fastText + KD"], rows["TF-IDF + SVM + KD"]
    return {"models": rows, "ranking_editorial": rank_e, "ranking_audited": rank_a, "ties_editorial": ties_e,
            "ties_audited": ties_a, "same_strict_order": rank_e == rank_a,
            "fasttext_minus_tfidf_correct": {"editorial": ft["correct_editorial"] - tf["correct_editorial"],
                                             "audited": ft["correct_audited"] - tf["correct_audited"]},
            "fasttext_tfidf_swap": bool(np.sign(ft["acc_editorial"] - tf["acc_editorial"]) != np.sign(ft["acc_audited"] - tf["acc_audited"])),
            "audit_fasttext_model": "ft_kd_subword (scripts/label_audit/analyze.py MODELS)"}


# ============================================================================ item 6
def item6(numbers: dict) -> dict:
    out = {}
    t_test = ev.macro_f1(ev.eval_view("test").y.to_numpy(), ev.pred_view("teacher", "test"))
    t_dev = ev.macro_f1(ev.eval_view("dev").y.to_numpy(), ev.pred_view("teacher", "dev"))
    for m, p in (("minilm_kd", "minilm_kd_t1_len512"), ("distilmbert_kd", "distilmbert_kd_len512"),
                 ("minilm_kd@256", "minilm_kd_t1"), ("distilmbert_kd@256", "distilmbert_kd"), ("statickd", "statickd")):
        r = {"prediction_key": p}
        for s in ("dev", "heldout", "test"):
            sr = an.score_row(p, s, n_boot=N_BOOT)
            r[s] = {"macro_f1": sr["macro_f1"], "ci95": [sr["ci_low"], sr["ci_high"]], "accuracy": sr["accuracy"]}
            if s == "test":
                r[s]["lang_macro_f1"] = sr["lang_macro_f1"]
        r["test_pct_teacher"] = r["test"]["macro_f1"] / t_test
        r["dev_pct_teacher"] = r["dev"]["macro_f1"] / t_dev
        if not m.endswith("@256") and m != "statickd":
            r["paper_len512"] = {k: v for k, v in numbers.items() if k.startswith(f"len512.{m}.")}
        out[m] = r
    return out


# ============================================================================ markdown
def write_md(res: dict, numbers: dict) -> None:
    L = []
    w = L.append
    w("# Review round 4: fact sheet\n")
    w("Script: `scripts/review_experiments/r4_reviewer_checks.py` (no training; stored data, predictions and trained students).")
    w("Machine-readable numbers: `results/review_experiments/r4_checks.json`.")
    w("Bootstrap settings are the notebook's: CIs use 1,000 resamples and paired tests 2,000 resamples, seed 0, site clusters "
      "on test and held-out, documents on dev. Δ = StaticKD − model unless stated otherwise.\n")
    s = res["sanity"]
    w(f"Sanity check against paper_numbers.json: unrestricted Δ vs Model2Vec {f3(s['m2v_delta'], True)} "
      f"[{f3(s['m2v_ci'][0], True)}, {f3(s['m2v_ci'][1], True)}] (paper {numbers['sig.test.model2vec_clf.ci']}), TOST 90% CI "
      f"[{f3(s['m2v_ci90'][0], True)}, {f3(s['m2v_ci90'][1], True)}] (paper {numbers['eqv.model2vec_clf.ci90']}); teacher test CI "
      f"[{f3(s['teacher_ci'][0])}, {f3(s['teacher_ci'][1])}] (paper {numbers['teacher.test.ci']}); Δ vs MiniLM Holm p {fp(s['minilm_p_holm'])} "
      f"(paper {numbers['sig.test.minilm_kd.p']}).\n")

    # ---- item 1
    i1 = res["item1"]
    w("## 1. Pooled macro-F1 artifact (M2)\n")
    w("### (a) Label inventory per source (gold labels in the test set)\n")
    w("| Source | n | # labels | Labels (count) |\n|---|---|---|---|")
    for src, v in sorted(i1["a_inventories"].items(), key=lambda kv: -kv[1]["n"]):
        w(f"| {src} | {v['n']:,} | {v['n_labels']} | " + ", ".join(f"{k} ({c})" for k, c in v["label_counts"].items()) + " |")
    w("\n### (b) Restricted-inventory evaluation (argmax over the labels of the article's source)\n")
    w("All models have stored 17-way logits for every test article (`results/final/predictions/<model>__test.npz`). "
      "Teacher, transformers, StaticKD and SD-style store real logits; Model2Vec stores log-probabilities (clipped at 1e-12), "
      "fastText log-probabilities (floored at 1e-5 = −11.513, so labels can tie at the floor), TF-IDF+SVM the SVM decision scores. "
      "Ties at the restricted maximum are broken by the lowest label index (argmax).\n")
    w("| Model | Logits | Macro-F1 | 95% CI | Acc. | Macro-F1 restricted | 95% CI | Acc. restricted | Pred. outside inventory | Ties at restricted max |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for m in ITEM1:
        r = i1["b_scores"][m]
        kind = "real" if r["logit_kind"].startswith("real") else "pseudo"
        w(f"| {DISPLAY[m]} | {kind} | {f3(r['macro_f1'])} | {f3(r['ci95'][0])}–{f3(r['ci95'][1])} | {f3(r['accuracy'])} | "
          f"{f3(r['macro_f1_restricted'])} | {f3(r['ci95_restricted'][0])}–{f3(r['ci95_restricted'][1])} | {f3(r['accuracy_restricted'])} | "
          f"{r['predictions_outside_inventory']:,} ({r['share_outside_inventory']:.1%}) | {r['ties_at_restricted_max']} |")
    w("\nOther models of the main table (restricted macro-F1 / acc.): " + "; ".join(
        f"{DISPLAY[m]} {f3(r['macro_f1_restricted'])} / {f3(r['accuracy_restricted'])} (unrestricted {f3(r['macro_f1'])})"
        for m, r in i1["b_scores"].items() if m not in ITEM1) + ".\n")
    w("### (c) Paired site-cluster bootstrap, StaticKD − model (Holm over the 14 comparisons of the significance table, per condition)\n")
    w("| Model | Δ unrestricted [95% CI] | Holm p | Δ restricted [95% CI] | Holm p |\n|---|---|---|---|---|")
    cu, cr = i1["c_statickd_minus"]["unrestricted"], i1["c_statickd_minus"]["restricted"]
    for m in cu:
        w(f"| {DISPLAY[m]} | {f3(cu[m]['delta'], True)} [{f3(cu[m]['ci95'][0], True)}, {f3(cu[m]['ci95'][1], True)}] | {fp(cu[m]['p_holm'])} | "
          f"{f3(cr[m]['delta'], True)} [{f3(cr[m]['ci95'][0], True)}, {f3(cr[m]['ci95'][1], True)}] | {fp(cr[m]['p_holm'])} |")
    for m in ("model2vec_clf", "distilmbert_kd"):
        w(f"\nTOST ±0.01 vs {DISPLAY[m]}: unrestricted 90% CI [{f3(cu[m]['tost_ci90'][0], True)}, {f3(cu[m]['tost_ci90'][1], True)}], p {fp(cu[m]['tost_p'])}, "
          f"equivalent {cu[m]['tost_equivalent']}; restricted 90% CI [{f3(cr[m]['tost_ci90'][0], True)}, {f3(cr[m]['tost_ci90'][1], True)}], "
          f"p {fp(cr[m]['tost_p'])}, equivalent {cr[m]['tost_equivalent']}.")
    ch = i1["c_conclusion_changes"]
    w("\nConclusion changes (significance at Holm 0.05 or sign of Δ): " + ("none." if not ch else "; ".join(
        f"{DISPLAY[c['vs']]}: unrestricted Δ {f3(c['unrestricted'][0], True)} (p {fp(c['unrestricted'][1])}) -> restricted Δ {f3(c['restricted'][0], True)} (p {fp(c['restricted'][1])})" for c in ch)) + "\n")
    d = i1["d_human_subgroup"]
    w("### (d) Why the pooled human-annotation subgroup is below both of its sources\n")
    w(f"The subgroup pools MasakhaNEWS ({d['n_masakhanews']:,} articles, 5 labels: {', '.join(d['masakhanews_labels'])}) and MN-DS "
      f"({d['n_mnds']} articles, 17 labels). Pooled macro-F1 averages over all 17 labels. The 12 labels that occur only in MN-DS "
      f"({', '.join(d['mnds_only_labels'])}) have few gold articles, but every MasakhaNEWS article that a model assigns to one of them "
      "is a false positive for that label, so their precision collapses in the pool.\n")
    t = d["models"]["teacher"]
    w(f"Teacher: pooled {f3(t['pooled_macro_f1'])} = mean of 17 label F1s. Mean F1 of the 5 MasakhaNEWS labels in the pool "
      f"{f3(t['pooled_mean_f1_masakhanews_labels'])}; mean F1 of the 12 MN-DS-only labels in the pool {f3(t['pooled_mean_f1_mnds_only_labels'])} "
      f"against {f3(t['mnds_within_mean_f1_mnds_only_labels'])} within MN-DS alone. {t['masakhanews_articles_predicted_into_mnds_only_labels']:,} "
      f"MasakhaNEWS articles ({t['share_masakhanews_predicted_outside_inventory']:.1%}) are predicted as an MN-DS-only label. "
      f"MasakhaNEWS alone {f3(t['masakhanews_macro_f1'])}, MN-DS alone {f3(t['mnds_macro_f1'])}, pooled with restricted argmax {f3(t['pooled_macro_f1_restricted'])}.\n")
    w("Teacher, per label in the pooled subgroup:\n")
    w("| Label | Gold Masakha | Gold MN-DS | TP | FP from Masakha | FP from MN-DS | FN | F1 pooled | Contribution (F1/17) | F1 within Masakha | F1 within MN-DS |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in d["models"]["teacher_per_label"]:
        w(f"| {r['label']} | {r['gold_masakhanews']} | {r['gold_mnds']} | {r['tp']} | {r['fp_from_masakhanews']} | {r['fp_from_mnds']} | {r['fn']} | "
          f"{f3(r['f1_pooled'])} | {f3(r['contribution_to_pooled_macro'])} | {f3(r['f1_within_masakhanews'])} | {f3(r['f1_within_mnds'])} |")
    w("\nAll models (pooled human subgroup):\n")
    w("| Model | Pooled | Masakha alone | MN-DS alone | Pool: mean F1 5 Masakha labels | Pool: mean F1 12 MN-DS-only labels | Same 12 within MN-DS | Masakha arts. -> MN-DS-only label | Pooled restricted |")
    w("|---|---|---|---|---|---|---|---|---|")
    for m in ITEM1:
        r = d["models"][m]
        w(f"| {DISPLAY[m]} | {f3(r['pooled_macro_f1'])} | {f3(r['masakhanews_macro_f1'])} | {f3(r['mnds_macro_f1'])} | {f3(r['pooled_mean_f1_masakhanews_labels'])} | "
          f"{f3(r['pooled_mean_f1_mnds_only_labels'])} | {f3(r['mnds_within_mean_f1_mnds_only_labels'])} | {r['masakhanews_articles_predicted_into_mnds_only_labels']:,} "
          f"({r['share_masakhanews_predicted_outside_inventory']:.1%}) | {f3(r['pooled_macro_f1_restricted'])} |")
    w("\n### (e) Mean of per-source macro-F1 (6 sources)\n")
    w("| Model | Pooled test macro-F1 | Mean unweighted | Mean article-weighted | Restricted: unweighted | Restricted: weighted |\n|---|---|---|---|---|---|")
    for m in i1["e_mean_per_source"]:
        e = i1["e_mean_per_source"][m]
        w(f"| {DISPLAY[m]} | {f3(i1['b_scores'][m]['macro_f1'])} | {f3(e['mean_unweighted'])} | {f3(e['mean_article_weighted'])} | "
          f"{f3(e['mean_unweighted_restricted'])} | {f3(e['mean_article_weighted_restricted'])} |")

    # ---- item 2
    i2 = res["item2"]
    w("\n## 2. Frozen-embedding variant, folded three-seed ensemble (M3)\n")
    if not i2.get("members_exist"):
        w(i2.get("note", "not built"))
    else:
        pc = i2["procedure_check_statickd_dev"]
        w(f"Trained frozen students are stored (`models/students/frozen_s0..s4`, config: {i2['member_config']}). The ensemble averages the "
          "token tables and biases of seeds 0, 1, 2 in memory (same arithmetic as `student.ensemble`; nothing written to `models/`). Its logits "
          "are saved in `results/review_experiments/r4_frozen_ensemble/`. Procedure check: the same in-memory fold of full_s0..2 reproduces the "
          f"stored StaticKD dev logits (max |Δlogit| {pc['max_abs_logit_diff']:.2e}, argmax agreement {pc['argmax_agreement']:.4f}). "
          "Agreement with the mean of the stored per-seed logits: " + ", ".join(
              f"{k} {v['argmax_agreement']:.4f}" for k, v in i2["check_vs_mean_of_stored_seed_logits"].items()) + ".\n")
        sc = i2["scores"]
        w("Main-table format:\n")
        w("| Model | Dev macro-F1 [95% CI] | Dev acc. | Held-out agreement | Test macro-F1 [95% CI] | Test acc. | Macro-F1/lang. | % teacher (test) |\n|---|---|---|---|---|---|---|---|")
        w(f"| Frozen embedding, 3-seed ensemble | {f3(sc['dev']['macro_f1'])} [{f3(sc['dev']['macro_f1_ci95'][0])}–{f3(sc['dev']['macro_f1_ci95'][1])}] | {f3(sc['dev']['accuracy'])} | "
          f"{f3(sc['heldout']['accuracy'])} | {f3(sc['test']['macro_f1'])} [{f3(sc['test']['macro_f1_ci95'][0])}–{f3(sc['test']['macro_f1_ci95'][1])}] | "
          f"{f3(sc['test']['accuracy'])} | {f3(sc['test']['lang_macro_f1'])} | {sc['test']['pct_teacher']:.1%} |")
        w(f"| StaticKD (paper) | {f3(numbers['statickd.dev.macro'])} [{numbers['statickd.dev.ci']}] | {f3(numbers['statickd.dev.acc'])} | {f3(numbers['statickd.heldout.acc'])} | "
          f"{f3(numbers['statickd.test.macro'])} [{numbers['statickd.test.ci']}] | {f3(numbers['statickd.test.acc'])} | {f3(numbers['statickd.test.langmacro'])} | "
          f"{numbers['statickd.test.pct_teacher']:.1%} |")
        w(f"\nDev % teacher: frozen ensemble {sc['dev']['pct_teacher']:.1%}. Held-out macro-F1 {f3(sc['heldout']['macro_f1'])} "
          f"[{f3(sc['heldout']['macro_f1_ci95'][0])}–{f3(sc['heldout']['macro_f1_ci95'][1])}], agreement CI "
          f"[{f3(sc['heldout']['accuracy_ci95'][0])}–{f3(sc['heldout']['accuracy_ci95'][1])}].\n")
        w("Seed view (macro-F1): " + "; ".join(
            f"{s}: frozen seeds mean {np.mean(sc[s]['frozen_seeds_macro_f1']):.3f} (s0–s2 {sc[s]['frozen_members_012_mean_macro_f1']:.3f}) -> ensemble {sc[s]['macro_f1']:.3f}; "
            f"final-recipe seeds mean {np.mean(sc[s]['full_seeds_macro_f1']):.3f} (s0–s2 {sc[s]['full_members_012_mean_macro_f1']:.3f}) -> StaticKD {sc[s]['statickd_macro_f1']:.3f}"
            for s in ("dev", "test")) + ".\n")
        pr = i2["paired_statickd_minus_frozen_ens"]
        w("Paired, StaticKD − frozen ensemble (single comparison, no Holm):\n")
        w("| Set / metric | Δ | 95% CI | p | TOST ±0.01 90% CI | TOST p | Equivalent |\n|---|---|---|---|---|---|---|")
        for s_, mt in (("test", "macro_f1"), ("test", "accuracy"), ("heldout", "accuracy"), ("dev", "macro_f1")):
            r = pr[s_][mt]
            w(f"| {s_} {mt} | {f3(r['delta'], True)} | [{f3(r['ci95'][0], True)}, {f3(r['ci95'][1], True)}] | {fp(r['p'])} | "
              f"[{f3(r['tost_ci90'][0], True)}, {f3(r['tost_ci90'][1], True)}] | {fp(r['tost_p'])} | {r['tost_equivalent']} |")
        w("\nTest macro-F1 by distillation data of the language:\n")
        w("| Tier | n | Frozen ens. | StaticKD | Teacher | Model2Vec | Frozen seed mean | Final-recipe seed mean | Frozen ens. % teacher | StaticKD % teacher |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for t_, r in i2["distil_tier"].items():
            w(f"| {t_} | {r['n']:,} | {f3(r['frozen_ens'])} | {f3(r['statickd'])} | {f3(r['teacher'])} | {f3(r['model2vec_clf'])} | {f3(r['frozen_seed_mean'])} | "
              f"{f3(r['full_seed_mean'])} | {r['frozen_ens_pct_teacher']:.1%} | {r['statickd_pct_teacher']:.1%} |")

    # ---- item 3
    i3 = res["item3"]
    w("\n## 3. Source-level sensitivity (M4)\n")
    w("Per source, Δ test macro-F1 = StaticKD − model, paired bootstrap 2,000 (site-cluster where the source has site information, "
      "documents otherwise; no multiplicity correction):\n")
    w("| Source | n | Eff. clusters | StaticKD | Δ vs Model2Vec [95% CI] | Δ vs teacher [95% CI] | Δ vs MiniLM [95% CI] |\n|---|---|---|---|---|---|---|")
    for src, r in sorted(i3["per_source"].items(), key=lambda kv: -kv[1]["n"]):
        cells = [f"{f3(r['vs'][m]['delta'], True)} [{f3(r['vs'][m]['ci95'][0], True)}, {f3(r['vs'][m]['ci95'][1], True)}]" for m in ("model2vec_clf", "teacher", "minilm_kd")]
        w(f"| {src} | {r['n']:,} | {r['effective_clusters']:.1f} | {f3(r['statickd'])} | " + " | ".join(cells) + " |")
    w("\nSign consistency over the 6 sources (macro-F1 Δ > 0 / < 0; accuracy Δ > 0): " + "; ".join(
        f"{DISPLAY[m]} {v['positive']}/{v['negative']} (acc. {v['accuracy_positive']}/6)" for m, v in i3["sign_consistency"].items()) + ".\n")
    w("Discordant articles per source (only StaticKD correct / only model correct): " + "; ".join(
        f"{src}: " + ", ".join(f"{m} {r['vs'][m]['only_statickd_correct']}/{r['vs'][m]['only_model_correct']}" for m in ("model2vec_clf", "teacher", "minilm_kd"))
        for src, r in i3["per_source"].items()) + ".\n")
    w("Leave-one-source-out range of Δ (StaticKD − model) over the 6 left-out sources:\n")
    w("| Model | Full test Δ | Min Δ (source left out) | Max Δ (source left out) |\n|---|---|---|---|")
    full_d = {m: res["item1"]["c_statickd_minus"]["unrestricted"][m]["delta"] for m in i3["loso_range"]}
    for m, r in i3["loso_range"].items():
        w(f"| {DISPLAY[m]} | {f3(full_d[m], True)} | {f3(r['min'], True)} ({r['argmin']}) | {f3(r['max'], True)} ({r['argmax']}) |")
    w("\nLeave-one-source-out paired tests (Holm over the three focus comparisons; TOST ±0.01 for Model2Vec):\n")
    w("| Left out | n | Eff. clusters | Δ vs Model2Vec [95% CI], TOST p | Δ vs teacher, p_Holm | Δ vs MiniLM, p_Holm |\n|---|---|---|---|---|---|")
    for src, r in i3["loso_paired_tests"].items():
        m2 = r["model2vec_clf"]
        w(f"| {src} | {r['n']:,} | {r['effective_clusters']:.1f} | {f3(m2['delta'], True)} [{f3(m2['ci95'][0], True)}, {f3(m2['ci95'][1], True)}], {fp(m2['tost_p'])}"
          f"{' (equiv.)' if m2['tost_equivalent'] else ' (not equiv.)'} | {f3(r['teacher']['delta'], True)}, {fp(r['teacher']['p_holm3'])} | "
          f"{f3(r['minilm_kd']['delta'], True)}, {fp(r['minilm_kd']['p_holm3'])} |")
    w(f"\nMixed-effects logistic regression: {i3['mixed_model']}\n")

    # ---- item 4
    i4 = res["item4"]
    w("## 4. Holm adjustment of the TOST p-values (margin ±0.01)\n")
    w(i4["note"] + ".\n")
    for fam, title in (("claimed_equivalences", "Family A: the equivalence claims of the paper"), ("all_tost_tests_in_paper", "Family B: every TOST reported in the paper")):
        w(f"**{title}** ({len(i4['families'][fam])} tests)\n")
        w("| Test | Δ | 90% CI | TOST p | Holm p | Survives | Holm p (conservative) | Survives |\n|---|---|---|---|---|---|---|---|")
        for k, v in sorted(i4["families"][fam].items(), key=lambda kv: kv[1]["tost_p"]):
            t = i4["tests"][k]
            w(f"| {k} | {f3(t['delta'], True)} | [{f3(t['tost_ci90'][0], True)}, {f3(t['tost_ci90'][1], True)}] | {fp(v['tost_p'])} | {fp(v['holm'])} | "
              f"{'yes' if v['survives_holm'] else 'no'} | {fp(v['holm_conservative'])} | {'yes' if v['survives_holm_conservative'] else 'no'} |")
        w("")

    # ---- item 5
    i5 = res["item5"]
    w("## 5. Label-audit ranking (300 audited articles)\n")
    w("| Model | Editorial: acc. (k/300) | Wilson 95% | Audited: acc. (k/300) | Wilson 95% |\n|---|---|---|---|---|")
    for m in i5["ranking_audited"]:
        r = i5["models"][m]
        w(f"| {m} | {r['acc_editorial']:.1%} ({r['correct_editorial']}) | {r['wilson_editorial'][0]:.1%}–{r['wilson_editorial'][1]:.1%} | "
          f"{r['acc_audited']:.1%} ({r['correct_audited']}) | {r['wilson_audited'][0]:.1%}–{r['wilson_audited'][1]:.1%} |")
    w(f"\nRanking, editorial labels: {' > '.join(i5['ranking_editorial'])} (ties: {i5['ties_editorial']}).")
    w(f"Ranking, audited labels: {' > '.join(i5['ranking_audited'])} (ties: {i5['ties_audited']}).")
    w(f"fastText + KD ({i5['audit_fasttext_model']}) minus TF-IDF + SVM + KD, correct articles: editorial {i5['fasttext_minus_tfidf_correct']['editorial']:+d}, "
      f"audited {i5['fasttext_minus_tfidf_correct']['audited']:+d}; swap: {i5['fasttext_tfidf_swap']}.\n")

    # ---- item 6
    i6 = res["item6"]
    w("## 6. 512-token transformers in main-table format\n")
    w("| Model | Dev macro-F1 [95% CI] | Dev acc. | Held-out agreement | Test macro-F1 [95% CI] | Test acc. | Macro-F1/lang. | % teacher (test) |\n|---|---|---|---|---|---|---|---|")
    names = {"minilm_kd": "MiniLM-L6 + KD, 512 tokens", "distilmbert_kd": "DistilmBERT + KD, 512 tokens",
             "minilm_kd@256": "MiniLM-L6 + KD, 256 tokens (paper row)", "distilmbert_kd@256": "DistilmBERT + KD, 256 tokens (paper row)",
             "statickd": "StaticKD (paper row)"}
    for m, r in i6.items():
        w(f"| {names[m]} | {f3(r['dev']['macro_f1'])} [{f3(r['dev']['ci95'][0])}–{f3(r['dev']['ci95'][1])}] | {f3(r['dev']['accuracy'])} | {f3(r['heldout']['accuracy'])} | "
          f"{f3(r['test']['macro_f1'])} [{f3(r['test']['ci95'][0])}–{f3(r['test']['ci95'][1])}] | {f3(r['test']['accuracy'])} | {f3(r['test']['lang_macro_f1'])} | "
          f"{r['test_pct_teacher']:.1%} |")
    w("\nDev % teacher: " + ", ".join(f"{names[m]} {r['dev_pct_teacher']:.1%}" for m, r in i6.items()) + ".")
    w("Paired tests vs StaticKD for the 512-token rows are in paper_numbers.json (`len512.*`): MiniLM-512 Δ "
      f"{numbers['len512.minilm_kd.delta']:+.3f} [{numbers['len512.minilm_kd.ci']}], p {numbers['len512.minilm_kd.p']}; DistilmBERT-512 Δ "
      f"{numbers['len512.distilmbert_kd.delta']:+.3f} [{numbers['len512.distilmbert_kd.ci']}], p {numbers['len512.distilmbert_kd.p']}.\n")
    OUT_MD.write_text("\n".join(L).replace("$-$", "−").replace("$<$", "<"), encoding="utf8")


def main() -> None:
    numbers = json.load(open(ROOT / "results/final/paper_numbers.json", encoding="utf8"))
    test = ev.eval_view("test")
    pool_langs = load_distillation_pool()["lang"].value_counts()
    lang_info = an.language_info(test, pool_langs)
    test = test.join(lang_info[["distil_tier"]], on="lang")
    res, boot = {}, {}
    res["item1"] = item1(test, boot)
    cu = res["item1"]["c_statickd_minus"]["unrestricted"]
    res["sanity"] = {"m2v_delta": cu["model2vec_clf"]["delta"], "m2v_ci": cu["model2vec_clf"]["ci95"],
                     "m2v_ci90": cu["model2vec_clf"]["tost_ci90"], "minilm_p_holm": cu["minilm_kd"]["p_holm"],
                     "teacher_ci": res["item1"]["b_scores"]["teacher"]["ci95"]}
    print("[sanity]", res["sanity"], flush=True)
    res["item2"] = item2(test, numbers)
    res["item3"] = item3(test, boot)
    res["item4"] = item4(test, numbers)
    res["item5"] = item5()
    res["item6"] = item6(numbers)
    OUT_JSON.write_text(json.dumps(jsonable(res), indent=1, ensure_ascii=False), encoding="utf8")
    write_md(res, numbers)
    print("written", OUT_JSON, OUT_MD)


if __name__ == "__main__":
    main()
