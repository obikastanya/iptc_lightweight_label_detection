"""Review round 5: statistics of parts B and C and the fact sheet.

Reads results/review_experiments/{r5_a.json, r5_b1.json, r5_b2_train.json, r5_c_prepare.json, r5_cpu_benchmark.jsonl}
and the stored predictions, computes the paired tests of B1, B2, C1, C2 (same conventions as r4: CIs 1,000 resamples,
paired 2,000, seed 0, site clusters on test/held-out, documents on dev) and writes
results/review_experiments/r5_checks.json and results/review_experiments/r5_facts.md.

Run from the project root:  env/Scripts/python.exe scripts/review_experiments/r5_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "review_experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from statickd import analysis as an  # noqa: E402
from statickd import evaluation as ev  # noqa: E402
from statickd import experiments as ex  # noqa: E402
from statickd.data import load_distillation_pool  # noqa: E402

from r4_reviewer_checks import N_PAIRED, acc, boot_values, ci_from, jsonable, paired_from  # noqa: E402
from r5_a_analyses import DISPLAY as DISP_A, FAMILY14, strat_groups, strat_samples  # noqa: E402

RE = ROOT / "results" / "review_experiments"
OUT_JSON, OUT_MD = RE / "r5_checks.json", RE / "r5_facts.md"
PRED_DIRS = [RE / "r5_b1_predictions", RE / "r5_b2_predictions", RE / "r5_c_predictions", RE / "r5_statickd_len512",
             RE / "r4_frozen_ensemble"]
DISPLAY = dict(DISP_A)
DISPLAY.update({"m2v_lin": "Model2Vec classifier, linear head (normalize=True)",
                "m2v_lin_folded": "  same, folded table without the norm (approximation)",
                "m2v_lin_nonorm": "Model2Vec classifier, linear head, normalize=False",
                "m2v_lin_nonorm_folded": "  same, folded fp16 table (StaticKD code)"})


def load_logits(name: str, s: str) -> np.ndarray:
    for d in PRED_DIRS:
        f = d / f"{name}__{s}.npz"
        if f.exists():
            return np.load(f)["logits"]
    return ev.load_logits(name, s)


def pred_view(name, s):
    return load_logits(name, s)[ev.eval_view(s).index.to_numpy()].argmax(1)


def load(path):
    return json.loads(path.read_text(encoding="utf8")) if path.exists() else None


def f3(x, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:+.3f}" if sign else f"{x:.3f}"


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def ci(c, sign=False):
    return f"[{f3(c[0], sign)}, {f3(c[1], sign)}]"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ----------------------------------------------------------------------------- generic scoring
def main_row(name: str) -> dict:
    """Main-table row: dev macro-F1 [CI], dev acc, held-out agreement, test macro-F1 [CI], test acc, lang macro, % teacher."""
    r = {}
    t_test = ev.macro_f1(ev.eval_view("test").y.to_numpy(), ev.pred_view("teacher", "test"))
    for s in ev.EVAL_SETS:
        view = ev.eval_view(s)
        y, p = view.y.to_numpy(), pred_view(name, s)
        lo, hi = ev.bootstrap_ci(y, p, clusters=None if s == "dev" else view.cluster.to_numpy(), n_boot=1000)
        r[s] = {"macro_f1": ev.macro_f1(y, p), "ci95": [lo, hi], "accuracy": acc(y, p)}
        if s == "test":
            r[s]["lang_macro_f1"] = ev.grouped_macro_f1(y, p, view.lang.to_numpy())
            r["pct_teacher"] = r[s]["macro_f1"] / t_test
    return r


def paired_test(a: str, others: list[str], s: str = "test", metric=ev.macro_f1, stratified=False) -> dict:
    """Δ = a − other for each other model, paired bootstrap 2,000 (site-cluster on test/held-out, document on dev)."""
    view = ev.eval_view(s)
    y = view.y.to_numpy()
    preds = {k: pred_view(k, s) for k in [a, *others]}
    if stratified:
        vals = {k: np.empty(N_PAIRED) for k in preds}
        for b, parts in enumerate(strat_samples(strat_groups(view.source.to_numpy(), view.cluster.to_numpy()))):
            i = np.concatenate(parts)
            for k, p in preds.items():
                vals[k][b] = metric(y[i], p[i])
    else:
        vals = boot_values(y, preds, None if s == "dev" else view.cluster.to_numpy(), metric=metric)
    return {o: paired_from([metric(y, preds[a])], [metric(y, preds[o])], [vals[a]], [vals[o]]) for o in others}


# ============================================================================ B1
def b1_stats(b1: dict) -> dict:
    out = {"raw": b1, "rows": {}, "paired": {}}
    names = []
    for v in ("m2v_lin", "m2v_lin_nonorm"):
        if v in b1:
            names += [v, f"{v}_folded"]
    for n in names + ["model2vec_clf", "statickd"]:
        out["rows"][n] = main_row(n)
    for n in names:
        out["paired"][n] = {
            "test_statickd_minus": paired_test("statickd", [n])[n],
            "test_statickd_minus_stratified": paired_test("statickd", [n], stratified=True)[n],
            "test_model2vec_default_minus": paired_test("model2vec_clf", [n])[n],
            "dev_statickd_minus": paired_test("statickd", [n], "dev")[n],
            "heldout_acc_statickd_minus": paired_test("statickd", [n], "heldout", metric=acc)[n]}
        if n.endswith("_folded"):
            out["paired"][n]["test_unfolded_minus_folded"] = paired_test(n.removesuffix("_folded"), [n])[n]
            out["paired"][n]["test_argmax_agreement_unfolded"] = float(
                (pred_view(n, "test") == pred_view(n.removesuffix("_folded"), "test")).mean())
    return out


# ============================================================================ B1b
B1B_MEMBERS = ["m2v_lin_nonorm_folded", "m2v_lin_nonorm_s1_folded", "m2v_lin_nonorm_s2_folded"]
B1B_ENS = "m2v_lin_nonorm_ens012_folded"


def mean_per_source(names: list[str], ref: str = "statickd") -> dict:
    """Mean of the six per-source test macro-F1s, stratified bootstrap CI (first 1,000) and paired Δ ref − model (2,000)."""
    view = ev.eval_view("test")
    y = view.y.to_numpy()
    src = view.source.to_numpy()
    preds = {n: pred_view(n, "test") for n in [ref, *names]}
    full = {n: float(np.mean([ev.macro_f1(y[src == s], p[src == s]) for s in np.unique(src)])) for n, p in preds.items()}
    vals = {n: np.empty(N_PAIRED) for n in preds}
    for b, parts in enumerate(strat_samples(strat_groups(src, view.cluster.to_numpy()))):
        ys = [y[q] for q in parts]
        for n, p in preds.items():
            vals[n][b] = np.mean([ev.macro_f1(yq, p[q]) for yq, q in zip(ys, parts)])
    return {n: {"mean_per_source": full[n], "ci95": ci_from(vals[n]),
                **({} if n == ref else {"paired_ref_minus": paired_from([full[ref]], [full[n]], [vals[ref]], [vals[n]])})}
            for n in preds}


def b1b_stats(b1b: dict) -> dict:
    names = [n for n in B1B_MEMBERS + [B1B_ENS] if (RE / "r5_b1_predictions" / f"{n}__test.npz").exists()]
    out = {"raw": b1b, "rows": {n: main_row(n) for n in names + ["statickd"]}, "paired": {}}
    for n in names:
        out["paired"][n] = {"test_statickd_minus": paired_test("statickd", [n])[n],
                            "test_statickd_minus_stratified": paired_test("statickd", [n], stratified=True)[n],
                            "test_acc_statickd_minus": paired_test("statickd", [n], metric=acc)[n],
                            "dev_statickd_minus": paired_test("statickd", [n], "dev")[n],
                            "heldout_acc_statickd_minus": paired_test("statickd", [n], "heldout", metric=acc)[n]}
    out["mean_per_source"] = mean_per_source(names)
    # seeds: the three single folded seeds vs the 10 single-seed runs of the final StaticKD recipe (stored ablation runs)
    members = [n for n in B1B_MEMBERS if n in names]
    full_runs = [ex.run_name("full", s) for s in ex.MAIN_SEEDS]
    metrics = {"dev_macro": ("dev", ev.macro_f1), "heldout_acc": ("heldout", acc), "test_macro": ("test", ev.macro_f1),
               "test_acc": ("test", acc)}
    seeds = {}
    for m, (s, fn) in metrics.items():
        view = ev.eval_view(s)
        y = view.y.to_numpy()
        pa, pb = [pred_view(n, s) for n in members], [pred_view(r, s) for r in full_runs]
        va, vb = [fn(y, p) for p in pa], [fn(y, p) for p in pb]
        vals = boot_values(y, {**{("a", k): p for k, p in enumerate(pa)}, **{("b", k): p for k, p in enumerate(pb)}},
                           None if s == "dev" else view.cluster.to_numpy(), metric=fn)
        r = paired_from(va, vb, [vals[("a", k)] for k in range(len(pa))], [vals[("b", k)] for k in range(len(pb))])
        r.update({"m2v_seeds": va, "m2v_mean": float(np.mean(va)), "m2v_sd": float(np.std(va, ddof=1)) if len(va) > 1 else None,
                  "statickd_runs_mean": float(np.mean(vb)), "statickd_runs_sd": float(np.std(vb, ddof=1)),
                  "welch_p": ev.welch(va, vb)["p"] if len(va) > 1 else None})
        seeds[m] = r
    view = ev.eval_view("test")
    seeds["test_lang_macro"] = {"m2v_seeds": [ev.grouped_macro_f1(view.y.to_numpy(), pred_view(n, "test"), view.lang.to_numpy()) for n in members],
                                "statickd_runs": [ev.grouped_macro_f1(view.y.to_numpy(), pred_view(r, "test"), view.lang.to_numpy()) for r in full_runs]}
    out["seeds_vs_final_recipe_runs"] = seeds
    out["seeds_note"] = ("paired-seed bootstrap as the ablation table (mean over the 3 Model2Vec seeds minus mean over the 10 "
                         "final-recipe seeds on the same resample; site clusters on test/held-out, documents on dev); Welch p unadjusted")
    return out


def write_b1b(L: list, R: dict) -> None:
    w = L.append
    b = R.get("B1b")
    w("## B1b. Linear-head Model2Vec (normalize=False): three seeds and the folded 3-seed ensemble\n")
    if not b:
        w("Not run.\n")
        return
    raw = b["raw"]
    w("Seeds 1 and 2 were trained exactly like seed 0 of B1 (m2v_lin_nonorm: n_layers=0, normalize=False, hard teacher labels, every other "
      "setting the library default; seed passed to fit(random_seed) and torch.manual_seed), folded exactly into fp16 tables and run with the "
      "StaticKD inference code. The ensemble averages the three tables and biases (fp32 mean -> fp16 table), as the StaticKD ensemble.\n")
    for n in ("m2v_lin_nonorm_s1", "m2v_lin_nonorm_s2"):
        if n in raw:
            t, fo = raw[n]["train"], raw[n]["fold"]
            w(f"- {n}: training {t['train_seconds'] / 60:.1f} min on GPU; fold check (test): fp32 table {pct(fo['checks']['test']['table_fp32_exact_form_vs_pipeline_argmax_agreement'], 2)} "
              f"argmax agreement with the pipeline (max |Δlogit| {fo['checks']['test']['table_fp32_exact_form_max_abs_logit_diff_vs_unfolded']:.1e}), fp16 table in the StaticKD code "
              f"{pct(fo['checks']['test']['statickd_code_vs_pipeline_argmax_agreement'], 2)}; folded size {fo['folded_size_mb']:.1f} MB.")
    e = raw.get("m2v_lin_nonorm_ens012")
    if e:
        w(f"- ensemble: {e['size_mb']:.1f} MB; logits vs the mean of the member logits: " + ", ".join(
            f"{s} max |Δ| {v['max_abs_logit_diff']:.1e}, argmax agreement {pct(v['argmax_agreement'], 2)}" for s, v in e["check_vs_mean_of_member_logits"].items()) + ".")
    w("\n| Model | Dev macro-F1 [95% CI] | Dev acc. | Held-out agr. | Test macro-F1 [95% CI] | Test acc. | Macro-F1/lang. | % teacher | Mean per-source macro-F1 [95% CI, stratified] |\n|---|---|---|---|---|---|---|---|---|")
    labels = {"m2v_lin_nonorm_folded": "Model2Vec linear, seed 0 (folded)", "m2v_lin_nonorm_s1_folded": "Model2Vec linear, seed 1 (folded)",
              "m2v_lin_nonorm_s2_folded": "Model2Vec linear, seed 2 (folded)", "m2v_lin_nonorm_ens012_folded": "Model2Vec linear, 3-seed folded ensemble",
              "statickd": "StaticKD (3-seed folded ensemble, paper)"}
    for n, r in b["rows"].items():
        mp = b["mean_per_source"][n]
        w(f"| {labels.get(n, n)} | {f3(r['dev']['macro_f1'])} {ci(r['dev']['ci95'])} | {f3(r['dev']['accuracy'])} | {f3(r['heldout']['accuracy'])} | "
          f"{f3(r['test']['macro_f1'])} {ci(r['test']['ci95'])} | {f3(r['test']['accuracy'])} | {f3(r['test']['lang_macro_f1'])} | {pct(r['pct_teacher'])} | "
          f"{f3(mp['mean_per_source'])} {ci(mp['ci95'])} |")
    w("\nPaired tests, Δ = StaticKD − model (single comparisons, unadjusted; 2,000 resamples):\n")
    w("| Model | Test macro-F1 Δ [95% CI], p (site-cluster) | same, stratified | Test acc. Δ, p | Mean per-source Δ [95% CI], p (stratified) | Dev macro-F1 Δ, p | Held-out agr. Δ, p |\n|---|---|---|---|---|---|---|")
    for n, p in b["paired"].items():
        mp = b["mean_per_source"][n]["paired_ref_minus"]
        w(f"| {labels.get(n, n)} | {f3(p['test_statickd_minus']['delta'], True)} {ci(p['test_statickd_minus']['ci95'], True)}, {fp(p['test_statickd_minus']['p'])} | "
          f"{ci(p['test_statickd_minus_stratified']['ci95'], True)}, {fp(p['test_statickd_minus_stratified']['p'])} | "
          f"{f3(p['test_acc_statickd_minus']['delta'], True)}, {fp(p['test_acc_statickd_minus']['p'])} | {f3(mp['delta'], True)} {ci(mp['ci95'], True)}, {fp(mp['p'])} | "
          f"{f3(p['dev_statickd_minus']['delta'], True)}, {fp(p['dev_statickd_minus']['p'])} | {f3(p['heldout_acc_statickd_minus']['delta'], True)}, {fp(p['heldout_acc_statickd_minus']['p'])} |")
    s = b["seeds_vs_final_recipe_runs"]
    w(f"\nSingle seeds against the single-seed runs of the final StaticKD recipe (3 vs 10 seeds; {b['seeds_note']}):\n")
    w("| Metric | Model2Vec linear, mean ± SD (3 seeds) | StaticKD recipe, mean ± SD (10 seeds) | Δ StaticKD − Model2Vec [95% CI] | p | Welch p |\n|---|---|---|---|---|---|")
    for m in ("dev_macro", "heldout_acc", "test_macro", "test_acc"):
        r = s[m]
        w(f"| {m} | {r['m2v_mean']:.3f} ± {r['m2v_sd']:.3f} | {r['statickd_runs_mean']:.3f} ± {r['statickd_runs_sd']:.3f} | "
          f"{f3(-r['delta'], True)} [{f3(-r['ci95'][1], True)}, {f3(-r['ci95'][0], True)}] | {fp(r['p'])} | {fp(r['welch_p'])} |")
    tl = s["test_lang_macro"]
    w(f"\nTest macro-F1 per language (mean over languages): Model2Vec linear seeds {', '.join(f'{v:.3f}' for v in tl['m2v_seeds'])} "
      f"(mean {np.mean(tl['m2v_seeds']):.3f}); StaticKD recipe seeds mean {np.mean(tl['statickd_runs']):.3f} ± {np.std(tl['statickd_runs'], ddof=1):.3f}.\n")
    w("**What changes for the paper:** " + R["verdicts"].get("B1b", "(to be written)") + "\n")


# ============================================================================ B2
def b2_stats(b2: dict, test: pd.DataFrame) -> dict:
    seeds_new = [f"ccnews20k_s{s}" for s in ex.ABLATION_SEEDS]
    groups = {"ccnews20k": seeds_new, "emm_only": [ex.run_name("emm_only", s) for s in ex.ABLATION_SEEDS],
              "full": [ex.run_name("full", s) for s in ex.MAIN_SEEDS]}
    metrics = {"dev_macro": ("dev", ev.macro_f1), "heldout_acc": ("heldout", acc), "test_macro": ("test", ev.macro_f1),
               "test_acc": ("test", acc)}
    per_seed = {g: {m: [] for m in metrics} for g in groups}
    preds = {}
    for g, runs in groups.items():
        for r in runs:
            for m, (s, fn) in metrics.items():
                preds[(r, s)] = preds.get((r, s), pred_view(r, s))
                per_seed[g][m].append(fn(ev.eval_view(s).y.to_numpy(), preds[(r, s)]))
    summ = {g: {m: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "seeds": v} for m, v in d.items()} for g, d in per_seed.items()}
    paired = {}
    for ref in ("full", "emm_only"):
        blk = {}
        for m, (s, fn) in metrics.items():
            view = ev.eval_view(s)
            y = view.y.to_numpy()
            pa, pb = [preds[(r, s)] for r in groups["ccnews20k"]], [preds[(r, s)] for r in groups[ref]]
            vals = boot_values(y, {**{("a", k): p for k, p in enumerate(pa)}, **{("b", k): p for k, p in enumerate(pb)}},
                               None if s == "dev" else view.cluster.to_numpy(), metric=fn)
            r = paired_from([fn(y, p) for p in pa], [fn(y, p) for p in pb], [vals[("a", k)] for k in range(len(pa))],
                            [vals[("b", k)] for k in range(len(pb))])
            r["welch_p"] = ev.welch(per_seed["ccnews20k"][m], per_seed[ref][m])["p"]
            blk[m] = r
        paired[f"ccnews20k_minus_{ref}"] = blk
    # per distillation-data tier (test), mean over seeds
    tiers = test.distil_tier.to_numpy()
    y = test.y.to_numpy()
    grp = {}
    for g, runs in groups.items():
        row = {}
        for t in ["cukup (≥500)", "sedikit (<500)", "tidak ada (0)"]:
            mk = tiers == t
            row[an.en(t)] = float(np.mean([ev.macro_f1(y[mk], load_logits(r, "test").argmax(1)[mk]) for r in runs]))
        row["dev"] = summ[g]["dev_macro"]["mean"]
        grp[g] = row
    grp["n"] = {an.en(t): int((tiers == t).sum()) for t in ["cukup (≥500)", "sedikit (<500)", "tidak ada (0)"]}
    return {"train": b2, "summary": summ, "paired": paired, "tiers": grp,
            "note": "paired-seed bootstrap as notebook part4 (mean over seeds of each variant on the same resample), no Holm; Welch p unadjusted"}


# ============================================================================ C
def c_stats(prep: dict) -> dict:
    out = {"raw": prep}
    rows = {}
    if prep and "c1_model2vec" in prep:
        for tag in ("fp16", "int8"):
            n = f"model2vec_clf_{tag}"
            if prep["c1_model2vec"].get(tag, {}).get("loads"):
                rows[n] = {"vs_fp32": paired_test(n, ["model2vec_clf"])["model2vec_clf"],
                           "vs_fp32_accuracy": paired_test(n, ["model2vec_clf"], metric=acc)["model2vec_clf"],
                           "statickd_minus": paired_test("statickd", [n])[n]}
    if prep and "c2_onnx_int8" in prep:
        for m, stored in (("minilm_kd", "minilm_kd_t1"), ("distilmbert_kd", "distilmbert_kd")):
            i8, f32 = f"{m}_onnx_int8", f"{m}_onnx_fp32"
            view = ev.eval_view("test")
            rows[i8] = {"vs_onnx_fp32": paired_test(i8, [f32])[f32], "vs_onnx_fp32_accuracy": paired_test(i8, [f32], metric=acc)[f32],
                        "vs_stored_fp32": paired_test(i8, [stored])[stored], "statickd_minus": paired_test("statickd", [i8])[i8],
                        "statickd_minus_accuracy": paired_test("statickd", [i8], metric=acc)[i8],
                        "statickd_minus_onnx_fp32": paired_test("statickd", [f32])[f32],
                        "test_macro_f1": ev.macro_f1(view.y.to_numpy(), pred_view(i8, "test")),
                        "test_accuracy": acc(view.y.to_numpy(), pred_view(i8, "test"))}
    out["paired"] = rows
    dg = load(RE / "r5_c_minilm_diag.json")
    if dg and "best_on_dev" in dg:
        out["minilm_diag"] = dg
        b = f"minilm_kd_int8_{dg['best_on_dev']}"
        out["minilm_diag_paired"] = paired_test("statickd", [b])[b]
        out["minilm_diag_vs_onnx_fp32"] = paired_test(b, ["minilm_kd_onnx_fp32"])["minilm_kd_onnx_fp32"]
    bf = RE / "r5_cpu_benchmark.jsonl"
    if bf.exists():
        b = pd.read_json(bf, lines=True).drop_duplicates(["model", "threads", "run"], keep="last")
        med = b.groupby(["model", "threads"])[["latency_ms_p50", "latency_ms_p95", "docs_per_sec", "rss_model_mb", "size_mb"]].median()
        rng = b.groupby(["model", "threads"]).docs_per_sec.agg(["min", "max", "count"])
        bench = {}
        for (m, t), r in med.iterrows():
            bench.setdefault(m, {})[f"{t}c"] = {**{k: float(v) for k, v in r.items()}, "dps_min": float(rng.loc[(m, t), "min"]),
                                               "dps_max": float(rng.loc[(m, t), "max"]), "n_runs": int(rng.loc[(m, t), "count"])}
        out["benchmark"] = bench
    return out


# ============================================================================ markdown
def write_md(R: dict, numbers: dict) -> None:
    L = []
    w = L.append
    A = R["A"]
    w("# Review round 5: fact sheet\n")
    w("Scripts: `scripts/review_experiments/r5_a_analyses.py` (part A), `r5_b1_model2vec_linear.py`, `r5_b2_ccnews20k.py` (part B, GPU), "
      "`r5_c_prepare_eval.py`, `r5_c_benchmark.py` (part C, CPU), `r5_report.py` (statistics of B/C and this sheet). "
      "Numbers: `results/review_experiments/r5_checks.json`. Bootstrap conventions as the notebook and r4: CIs from 1,000 resamples, "
      "paired tests from 2,000, seed 0; site clusters on test and held-out, documents on dev. Δ = StaticKD − model unless stated otherwise.\n")
    a1 = A["A1_A2"]
    s = a1["sanity_vs_r4"]
    w(f"Sanity: the unstratified pass reproduces r4 and the paper (Δ vs Model2Vec {f3(s['m2v_delta_ci'][0], True)} {ci(s['m2v_delta_ci'][1], True)}, "
      f"MiniLM Holm p {fp(s['minilm_p_holm'])} (paper {s['paper_minilm_p']}), teacher CI {ci(s['teacher_ci'])} (paper {s['paper_teacher_ci']}), "
      f"TOST p vs Model2Vec {s['tost_m2v_p']:.4f} (r4 {s['r4_tost_m2v_p']:.4f})).\n")

    # ---- A1
    w("## A1. Source-stratified site-cluster bootstrap\n")
    st = a1["strata"]
    w("Stratified scheme: the six test sources are fixed; within each source its sites are resampled with replacement. Sources without site "
      "information form a single cluster in the paper's bootstrap; within the stratified scheme they are resampled by documents. Units per source: "
      + ", ".join(f"{k} {v['n_units']} {v['kind']} (n={v['n']:,})" for k, v in st.items()) + ".\n")
    w("### (a) Test macro-F1, 95% CI\n")
    w("| Model | Macro-F1 | CI unstratified (paper) | CI stratified |\n|---|---|---|---|")
    for m, v in a1["unstratified"]["ci_macro_f1"].items():
        sv = a1["stratified"]["ci_macro_f1"][m]
        w(f"| {DISPLAY.get(m, m)} | {f3(v['macro_f1'])} | {f3(v['ci95'][0])}–{f3(v['ci95'][1])} | {f3(sv['ci95'][0])}–{f3(sv['ci95'][1])} |")
    w("\n### (b) Paired Δ macro-F1 = StaticKD − model (Holm over the 14 comparisons of the significance table)\n")
    w("| Model | Δ | 95% CI unstratified | Holm p | 95% CI stratified | Holm p |\n|---|---|---|---|---|---|")
    for m in FAMILY14:
        u, sv = a1["unstratified"]["paired"]["family14"][m], a1["stratified"]["paired"]["family14"][m]
        w(f"| {DISPLAY.get(m, m)} | {f3(u['delta'], True)} | {ci(u['ci95'], True)} | {fp(u['p_holm14'])} | {ci(sv['ci95'], True)} | {fp(sv['p_holm14'])} |")
    for m in ("minilm_kd_len512", "distilmbert_kd_len512", "frozen_ens"):
        u, sv = a1["unstratified"]["paired"]["extra"][m], a1["stratified"]["paired"]["extra"][m]
        pk = "p_holm2" if "p_holm2" in u else "p"
        lab = "Holm over 2 (paper)" if pk == "p_holm2" else "unadjusted (paper)"
        w(f"| {DISPLAY.get(m, m)} ({lab}) | {f3(u['delta'], True)} | {ci(u['ci95'], True)} | {fp(u[pk])} | {ci(sv['ci95'], True)} | {fp(sv[pk])} |")
    w("\n### (c) TOST ±0.01, the eleven equivalence claims (Holm over 11)\n")
    w("| Test | Δ | 90% CI unstrat. | TOST p | Holm p | 90% CI strat. | TOST p | Holm p | Holm p (k+1)/(B+1) strat. |\n|---|---|---|---|---|---|---|---|---|")
    tu, ts = a1["unstratified"]["tost_eleven"], a1["stratified"]["tost_eleven"]
    for k in sorted(tu, key=lambda k: tu[k]["tost_p"]):
        u, sv = tu[k], ts[k]
        w(f"| {k} | {f3(u['delta'], True)} | {ci(u['tost_ci90'], True)} | {fp(u['tost_p'])} | {fp(u['tost_holm11'])} | {ci(sv['tost_ci90'], True)} | "
          f"{fp(sv['tost_p'])} | {fp(sv['tost_holm11'])} | {fp(sv['tost_holm11_conservative'])} |")
    ch = a1["conclusion_changes"]
    w("\nConclusions that change from the unstratified to the stratified bootstrap (Holm 0.05 significance, TOST equivalence, Holm-surviving "
      "equivalence): " + ("none." if not ch else "; ".join(
          f"{c['test']}: " + (f"unstratified {c['unstratified']}, stratified {c['stratified']}" if "unstratified" in c else
                              f"p {c.get('unstratified_p_holm', c.get('unstratified_p')):.3f} -> {c.get('stratified_p_holm', c.get('stratified_p')):.3f}")
          for c in ch)) + " (lists: TOST p, Holm p, flag).\n")
    w("Restricted-inventory paired tests, stratified (Holm 14): " + "; ".join(
        f"{DISPLAY.get(m, m)} {f3(v['delta'], True)} {ci(v['ci95'], True)} p {fp(v['p_holm14'])} (unstrat. p {fp(a1['unstratified']['paired_restricted']['family14'][m]['p_holm14'])})"
        for m, v in a1["stratified"]["paired_restricted"]["family14"].items()) + ".\n")
    w("**What changes for the paper:** " + R["verdicts"]["A1"] + "\n")

    # ---- A2
    mp = a1["stratified"]["mean_per_source"]
    w("## A2. Mean of per-source macro-F1 and restricted-inventory macro-F1 (stratified bootstrap)\n")
    w("| Model | Mean per-source macro-F1 [95% CI] | Δ StaticKD − model [95% CI] | Holm p (14) | Restricted pooled macro-F1 [95% CI] |\n|---|---|---|---|---|")
    for m, v in mp["scores"].items():
        if m == "statickd":
            d = "--"
            p_ = "--"
        else:
            pr = mp["paired_family14"].get(m) or mp["paired_extra"].get(m)
            d = f"{f3(pr['delta'], True)} {ci(pr['ci95'], True)}"
            p_ = fp(pr["p_holm14"]) if "p_holm14" in pr else f"{fp(pr['p'])} (unadj.)"
        rr = a1["stratified"]["ci_macro_f1_restricted"][m]
        w(f"| {DISPLAY.get(m, m)} | {f3(v['mean_per_source'])} {ci(v['ci95'])} | {d} | {p_} | {f3(rr['macro_f1'])} {ci(rr['ci95'])} |")
    w("\n**What changes for the paper:** " + R["verdicts"]["A2"] + "\n")

    # ---- A3
    a3 = A["A3"]
    w("## A3. Expected calibration error\n")
    w(f"Definition: {a3['definition']}.\n")
    w("| Set (bootstrap) | StaticKD ECE [95% CI] | Teacher ECE [95% CI] | Δ StaticKD − teacher [95% CI] | p |\n|---|---|---|---|---|")
    for k in ("dev_document", "test_site-cluster", "test_stratified"):
        v = a3[k]
        d = v["diff_statickd_minus_teacher"]
        w(f"| {k} | {f3(v['statickd']['ece'])} {ci(v['statickd']['ci95'])} | {f3(v['teacher']['ece'])} {ci(v['teacher']['ci95'])} | "
          f"{f3(d['delta'], True)} {ci(d['ci95'], True)} | {fp(d['p'])} |")
    w("\n**What changes for the paper:** " + R["verdicts"]["A3"] + "\n")

    # ---- A4
    a4 = A["A4"]
    w("## A4. Audited and human labels\n")
    w("### 300 audited articles (copy of r4 item 5, re-derived per article)\n")
    w("| Model | Editorial acc. (k) | Wilson 95% | Audited acc. (k) | Wilson 95% | Gain [95% CI] |\n|---|---|---|---|---|---|")
    for m in a4["copied_from_r4_item5"]["ranking_audited"]:
        r = a4["copied_from_r4_item5"]["models"][m]
        g = a4["audit"]["gain"][m]
        w(f"| {m} | {pct(r['acc_editorial'])} ({r['correct_editorial']}) | {pct(r['wilson_editorial'][0])}–{pct(r['wilson_editorial'][1])} | "
          f"{pct(r['acc_audited'])} ({r['correct_audited']}) | {pct(r['wilson_audited'][0])}–{pct(r['wilson_audited'][1])} | "
          f"{g['gain'] * 100:+.1f} pp [{g['ci95'][0] * 100:+.1f}, {g['ci95'][1] * 100:+.1f}] |")
    w(f"\n{a4['audit']['n_items_with_two_acceptable_labels']} of the 300 audited items accept two labels (a prediction matching either counts as correct, as in final_model_check.json).\n")
    w("Paired Δ accuracy StaticKD − model (article bootstrap, 2,000):\n")
    w("| vs | Editorial labels Δ [95% CI], p | Audited labels Δ [95% CI], p | Change (audited − editorial) [95% CI], p |\n|---|---|---|---|")
    for m, d in a4["audit"]["delta"].items():
        w(f"| {m} | {d['ref_a']['delta'] * 100:+.1f} pp [{d['ref_a']['ci95'][0] * 100:+.1f}, {d['ref_a']['ci95'][1] * 100:+.1f}], {fp(d['ref_a']['p'])} | "
          f"{d['ref_b']['delta'] * 100:+.1f} pp [{d['ref_b']['ci95'][0] * 100:+.1f}, {d['ref_b']['ci95'][1] * 100:+.1f}], {fp(d['ref_b']['p'])} | "
          f"{d['diff_in_diff']['delta'] * 100:+.1f} pp [{d['diff_in_diff']['ci95'][0] * 100:+.1f}, {d['diff_in_diff']['ci95'][1] * 100:+.1f}], {fp(d['diff_in_diff']['p'])} |")
    g = a4["audit"]["group"]
    w(f"\nGain from editorial to audited labels, teacher-like models ({', '.join(g['teacher_like'])}) {g['mean_gain_teacher_like'] * 100:+.1f} pp on average "
      f"vs the others ({', '.join(g['others'])}) {g['mean_gain_others'] * 100:+.1f} pp; difference {g['difference'] * 100:+.1f} pp "
      f"[{g['ci95'][0] * 100:+.1f}, {g['ci95'][1] * 100:+.1f}], p {fp(g['p'])}.\n")
    h = a4["human100"]
    w(f"### 100 blind double-annotated articles (adjudicated human main topic, no LLM in labelling)\n")
    w(f"Sources: {h['sources']}; overlap with the 300 audited articles: {h['overlap_with_audit_300']}; editorial label = human main topic "
      f"for {pct(h['editorial_equals_human'], 0)} of the articles.\n")
    w("| Model | Editorial acc. | Wilson 95% | Human acc. | Wilson 95% | Gain [95% CI] |\n|---|---|---|---|---|---|")
    for m in sorted(h["models"], key=lambda m: -h["models"][m]["acc_human"]):
        r, gg = h["models"][m], h["gain"][m]
        w(f"| {m} | {pct(r['acc_editorial'], 0)} ({r['correct_editorial']}) | {pct(r['wilson_editorial'][0])}–{pct(r['wilson_editorial'][1])} | "
          f"{pct(r['acc_human'], 0)} ({r['correct_human']}) | {pct(r['wilson_human'][0])}–{pct(r['wilson_human'][1])} | "
          f"{gg['gain'] * 100:+.0f} pp [{gg['ci95'][0] * 100:+.0f}, {gg['ci95'][1] * 100:+.0f}] |")
    w("\nPaired Δ accuracy StaticKD − model on the 100 articles: " + "; ".join(
        f"{m}: editorial {d['ref_a']['delta'] * 100:+.0f} pp [{d['ref_a']['ci95'][0] * 100:+.0f}, {d['ref_a']['ci95'][1] * 100:+.0f}] p {fp(d['ref_a']['p'])}, "
        f"human {d['ref_b']['delta'] * 100:+.0f} pp [{d['ref_b']['ci95'][0] * 100:+.0f}, {d['ref_b']['ci95'][1] * 100:+.0f}] p {fp(d['ref_b']['p'])}, "
        f"change {d['diff_in_diff']['delta'] * 100:+.0f} pp p {fp(d['diff_in_diff']['p'])}" for m, d in h["delta"].items()) + ".")
    g = h["group"]
    w(f"\nGain teacher-like {g['mean_gain_teacher_like'] * 100:+.1f} pp vs others {g['mean_gain_others'] * 100:+.1f} pp; difference "
      f"{g['difference'] * 100:+.1f} pp [{g['ci95'][0] * 100:+.1f}, {g['ci95'][1] * 100:+.1f}], p {fp(g['p'])}.\n")
    w("**What changes for the paper:** " + R["verdicts"]["A4"] + "\n")

    # ---- A5
    a5 = A["A5"]
    w("## A5. StaticKD with input truncated to 512 tokens\n")
    w("Token counts (StaticKD tokenizer, [UNK] removed): " + "; ".join(
        f"{s} median {v['median_tokens']:.0f}, {pct(v['share_over_512'])} > 512, {pct(v['share_over_1024'])} > 1,024" for s, v in a5["token_counts"].items())
      + f". Procedure check: the same code with 1,024 tokens reproduces the stored dev logits (max |Δ| {a5['check_1024_vs_stored_dev_max_abs']:.1e}).\n")
    w("| Set | StaticKD-512 macro-F1 [95% CI] | Acc. | StaticKD-1024 macro-F1 / acc. | Argmax agreement with 1,024 |\n|---|---|---|---|---|")
    for s in ("dev", "heldout", "test"):
        v = a5[s]
        w(f"| {s} | {f3(v['macro_f1'])} {ci(v['macro_f1_ci95'])} | {f3(v['accuracy'])} | {f3(v['statickd_1024_macro_f1'])} / {f3(v['statickd_1024_accuracy'])} | {pct(v['argmax_agreement_with_1024'])} |")
    w(f"\nTest macro-F1/language {f3(a5['test']['lang_macro_f1'])}, % teacher {pct(a5['test']['pct_teacher'])}.\n")
    w("Paired Δ = StaticKD-512 − model (Holm over the 3 comparisons per set and metric):\n")
    w("| Set, metric | vs StaticKD-1024 | vs MiniLM-L6 512 | vs DistilmBERT 512 |\n|---|---|---|---|")
    for s in ("dev", "test"):
        for mt in ("macro_f1", "accuracy"):
            b = a5[s][f"paired_{mt}_statickd512_minus"]
            w(f"| {s} {mt} | " + " | ".join(f"{f3(b[k]['delta'], True)} {ci(b[k]['ci95'], True)} p {fp(b[k]['p_holm3'])}"
                                           for k in ("statickd", "minilm_kd_len512", "distilmbert_kd_len512")) + " |")
    w("\n**What changes for the paper:** " + R["verdicts"]["A5"] + "\n")

    # ---- A6
    a6 = A["A6"]
    w("## A6. Teacher output entropy and confidence on the distillation data\n")
    w(f"{a6['note']}.\n")
    w("| Part | n | Entropy mean (nats) | Entropy median | Max prob. mean | Max prob. median | Share max prob. > 0.9 | Share < 0.5 |\n|---|---|---|---|---|---|---|---|")
    for k in ("emmediatopic_train", "ccnews_all", "ccnews", "ccnews_extra", "ccnews_in_emm_languages", "ccnews_other_languages", "all"):
        v = a6[k]
        w(f"| {k} | {v['n']:,} | {v['entropy_mean_nats']:.3f} | {v['entropy_median_nats']:.3f} | {v['maxprob_mean']:.3f} | {v['maxprob_median']:.3f} | "
          f"{pct(v['share_maxprob_gt_0.9'])} | {pct(v['share_maxprob_lt_0.5'])} |")
    w("\n**What changes for the paper:** " + R["verdicts"]["A6"] + "\n")

    # ---- A7
    a7 = A["A7"]
    w("## A7. Filtering use case (test set)\n")
    w("Per-label precision / recall / F1:\n")
    w("| Label | Support | StaticKD P / R / F1 | Teacher P / R / F1 | MiniLM-L6 P / R / F1 |\n|---|---|---|---|---|")
    for lab, v in a7["statickd"]["per_label"].items():
        cells = [f"{a7[m]['per_label'][lab]['precision']:.2f} / {a7[m]['per_label'][lab]['recall']:.2f} / {a7[m]['per_label'][lab]['f1']:.2f}"
                 for m in ("statickd", "teacher", "minilm_kd")]
        w(f"| {lab} | {v['support']} | " + " | ".join(cells) + " |")
    w("\nConfidence threshold (keep predictions with max probability ≥ θ):\n")
    w("| θ | StaticKD coverage / precision | Teacher coverage / precision | MiniLM-L6 coverage / precision |\n|---|---|---|---|")
    for th in ("0.5", "0.7", "0.9"):
        w(f"| {th} | " + " | ".join(f"{pct(a7[m]['thresholds'][th]['coverage'])} / {pct(a7[m]['thresholds'][th]['precision_kept'])}"
                                    for m in ("statickd", "teacher", "minilm_kd")) + " |")
    w("\nPer label at θ = 0.7 and 0.9 (StaticKD; precision of kept predictions / recall):\n")
    w("| Label | θ=0.7 P / R | θ=0.9 P / R | Teacher θ=0.9 P / R |\n|---|---|---|---|")
    for lab in a7["statickd"]["per_label"]:
        c = []
        for m, th in (("statickd", "0.7"), ("statickd", "0.9"), ("teacher", "0.9")):
            v = a7[m]["thresholds"][th]["per_label"][lab]
            prec = "--" if v["precision"] is None else f"{v['precision']:.2f}"
            c.append(f"{prec} / {v['recall']:.2f} (n {v['n_kept_predicted']})")
        w(f"| {lab} | " + " | ".join(c) + " |")
    w(f"\n{a7['note']}.\n")
    w("**What changes for the paper:** " + R["verdicts"]["A7"] + "\n")

    # ---- A8
    a8 = A["A8"]
    w("## A8. Operating regime (stored benchmark, single-core throughput)\n")
    w("| Model | Articles/s (1c) | Cores for 1M/day | Cores for 100M/day | RAM per process (MB) | RAM for 100M/day (GB, ceil processes) |\n|---|---|---|---|---|---|")
    for m, v in sorted(a8["models"].items(), key=lambda kv: -kv[1]["docs_per_sec_1c"]):
        def g3(x):
            return f"{x:,.0f}" if x >= 100 else f"{x:.3g}"
        w(f"| {DISPLAY.get(m, m)} | {g3(v['docs_per_sec_1c'])} | {g3(v['cores_1M_per_day'])} | {g3(v['cores_100M_per_day'])} | {v['rss_model_mb_1c']:,.0f} | "
          f"{v['ram_mb_100M_ceil_processes'] / 1024:,.1f} |")
    w(f"\n{a8['note']}.\n")
    w("**What changes for the paper:** " + R["verdicts"]["A8"] + "\n")

    # ---- A9
    a9 = A["A9"]
    w("## A9. Release check: test-set manifest\n")
    for k, v in a9.items():
        if isinstance(v, dict) and "fields" in v:
            w(f"- {k}: fields {', '.join(v['fields'])}; CC-NEWS rows {v['ccnews_n']:,}, with warc_file {v['ccnews_with_warc_file']:,}; "
              f"byte-offset field: {'yes' if v['has_byte_offset_field'] else 'no'}; WARC-Record-ID field: {'yes' if v['has_warc_record_id_field'] else 'no'}; "
              f"example {v['example_ccnews']}.")
    w(f"- {a9['extraction_record_fields_note']}.")
    w(f"- `data/processed/testset_jsonld/unified_test.jsonl` fields: {', '.join(a9['unified_test_jsonl_fields (src/statickd/testset/unified.py)'])}.")
    w("\n**What changes for the paper:** " + R["verdicts"]["A9"] + "\n")

    # ---- A10
    w("## A10. Reference check: Zschache and Hartwig\n")
    w(R["A10"] + "\n")

    # ---- B1
    w("## B1. Model2Vec classifier with a linear head\n")
    b1 = R.get("B1")
    if not b1:
        w("Not run.\n")
    else:
        raw = b1["raw"]
        for v in ("m2v_lin", "m2v_lin_nonorm"):
            if v in raw:
                t = raw[v].get("train", {})
                fo = raw[v]["fold"]
                w(f"**{v}** ({DISPLAY[v]}): {' '.join(t.get('head', '').split())}, normalize={t.get('normalize')}, train {t.get('train_seconds', 0) / 60:.1f} min on GPU. "
                  f"Pipeline {fo['pipeline_size_mb']:.0f} MB; folded fp16 table directory {fo['folded_size_mb']:.1f} MB ({fo['folded_files_mb']}). "
                  "Fold checks (argmax agreement with the trained pipeline): " + "; ".join(
                      f"{s}: exact form (fp32 table{' ÷ document norm' if fo['normalize'] else ''}) {pct(c['table_fp32_exact_form_vs_pipeline_argmax_agreement'], 2)} "
                      f"(max |Δlogit| {c['table_fp32_exact_form_max_abs_logit_diff_vs_unfolded']:.1e}), fp16 table plain mean {pct(c['table_fp16_plain_mean_vs_pipeline_argmax_agreement'], 2)}, "
                      f"StaticKD inference code {pct(c['statickd_code_vs_pipeline_argmax_agreement'], 2)}" for s, c in fo["checks"].items()) + ".\n")
        w("| Model | Dev macro-F1 [95% CI] | Dev acc. | Held-out agr. | Test macro-F1 [95% CI] | Test acc. | Macro-F1/lang. | % teacher |\n|---|---|---|---|---|---|---|---|")
        for n, r in b1["rows"].items():
            w(f"| {DISPLAY.get(n, n)} | {f3(r['dev']['macro_f1'])} {ci(r['dev']['ci95'])} | {f3(r['dev']['accuracy'])} | {f3(r['heldout']['accuracy'])} | "
              f"{f3(r['test']['macro_f1'])} {ci(r['test']['ci95'])} | {f3(r['test']['accuracy'])} | {f3(r['test']['lang_macro_f1'])} | {pct(r['pct_teacher'])} |")
        w("\nPaired tests (single comparisons, unadjusted):\n")
        w("| Model | Test Δ StaticKD − model [95% CI], p | same, stratified | Test Δ Model2Vec default − model, p | Dev Δ StaticKD − model, p | Held-out agr. Δ StaticKD − model, p |\n|---|---|---|---|---|---|")
        for n, p in b1["paired"].items():
            w(f"| {DISPLAY.get(n, n)} | {f3(p['test_statickd_minus']['delta'], True)} {ci(p['test_statickd_minus']['ci95'], True)}, {fp(p['test_statickd_minus']['p'])} | "
              f"{ci(p['test_statickd_minus_stratified']['ci95'], True)}, {fp(p['test_statickd_minus_stratified']['p'])} | "
              f"{f3(p['test_model2vec_default_minus']['delta'], True)} {ci(p['test_model2vec_default_minus']['ci95'], True)}, {fp(p['test_model2vec_default_minus']['p'])} | "
              f"{f3(p['dev_statickd_minus']['delta'], True)}, {fp(p['dev_statickd_minus']['p'])} | {f3(p['heldout_acc_statickd_minus']['delta'], True)}, {fp(p['heldout_acc_statickd_minus']['p'])} |")
        w("")
    w("**What changes for the paper:** " + R["verdicts"]["B1"] + "\n")
    write_b1b(L, R)

    # ---- B2
    w("## B2. Size-matched multilingual data ablation (20,000 CC-News documents)\n")
    b2 = R.get("B2")
    if not b2:
        w("Not run.\n")
    else:
        sub = b2["train"]["subset"]
        w(f"Subset: {sub['subset_docs']:,} of {sub['pool_docs']:,} CC-News documents, {sub['subset_languages']} of {sub['pool_languages']} languages "
          f"({sub['rule']}, sampling seed {sub['sample_seed']}); by source {sub['by_source']}. Training: final recipe, {b2['train']['settings']['epochs']} epochs "
          f"(as emm_only), seeds {b2['train']['settings']['seeds']}. Mean training time "
          f"{np.mean([r['train_seconds'] for r in b2['train']['runs'].values()]):.0f} s.\n")
        w("| Variant | n seeds | Dev macro-F1 | Held-out agr. | Test macro-F1 | Test acc. |\n|---|---|---|---|---|---|")
        for g, lab in (("full", "Final recipe (one seed)"), ("emm_only", "EMMediaTopic only (4 languages, 20k)"), ("ccnews20k", "CC-News 20k, all languages (new)")):
            sm = b2["summary"][g]
            w(f"| {lab} | {len(sm['test_macro']['seeds'])} | " + " | ".join(f"{sm[m]['mean']:.3f}±{sm[m]['sd']:.3f}" for m in ("dev_macro", "heldout_acc", "test_macro", "test_acc")) + " |")
        w("\nPaired-seed bootstrap (Δ = CC-News 20k − reference; no Holm) and Welch p:\n")
        w("| Reference | Metric | Δ [95% CI] | p | TOST 90% CI, p | Welch p |\n|---|---|---|---|---|---|")
        for ref, blk in b2["paired"].items():
            for m, r in blk.items():
                w(f"| {ref.removeprefix('ccnews20k_minus_')} | {m} | {f3(r['delta'], True)} {ci(r['ci95'], True)} | {fp(r['p'])} | {ci(r['tost_ci90'], True)}, {fp(r['tost_p'])} | {fp(r['welch_p'])} |")
        w("\nTest macro-F1 by distillation data of the language (mean over seeds):\n")
        n = b2["tiers"]["n"]
        w(f"| Variant | Dev | sufficient (≥500), n={n['sufficient (≥500)']:,} | few (<500), n={n['few (<500)']:,} | none (0), n={n['none (0)']:,} |\n|---|---|---|---|---|")
        for g in ("full", "emm_only", "ccnews20k"):
            r = b2["tiers"][g]
            w(f"| {g} | {r['dev']:.3f} | {r['sufficient (≥500)']:.3f} | {r['few (<500)']:.3f} | {r['none (0)']:.3f} |")
        w("")
    w("**What changes for the paper:** " + R["verdicts"]["B2"] + "\n")

    # ---- C
    C = R.get("C") or {}
    raw = C.get("raw") or {}
    bench = C.get("benchmark", {})
    w("## C. CPU benchmarks\n")
    w("Protocol of statickd.benchmark / run_benchmarks.py (fresh process per model and thread count, pinned P-cores, thread limits in the "
      "environment, 100/96 documents with batch 8 for the ONNX transformers, 500/3,000 with batch 64 for the light models). Three repetitions "
      "in ONE later session (the paper used three separate sessions); the table gives the median and the 1-core min–max. StaticKD, the fp32 "
      "Model2Vec classifier and the fp32 ONNX transformers were rerun in the same session as references.\n")
    if bench:
        w("| Model | Latency p50 / p95, 1c (ms) | Throughput 1c (art./s) [min–max] | Throughput 4c | Model RAM 1c (MB) | File (MB) |\n|---|---|---|---|---|---|")
        for m, v in bench.items():
            one, four = v.get("1c", {}), v.get("4c", {})
            w(f"| {m} | {one.get('latency_ms_p50', float('nan')):.3g} / {one.get('latency_ms_p95', float('nan')):.3g} | "
              f"{one.get('docs_per_sec', float('nan')):.4g} [{one.get('dps_min', float('nan')):.4g}–{one.get('dps_max', float('nan')):.4g}] | "
              f"{four.get('docs_per_sec', float('nan')):.4g} | {one.get('rss_model_mb', float('nan')):.0f} | {one.get('size_mb', float('nan')):.0f} |")
        w("")
    w("### C1. Model2Vec classifier in fp16 / int8\n")
    c1 = raw.get("c1_model2vec")
    if c1:
        for tag in ("fp16", "int8"):
            r = c1.get(tag, {})
            if not r.get("loads"):
                w(f"- {tag}: does not load ({r.get('error')}).")
                continue
            t = r["test"]
            pr = C["paired"].get(f"model2vec_clf_{tag}", {})
            w(f"- {tag}: directory {r['size_mb']:.0f} MB (fp32 {c1['fp32_size_mb']:.0f} MB), loaded dtype {r['loaded_embedding_dtype']}; test macro-F1 {f3(t['macro_f1'])} "
              f"(fp32 {f3(t['fp32']['macro_f1'])}), accuracy {f3(t['accuracy'])} (fp32 {f3(t['fp32']['accuracy'])}); argmax changes vs fp32 "
              f"{t['argmax_changes_vs_fp32']} of {t['n']:,}; dev argmax changes {r['dev']['argmax_changes_vs_fp32']}"
              + (f"; Δ macro-F1 {tag} − fp32 {f3(pr['vs_fp32']['delta'], True)} {ci(pr['vs_fp32']['ci95'], True)}, p {fp(pr['vs_fp32']['p'])}; "
                 f"Δ StaticKD − {tag} {f3(pr['statickd_minus']['delta'], True)} {ci(pr['statickd_minus']['ci95'], True)}, p {fp(pr['statickd_minus']['p'])}" if pr else "") + ".")
        w("")
    w("### C2. MiniLM-L6 + KD and DistilmBERT + KD, ONNX int8 (quantize_dynamic)\n")
    c2 = raw.get("c2_onnx_int8")
    if c2:
        w("| Model | ONNX fp32 / int8 (MB) | Test macro-F1 fp32 ONNX / int8 | Test acc. fp32 ONNX / int8 | Argmax changes int8 vs fp32 ONNX | Δ int8 − fp32 ONNX [95% CI], p | Δ StaticKD − int8 [95% CI], p | Δ acc. StaticKD − int8, p |\n|---|---|---|---|---|---|---|---|")
        for m, r in c2.items():
            t = r["test"]
            pr = C["paired"].get(f"{m}_onnx_int8", {})
            w(f"| {m} | {r['fp32_onnx_mb']:.0f} / {r['int8_onnx_mb']:.0f} | {f3(t['onnx_fp32']['macro_f1'])} / {f3(t['onnx_int8']['macro_f1'])} | "
              f"{f3(t['onnx_fp32']['accuracy'])} / {f3(t['onnx_int8']['accuracy'])} | {t['argmax_changes_int8_vs_onnx_fp32']} of {t['n']:,} | "
              f"{f3(pr['vs_onnx_fp32']['delta'], True)} {ci(pr['vs_onnx_fp32']['ci95'], True)}, {fp(pr['vs_onnx_fp32']['p'])} | "
              f"{f3(pr['statickd_minus']['delta'], True)} {ci(pr['statickd_minus']['ci95'], True)}, {fp(pr['statickd_minus']['p'])} | "
              f"{f3(pr['statickd_minus_accuracy']['delta'], True)}, {fp(pr['statickd_minus_accuracy']['p'])} |")
        w("\nStored (PyTorch GPU) fp32 test scores and argmax changes of the ONNX fp32 CPU path vs the stored predictions: " + "; ".join(
            f"{m}: stored {f3(r['test']['stored_pytorch_gpu']['macro_f1'])}, ONNX fp32 changes {r['test']['argmax_changes_onnx_fp32_vs_stored']}, "
            f"int8 Δ vs stored {f3(C['paired'][f'{m}_onnx_int8']['vs_stored_fp32']['delta'], True)} (p {fp(C['paired'][f'{m}_onnx_int8']['vs_stored_fp32']['p'])})"
            for m, r in c2.items()) + ". Quantized graph ops: " + "; ".join(f"{m} {r['int8_ops']}" for m, r in c2.items()) + ".\n")
    dg = C.get("minilm_diag")
    if dg:
        w("MiniLM-L6 quantization diagnostic (dev macro-F1; ONNX fp32 " + f3(c2["minilm_kd"]["dev"]["onnx_fp32"]["macro_f1"]) + ", teacher-style int8 "
          + f3(c2["minilm_kd"]["dev"]["onnx_int8"]["macro_f1"]) + "): " + "; ".join(
              f"{k} {dg[k]['settings']} {f3(dg[k]['dev']['macro_f1'])} ({dg[k]['onnx_mb']:.0f} MB, {dg[k]['dev']['argmax_changes_vs_onnx_fp32']} argmax changes)"
              for k in dg if isinstance(dg[k], dict) and "settings" in dg[k]) + f". Best on dev: {dg['best_on_dev']}, test macro-F1 "
          f"{f3(dg['best_test']['macro_f1'])}, accuracy {f3(dg['best_test']['accuracy'])}, {dg['best_test']['argmax_changes_vs_onnx_fp32']} argmax changes vs ONNX fp32"
          + (f"; Δ StaticKD − it {f3(C['minilm_diag_paired']['delta'], True)} {ci(C['minilm_diag_paired']['ci95'], True)}, p {fp(C['minilm_diag_paired']['p'])}"
             if C.get("minilm_diag_paired") else "") + ".\n")
    w("### C3. Folded linear-head Model2Vec table\n")
    w(R["verdicts"].get("C3_text", "") + "\n")
    w("**What changes for the paper:** " + R["verdicts"]["C"] + "\n")

    w("## Run times\n")
    w(R["runtimes"] + "\n")
    OUT_MD.write_text("\n".join(L), encoding="utf8")


def main() -> None:
    numbers = json.load(open(ROOT / "results/final/paper_numbers.json", encoding="utf8"))
    test = ev.eval_view("test")
    pool_langs = load_distillation_pool()["lang"].value_counts()
    test = test.join(an.language_info(test, pool_langs)[["distil_tier"]], on="lang")
    R = {"A": load(RE / "r5_a.json")}
    b1 = load(RE / "r5_b1.json")
    R["B1"] = b1_stats(b1) if b1 else None
    b1b = load(RE / "r5_b1b.json")
    R["B1b"] = b1b_stats(b1b) if b1b and "m2v_lin_nonorm_ens012" in b1b else None
    b2 = load(RE / "r5_b2_train.json")
    R["B2"] = b2_stats(b2, test) if b2 and len(b2.get("runs", {})) == len(ex.ABLATION_SEEDS) else None
    R["C"] = c_stats(load(RE / "r5_c_prepare.json"))
    notes = load(RE / "r5_notes.json") or {}
    R["A10"] = notes.get("A10", "")
    R["verdicts"] = notes.get("verdicts", {})
    R["runtimes"] = notes.get("runtimes", "")
    for k in ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "B1", "B2", "C"):
        R["verdicts"].setdefault(k, "(to be written)")
    OUT_JSON.write_text(json.dumps(jsonable(R), indent=1, ensure_ascii=False), encoding="utf8")
    write_md(R, numbers)
    print("written", OUT_JSON, OUT_MD)


if __name__ == "__main__":
    main()
