"""E1 analysis: the Sparse-Distillation-style baseline next to StaticKD, with the paper's protocol.

Mirrors src/notebook_build/part3.py-part4.py:
  * main-table numbers (analysis.score_row, 1,000 resamples; site-cluster bootstrap on test,
    document bootstrap on dev), held-out agreement with the teacher, macro-F1 per language, % teacher;
  * the "sig" family: paired bootstrap (2,000 resamples, seed 0, Delta = StaticKD - model) + exact
    McNemar, Holm per set, recomputed with the new comparisons added to the family;
  * TOST +-0.01 on test (paired_bootstrap_seeds diffs, 2,000 resamples, as part4.py);
  * subgroups by distillation data of the language and by teacher pre-training coverage;
  * 3-seed mean +- SD; size check (large vs small, seed-averaged paired bootstrap + Welch).

Usage (from the project root):
    env/Scripts/python.exe scripts/review_experiments/e1_analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from statickd import analysis as an  # noqa: E402
from statickd import evaluation as ev  # noqa: E402
from statickd.data import load_distillation_pool  # noqa: E402

N_BOOT = 1000
OUT = ROOT / "results" / "review_experiments" / "e1" / "analysis.json"
SIZES = ("small", "large")
SEEDS = (0, 1, 2)
NEW = {f"sdstyle_{s}": f"sdstyle_{s}_s0" for s in SIZES}


def minilm_key() -> str:
    """As part3.py: the MiniLM temperature chosen on dev."""
    dev = {t: ev.macro_f1(ev.eval_view("dev").y, ev.pred_view(f"minilm_kd_t{t}", "dev"))
           for t in (1, 2) if ev.has_predictions(f"minilm_kd_t{t}", "dev")}
    return f"minilm_kd_t{max(dev, key=dev.get)}"


MAIN = ["teacher", "teacher_onnx_int8", "distilmbert_kd", "minilm_kd", "statickd", "statickd_compact",
        "ft_kd_subword_q", "ft_kd_subword", "ft_kd_word", "tfidf_svm_kd", "model2vec_clf", "ft_gpt_subword", "ft_gpt_word"]
KEYS = {"minilm_kd": minilm_key(), **NEW}


def key(m: str) -> str:
    return KEYS.get(m, m)


def main_rows(models) -> dict:
    out = {}
    teacher = {s: an.score_row("teacher", s, n_boot=N_BOOT) for s in ("dev", "test")}
    for m in models:
        r = {}
        for s in ("dev", "heldout", "test"):
            if not ev.has_predictions(key(m), s):
                continue
            sr = an.score_row(key(m), s, n_boot=N_BOOT)
            r[s] = {k: sr[k] for k in ("macro_f1", "accuracy", "ci_low", "ci_high", "n")}
            if s == "test":
                r[s]["lang_macro_f1"] = sr["lang_macro_f1"]
                r[s]["pct_teacher"] = sr["macro_f1"] / teacher["test"]["macro_f1"]
            if s == "dev":
                r[s]["pct_teacher"] = sr["macro_f1"] / teacher["dev"]["macro_f1"]
        out[m] = r
    return out


def sig_family(models_in_family) -> pd.DataFrame:
    """part4.py 8.2, for an arbitrary list of comparison models."""
    rows = []
    for s in ("dev", "test"):
        view = ev.eval_view(s)
        y = view.y.to_numpy()
        clusters = view.cluster.to_numpy() if s == "test" else None
        p_s = ev.pred_view("statickd", s)
        block = []
        for m in [m for m in models_in_family if m != "statickd" and ev.has_predictions(key(m), s)]:
            p_m = ev.pred_view(key(m), s)
            pb = ev.paired_bootstrap(y, p_s, p_m, clusters=clusters, n_boot=2 * N_BOOT)
            mc = ev.mcnemar(y, p_s, p_m)
            block.append({"set": s, "vs": m, **{f"f1_{k}": v for k, v in pb.items()},
                          "acc_delta": float((p_s == y).mean() - (p_m == y).mean()),
                          "only_static": mc["only_a"], "only_other": mc["only_b"], "p_mcnemar": mc["p"]})
        padj = ev.holm([b["f1_p"] for b in block])
        madj = ev.holm([b["p_mcnemar"] for b in block])
        for b, p1, p2 in zip(block, padj, madj):
            b["p_boot_holm"], b["p_mcnemar_holm"] = float(p1), float(p2)
        rows += block
    return pd.DataFrame(rows)


def tost_test(model: str) -> dict:
    view = ev.eval_view("test")
    y, cl = view.y.to_numpy(), view.cluster.to_numpy()
    pb = ev.paired_bootstrap_seeds(y, [ev.pred_view("statickd", "test")], [ev.pred_view(model, "test")],
                                   clusters=cl, n_boot=2 * N_BOOT)
    t = ev.tost(pb["diffs"], 0.01)
    return {"delta": pb["delta"], "ci95": [pb["ci_low"], pb["ci_high"]], "ci90": [t["ci90_low"], t["ci90_high"]],
            "tost_p": t["p"], "equivalent": t["equivalent"]}


def seed_stats() -> dict:
    out = {}
    for size in SIZES:
        rows = []
        for seed in SEEDS:
            name = f"sdstyle_{size}_s{seed}"
            if not all(ev.has_predictions(name, s) for s in ev.EVAL_SETS):
                continue
            r = {"seed": seed}
            for s in ev.EVAL_SETS:
                y, p = ev.eval_view(s).y.to_numpy(), ev.pred_view(name, s)
                r[f"{s}_macro"], r[f"{s}_acc"] = ev.macro_f1(y, p), float((y == p).mean())
            rows.append(r)
        df = pd.DataFrame(rows)
        out[size] = {"per_seed": rows, "n_seeds": len(rows),
                     **{c: {"mean": float(df[c].mean()), "sd": float(df[c].std(ddof=1)) if len(df) > 1 else None}
                        for c in df.columns if c != "seed"}}
    full = an.seed_table(codes=["full"])
    out["statickd_single_10seeds"] = {c: {"mean": float(full[c].mean()), "sd": float(full[c].std(ddof=1))}
                                      for c in ["dev_macro", "dev_acc", "heldout_macro", "heldout_acc", "test_macro", "test_acc"]}
    return out


def size_check() -> dict:
    out = {}
    for s in ("dev", "test"):
        view = ev.eval_view(s)
        y = view.y.to_numpy()
        cl = view.cluster.to_numpy() if s == "test" else None
        a = [ev.pred_view(f"sdstyle_large_s{k}", s) for k in SEEDS]
        b = [ev.pred_view(f"sdstyle_small_s{k}", s) for k in SEEDS]
        pb = ev.paired_bootstrap_seeds(y, a, b, clusters=cl, n_boot=2 * N_BOOT)
        w = ev.welch([ev.macro_f1(y, p) for p in a], [ev.macro_f1(y, p) for p in b])
        s0 = ev.paired_bootstrap(y, a[0], b[0], clusters=cl, n_boot=2 * N_BOOT)
        out[s] = {"seed_avg_delta_large_minus_small": pb["delta"], "ci95": [pb["ci_low"], pb["ci_high"]], "p_boot": pb["p"],
                  "welch_p": w["p"], "seed0_delta": s0["delta"], "seed0_ci95": [s0["ci_low"], s0["ci_high"]], "seed0_p": s0["p"]}
    return out


def subgroups() -> dict:
    test = ev.eval_set("test")
    pool_langs = load_distillation_pool()["lang"].value_counts()
    lang_info = an.language_info(test, pool_langs)
    test = test.join(lang_info[["distil_tier", "xlmr"]], on="lang")
    models = ["teacher", "statickd", "sdstyle_small", "sdstyle_large"]
    out = {}
    for col in ("distil_tier", "xlmr"):
        g = an.subgroup_scores([key(m) for m in models], col, test)
        g["model"] = g["model"].map({key(m): m for m in models})
        wide = g.pivot(index=col, columns="model", values="macro_f1")[models]
        wide.insert(0, "n", g.groupby(col).n.first())
        wide.insert(1, "langs", test.groupby(col).lang.nunique())
        for m in models[1:]:
            wide[f"{m} % teacher"] = wide[m] / wide["teacher"]
        wide.index = [an.en(i) for i in wide.index]
        out[col] = json.loads(wide.to_json(orient="index"))
        print(f"\n== subgroups by {col}\n", wide.round(3).to_string())
    return out


def main() -> None:
    sanity = {m: an.score_row(m, "test", n_boot=10)["macro_f1"] for m in ("statickd", "teacher")}
    print("sanity test macro-F1:", {k: round(v, 3) for k, v in sanity.items()})
    assert round(sanity["statickd"], 3) == 0.545 and round(sanity["teacher"], 3) == 0.598

    table = main_rows(["teacher", "statickd", "sdstyle_small", "sdstyle_large"])
    print(json.dumps(table, indent=1))

    base = sig_family(MAIN)
    fams = {"paper": base,
            "plus_small": sig_family(MAIN + ["sdstyle_small"]),
            "plus_large": sig_family(MAIN + ["sdstyle_large"]),
            "plus_both": sig_family(MAIN + ["sdstyle_small", "sdstyle_large"])}
    changes = {}
    for name, fam in fams.items():
        if name == "paper":
            continue
        merged = fam.merge(base[["set", "vs", "p_boot_holm", "p_mcnemar_holm"]], on=["set", "vs"], how="left", suffixes=("", "_paper"))
        flips = merged[merged.p_boot_holm_paper.notna() & (((merged.p_boot_holm < 0.05) != (merged.p_boot_holm_paper < 0.05))
                                                           | ((merged.p_mcnemar_holm < 0.05) != (merged.p_mcnemar_holm_paper < 0.05)))]
        changes[name] = flips[["set", "vs", "p_boot_holm_paper", "p_boot_holm", "p_mcnemar_holm_paper", "p_mcnemar_holm"]].to_dict("records")
    both = fams["plus_both"].merge(base[["set", "vs", "p_boot_holm", "p_mcnemar_holm"]], on=["set", "vs"], how="left",
                                   suffixes=("", "_paper"))
    pd.set_option("display.width", 250)
    print("\n== sig family with both sdstyle sizes added (Delta = StaticKD - model)\n",
          both.drop(columns=["f1_p", "p_mcnemar"]).round(4).to_string())
    print("\nsignificance flips vs the paper family:", json.dumps(changes, indent=1))

    tost = {m: tost_test(key(m)) for m in NEW}
    print("\nTOST:", json.dumps(tost, indent=1))
    seeds = seed_stats()
    print("\nseeds:", json.dumps(seeds, indent=1))
    size = size_check()
    print("\nsize check:", json.dumps(size, indent=1))
    groups = subgroups()

    result = {"sanity_test_macro_f1": sanity, "minilm_key": key("minilm_kd"), "main_table": table,
              "sig_families": {k: v.to_dict("records") for k, v in fams.items()},
              "significance_flips_vs_paper": changes, "tost_test": tost, "seeds": seeds, "size_check": size,
              "subgroups": groups}
    OUT.write_text(json.dumps(result, indent=1, default=float), encoding="utf8")
    print("written", OUT)


if __name__ == "__main__":
    main()
