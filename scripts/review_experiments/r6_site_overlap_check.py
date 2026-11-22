"""Review round 6: test set without the BBC articles whose site and language occur in the distillation data.

Main text (Section VI-B) reports a check without the 694 MasakhaNEWS articles from bbc.com whose language
(Amharic, Somali, Swahili) has bbc.com documents in the distillation data. This script recomputes it.

Exclusion rule (same host and language matching as r3_reviewer_checks.py item 1): a test article is removed
when its host (lower case, leading "www." stripped) is a BBC host (bbc.com, bbc.co.uk) and the pair
(host, language) occurs in the CC-News distillation data (train + extra). A sensitivity variant removes every
article whose (host, language) pair occurs in the distillation data or the held-out set, whatever the site
(r3 column "Same site and same language", 695 articles).

Bootstrap conventions as the notebook and r4_reviewer_checks.py: site clusters from data.site_clusters,
seed 0, 2,000 resamples for paired tests (95% CI, two-sided p and TOST from all 2,000 differences, as
ev.paired_bootstrap / ev.tost), per-model 95% CIs from the first 1,000 resamples (= ev.bootstrap_ci).
All statistics come from one pass over ev.bootstrap_samples, so the draws equal those of the ev functions.

Run from the project root:  env/Scripts/python.exe scripts/review_experiments/r6_site_overlap_check.py
Output: results/review_experiments/r6_site_overlap_check.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from statickd import evaluation as ev  # noqa: E402
from statickd.data import load_ccnews_heldout, load_distillation_pool  # noqa: E402

OUT_JSON = ROOT / "results" / "review_experiments" / "r6_site_overlap_check.json"
N_BOOT = 1000
N_PAIRED = 2 * N_BOOT
MARGIN = 0.01
BBC_HOSTS = {"bbc.com", "bbc.co.uk"}
MODELS = {"statickd": "statickd", "teacher": "teacher", "minilm_kd": "minilm_kd_t1", "model2vec_clf": "model2vec_clf"}
COMPARE = ["teacher", "minilm_kd", "model2vec_clf"]


def host(d) -> str:
    d = ("" if d is None or (isinstance(d, float) and np.isnan(d)) else str(d)).lower().strip()
    return d[4:] if d.startswith("www.") else d


def analyse(y, clusters, preds: dict) -> dict:
    vals = {k: np.empty(N_PAIRED) for k in preds}
    for b, i in enumerate(ev.bootstrap_samples(len(y), clusters, N_PAIRED, 0)):
        yi = y[i]
        for k, p in preds.items():
            vals[k][b] = ev.macro_f1(yi, p[i])
    full = {k: ev.macro_f1(y, p) for k, p in preds.items()}
    out = {"n": int(len(y)), "n_clusters": int(len(np.unique(clusters))),
           "effective_clusters": ev.effective_clusters(clusters),
           "macro_f1": {k: {"value": full[k], "ci95_first1000": [float(v) for v in np.percentile(vals[k][:N_BOOT], [2.5, 97.5])],
                            "accuracy": float((y == preds[k]).mean())} for k in preds},
           "statickd_minus": {}}
    for m in COMPARE:
        diffs = vals["statickd"] - vals[m]
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        r = {"delta": full["statickd"] - full[m], "ci95": [float(lo), float(hi)], "p_two_sided": float(min(1.0, p)),
             "n_boot": N_PAIRED}
        if m == "model2vec_clf":
            t = ev.tost(diffs, MARGIN)
            r["tost"] = {"margin": MARGIN, "ci90": [t["ci90_low"], t["ci90_high"]], "p": t["p"], "equivalent": t["equivalent"],
                         "tail_count": max(int((diffs <= -MARGIN).sum()), int((diffs >= MARGIN).sum()))}
        out["statickd_minus"][m] = r
    return out


def main() -> None:
    test = ev.eval_set("test")
    y = test["y"].to_numpy()
    clusters = np.asarray(test["cluster"])
    preds = {k: ev.load_pred(v, "test") for k, v in MODELS.items()}

    pool = load_distillation_pool()
    cc = pool[pool.source != "emmediatopic"]
    ho = load_ccnews_heldout()
    train_pairs = set(zip(cc.domain.map(host), cc.lang))
    any_pairs = train_pairs | set(zip(ho.domain.map(host), ho.lang))
    h = test["domain"].map(host)
    known = h.ne("")
    pair_train = pd.Series([(a, b) in train_pairs for a, b in zip(h, test["lang"])], index=test.index)
    pair_any = pd.Series([(a, b) in any_pairs for a, b in zip(h, test["lang"])], index=test.index)
    drop_main = (known & h.isin(BBC_HOSTS) & pair_train).to_numpy()
    drop_any = (known & pair_any).to_numpy()

    result = {"script": "scripts/review_experiments/r6_site_overlap_check.py",
              "conventions": "site-cluster paired bootstrap (data.site_clusters), seed 0, 2,000 resamples for paired "
                             "tests and TOST, per-model 95% CI from the first 1,000 resamples; delta = StaticKD - model; "
                             "minilm_kd = minilm_kd_t1 predictions; no multiplicity correction",
              "subsets": {}}
    for name, drop in (("full_test", np.zeros(len(y), bool)),
                       ("without_bbc_same_site_same_language_in_distillation", drop_main),
                       ("sensitivity_without_any_same_site_same_language_in_distillation_or_heldout", drop_any)):
        keep = ~drop
        removed = test.loc[drop]
        r = analyse(y[keep], clusters[keep], {k: p[keep] for k, p in preds.items()})
        r["removed"] = {"n": int(drop.sum()),
                        "by_source": removed["source"].value_counts().to_dict(),
                        "by_lang": removed["lang"].value_counts().to_dict(),
                        "by_host": h[drop].value_counts().to_dict()}
        result["subsets"][name] = r
        print(name, r["n"], {m: (round(v["delta"], 4), [round(x, 4) for x in v["ci95"]], v["p_two_sided"])
                             for m, v in r["statickd_minus"].items()},
              "TOST", r["statickd_minus"]["model2vec_clf"]["tost"])

    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    OUT_JSON.write_text(json.dumps(conv(result), indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
