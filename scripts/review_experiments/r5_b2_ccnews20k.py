"""Review round 5, B2: size-matched multilingual data ablation.

The StaticKD recipe trained on 20,000 documents of the CC-News part of the distillation data (sources ccnews +
ccnews_extra, 265,999 documents, all their languages), with the settings of the "emm_only" ablation (final recipe,
10 epochs) and its 5 seeds (experiments.ABLATION_SEEDS). The 20,000 documents are drawn once (sampling seed 0),
stratified by language in proportion to the language shares of the CC-News part (largest-remainder rounding,
at least one document per language), and are the same for every training seed.

Training uses statickd.student.train unchanged; only the data loader inside the student module is replaced by the
fixed subset, and the token cache is bypassed (nothing is written to data/). Models go to models/review_r5/, the
logits to results/review_experiments/r5_b2_predictions/<run>__<set>.npz.

Run from the project root:  env/Scripts/python.exe scripts/review_experiments/r5_b2_ccnews20k.py
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

from statickd import evaluation as ev  # noqa: E402
from statickd import experiments as ex  # noqa: E402
from statickd import student  # noqa: E402
from statickd.config import MODELS_DIR  # noqa: E402
from statickd.data import load_distillation_pool  # noqa: E402

N_DOCS = 20_000
SAMPLE_SEED = 0
OUT_MODELS = MODELS_DIR / "review_r5"
PRED_DIR = ROOT / "results/review_experiments/r5_b2_predictions"
OUT_JSON = ROOT / "results/review_experiments/r5_b2_train.json"


def sample_subset() -> tuple[pd.DataFrame, dict]:
    pool = load_distillation_pool(("ccnews", "ccnews_extra"))
    share = pool["lang"].value_counts()
    quota = share / share.sum() * N_DOCS
    n = np.maximum(np.floor(quota).astype(int), 1)
    rest = N_DOCS - int(n.sum())
    if rest > 0:   # largest remainder
        order = (quota - np.floor(quota)).sort_values(ascending=False).index
        n.loc[order[:rest]] += 1
    elif rest < 0:  # minimum-one rule overshot: take back from the largest languages
        for lang in share.index[: -rest]:
            n.loc[lang] -= 1
    assert int(n.sum()) == N_DOCS and (n <= share).all()
    rng = np.random.default_rng(SAMPLE_SEED)
    keep = np.concatenate([rng.choice(np.flatnonzero(pool["lang"].to_numpy() == lang), int(k), replace=False) for lang, k in n.items()])
    sub = pool.iloc[np.sort(keep)].reset_index(drop=True)
    info = {"pool_docs": len(pool), "pool_languages": int(share.size), "subset_docs": len(sub), "subset_languages": int(sub.lang.nunique()),
            "by_source": sub.source.value_counts().to_dict(), "top10_languages": sub.lang.value_counts().head(10).to_dict(),
            "allocation": {k: int(v) for k, v in n.items()}, "sample_seed": SAMPLE_SEED,
            "rule": "proportional to the CC-News language shares, largest remainder, >= 1 per language"}
    return sub, info


def main() -> None:
    from statickd.student import StaticKDConfig, StaticKDModel

    sub, info = sample_subset()
    print(json.dumps({k: v for k, v in info.items() if k != "allocation"}), flush=True)
    student.load_distillation_pool = lambda sources=None: sub          # fixed subset for every seed
    student.encode_cached = lambda tok, texts, key: student.encode(tok, texts)   # no token cache written
    emm = ex.ABLATIONS["emm_only"][1]
    res = {"subset": info, "runs": {}, "settings": {"epochs": emm["epochs"], "seeds": list(ex.ABLATION_SEEDS),
                                                     "other": "final recipe defaults (StaticKDConfig)"}}
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    sets = {s: ev.eval_set(s)["text"].tolist() for s in ev.EVAL_SETS}
    for seed in ex.ABLATION_SEEDS:
        name = f"ccnews20k_s{seed}"
        cfg = StaticKDConfig(name=name, seed=seed, sources=("ccnews", "ccnews_extra"), epochs=emm["epochs"])
        t0 = time.time()
        out = student.train(cfg, out_dir=OUT_MODELS / name)
        meta = json.loads((out / "meta.json").read_text(encoding="utf8"))
        model = StaticKDModel(out)
        scores = {}
        for s, texts in sets.items():
            L = np.concatenate([model.logits(texts[i:i + 512]) for i in range(0, len(texts), 512)]).astype(np.float32)
            np.savez_compressed(PRED_DIR / f"{name}__{s}.npz", logits=L)
            view = ev.eval_view(s)
            p = L[view.index.to_numpy()].argmax(1)
            scores[s] = {"macro_f1": ev.macro_f1(view.y.to_numpy(), p), "accuracy": float((view.y == p).mean())}
        res["runs"][name] = {"train_docs": meta["train_docs"], "best_val_agreement": meta["best_val_agreement"],
                             "train_seconds": meta["train_seconds"], "device": meta["device"], "scores": scores,
                             "wall_seconds": round(time.time() - t0, 1)}
        print(name, json.dumps(res["runs"][name]), flush=True)
        OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf8")
    print("written", OUT_JSON)


if __name__ == "__main__":
    main()
