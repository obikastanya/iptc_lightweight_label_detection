"""Review round 5, B1b: more seeds of the foldable linear-head Model2Vec classifier and its folded 3-seed ensemble.

  --seed S     train m2v_lin_nonorm with seed S (n_layers=0, normalize=False, hard teacher labels, everything else as
               r5_b1_model2vec_linear.py / scripts/train_model2vec.py), evaluate the pipeline, fold it exactly into a fp16
               table (same checks as B1) and evaluate the folded table with the StaticKD inference code.
               Seed 0 is the existing models/review_r5/m2v_lin_nonorm (B1); seeds 1, 2 -> m2v_lin_nonorm_s1, _s2.
  --ensemble   average the three folded tables and biases (fp32 mean -> fp16 table, fp32 bias; the arithmetic of
               statickd.student.ensemble, but with max_tokens 512 kept from the members) and evaluate it.

Outputs: models/review_r5/m2v_lin_nonorm_s{1,2}[_folded], models/review_r5/m2v_lin_nonorm_ens012_folded,
         results/review_experiments/r5_b1_predictions/<name>__<set>.npz, results/review_experiments/r5_b1b.json
Run from the project root (one process per seed):
    env/Scripts/python.exe scripts/review_experiments/r5_b1b_seeds.py --seed 1
    env/Scripts/python.exe scripts/review_experiments/r5_b1b_seeds.py --seed 2
    env/Scripts/python.exe scripts/review_experiments/r5_b1b_seeds.py --ensemble
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "review_experiments"))

import numpy as np  # noqa: E402

from r5_b1_model2vec_linear import OUT_MODELS, VARIANTS, evaluate, fold, train_variant  # noqa: E402
from statickd.baselines import Model2VecPredictor, dir_size_mb  # noqa: E402
from statickd.config import LABELS  # noqa: E402
from statickd.data import load_distillation_pool, teacher_probs  # noqa: E402

OUT_JSON = ROOT / "results/review_experiments/r5_b1b.json"
MEMBERS = ["m2v_lin_nonorm", "m2v_lin_nonorm_s1", "m2v_lin_nonorm_s2"]
ENS = "m2v_lin_nonorm_ens012"


def save(res):
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ensemble", action="store_true")
    args = parser.parse_args()
    res = json.loads(OUT_JSON.read_text(encoding="utf8")) if OUT_JSON.exists() else {}
    from statickd.student import StaticKDModel

    if args.seed is not None:
        assert args.seed in (1, 2)
        name = f"m2v_lin_nonorm_s{args.seed}"
        pool = load_distillation_pool()
        texts = pool["text"].tolist()
        y = teacher_probs(pool).argmax(1).astype(int).tolist()
        r = {"train": train_variant(name, VARIANTS["m2v_lin_nonorm"], texts, y, seed=args.seed)}
        del pool, texts, y
        r["eval_pipeline"] = evaluate(name, Model2VecPredictor(OUT_MODELS / name), batch_size=1024)
        r["fold"] = fold(name)
        r["eval_folded_statickd_code"] = evaluate(f"{name}_folded", StaticKDModel(OUT_MODELS / f"{name}_folded"), batch_size=512)
        res[name] = r
        save(res)
    if args.ensemble:
        dirs = [OUT_MODELS / f"{m}_folded" for m in MEMBERS]
        datas = [np.load(d / "model.npz") for d in dirs]
        metas = [json.loads((d / "meta.json").read_text(encoding="utf8")) for d in dirs]
        assert all("token_table" in d and "token_weight" not in d for d in datas)
        assert len({m["max_tokens"] for m in metas}) == 1
        out = OUT_MODELS / f"{ENS}_folded"
        out.mkdir(parents=True, exist_ok=True)
        table = np.mean([d["token_table"].astype(np.float32) for d in datas], axis=0).astype(np.float16)
        bias = np.mean([d["bias"].astype(np.float32) for d in datas], axis=0)
        np.savez(out / "model.npz", token_table=table, bias=bias)
        (out / "tokenizer.json").write_bytes((dirs[0] / "tokenizer.json").read_bytes())
        (out / "meta.json").write_text(json.dumps({"format": "token_table_fp16", "labels": LABELS, "max_tokens": metas[0]["max_tokens"],
                                                   "ensemble_of": [d.name for d in dirs]}, indent=1), encoding="utf8")
        r = {"members": [d.name for d in dirs], "size_mb": round(dir_size_mb(out), 2)}
        r["eval_folded_statickd_code"] = evaluate(f"{ENS}_folded", StaticKDModel(out), batch_size=512)
        # linearity check: ensemble logits = mean of member logits (up to fp16 rounding of the averaged table)
        pd_ = ROOT / "results/review_experiments/r5_b1_predictions"
        chk = {}
        for s in ("dev", "test"):
            L = np.load(pd_ / f"{ENS}_folded__{s}.npz")["logits"]
            avg = np.mean([np.load(pd_ / f"{m}_folded__{s}.npz")["logits"] for m in MEMBERS], axis=0)
            chk[s] = {"max_abs_logit_diff": float(np.abs(L - avg).max()), "argmax_agreement": float((L.argmax(1) == avg.argmax(1)).mean())}
        r["check_vs_mean_of_member_logits"] = chk
        print(json.dumps(r), flush=True)
        res[ENS] = r
        save(res)


if __name__ == "__main__":
    main()
