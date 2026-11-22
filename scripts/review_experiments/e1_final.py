"""E1 final runs: the configuration selected on dev (tuning.json) for each size, seeds 0, 1, 2.

Each model is saved in models/review_e1/sdstyle_<size>_s<seed>/ (fp16 embeddings) and evaluated with
the NumPy predictor through statickd.evaluation.evaluate_model, so its logits are stored as
results/final/predictions/sdstyle_<size>_s<seed>__{dev,heldout,test}.npz. The reported model of
each size is seed 0.

Usage (from the project root):
    env/Scripts/python.exe scripts/review_experiments/e1_final.py --size small --seeds 0 1 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from statickd.evaluation import EVAL_SETS, evaluate_model, has_predictions  # noqa: E402
from statickd.sdstyle import SD_DIR, SDConfig, SDStyleModel, train  # noqa: E402

TUNING = ROOT / "results" / "review_experiments" / "e1" / "tuning.json"
SIZES = {"small": {"dim": 128, "buckets": 524_288}, "large": {"dim": 128, "buckets": 1_600_000}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(SIZES), required=True)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    sel = json.loads(TUNING.read_text(encoding="utf8"))["selected"][args.size]
    for seed in args.seeds:
        name = f"sdstyle_{args.size}_s{seed}"
        out = SD_DIR / name
        t0 = time.perf_counter()
        if not (out / "embeddings.npy").exists():
            cfg = SDConfig(name=name, seed=seed, hidden=sel["hidden"], lr_emb=sel["lr_emb"], epochs=sel["max_epochs"],
                           **SIZES[args.size])
            train(cfg, out_dir=out)
        if not all(has_predictions(name, s) for s in EVAL_SETS):
            summary = evaluate_model(name, SDStyleModel(out))
        else:
            summary = evaluate_model(name, None)
        line = " ".join(f"{s}={summary[s]['macro_f1']:.4f}/{summary[s]['accuracy']:.4f}" for s in EVAL_SETS)
        print(f"{name}: {line} ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
