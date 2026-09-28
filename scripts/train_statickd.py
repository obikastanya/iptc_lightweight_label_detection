"""Train every StaticKD run of the registry (statickd.experiments) and store its predictions.

Runs that already have a model and all predictions are skipped, so the script can be restarted.
Several workers can share one GPU:  --worker 0 --num-workers 2  and  --worker 1 --num-workers 2.

Usage (from the project root):
    python scripts/train_statickd.py                      # all runs
    python scripts/train_statickd.py --codes full tau2    # a subset
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statickd.evaluation import EVAL_SETS, evaluate_model, has_predictions  # noqa: E402
from statickd.experiments import ABLATIONS, all_runs, config_for, run_name  # noqa: E402
from statickd.student import STUDENTS_DIR, StaticKDModel, train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", nargs="*", default=list(ABLATIONS))
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    runs = [r for r in all_runs() if r[0] in args.codes]
    runs = runs[args.worker::args.num_workers]
    for k, (code, seed) in enumerate(runs, 1):
        name = run_name(code, seed)
        model_dir = STUDENTS_DIR / name
        if (model_dir / "model.npz").exists() and all(has_predictions(name, s) for s in EVAL_SETS):
            print(f"[{k}/{len(runs)}] {name}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        if not (model_dir / "model.npz").exists():
            train(config_for(code, seed), out_dir=model_dir, verbose=False)
        summary = evaluate_model(name, StaticKDModel(model_dir))
        line = " ".join(f"{s}={summary[s]['macro_f1']:.4f}/{summary[s]['accuracy']:.4f}" for s in EVAL_SETS)
        print(f"[{k}/{len(runs)}] {name}: {line} ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
