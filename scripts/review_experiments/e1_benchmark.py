"""E1 CPU benchmark: StaticKD and the two Sparse-Distillation-style models in the same session.

Paper protocol (scripts/run_benchmarks.py + statickd.benchmark): fresh process per (model, threads),
thread variables set before the child starts, process pinned to physical P-cores, same random
test-set sample, latency p50/p95 one article per call (500 docs), throughput batch 64 (3,000 docs),
1 and 4 cores, RAM = RSS growth after load. Run three times (--run 1, 2, 3); results are appended to
results/review_experiments/e1/cpu_benchmark.jsonl (the paper's results/final file is not touched).

Usage (from the project root):
    env/Scripts/python.exe scripts/review_experiments/e1_benchmark.py --run 1
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "review_experiments" / "e1" / "cpu_benchmark.jsonl"
MODELS = {
    "statickd": f"static:{ROOT / 'models' / 'final' / 'statickd'}",
    "sdstyle_small": f"sdstyle:{ROOT / 'models' / 'review_e1' / 'sdstyle_small_s0'}",
    "sdstyle_large": f"sdstyle:{ROOT / 'models' / 'review_e1' / 'sdstyle_large_s0'}",
}


def thread_env(threads: int) -> dict:
    """As scripts/run_benchmarks.py."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "TOKENIZERS_PARALLELISM": "true" if threads > 1 else "false"}
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        env[var] = str(threads)
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--threads", nargs="*", type=int, default=[1, 4])
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for name, spec in MODELS.items():
        if args.only and name not in args.only:
            continue
        for threads in args.threads:
            print(f"== {name} ({threads} thread), run {args.run}", flush=True)
            subprocess.run([sys.executable, str(ROOT / "scripts" / "review_experiments" / "e1_bench_child.py"),
                            "--model", spec, "--name", name, "--threads", str(threads), "--run", str(args.run),
                            "--out", str(OUT)], cwd=ROOT, env=thread_env(threads), check=True)


if __name__ == "__main__":
    main()
