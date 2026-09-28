"""CPU benchmark of every deployable model in ONE session, each run in a fresh process (1 and 4 cores).

Run it on an otherwise idle machine. Results are appended to results/final/cpu_benchmark.jsonl; the
notebook keeps the last row per (model, threads).

Usage (from the project root):
    python scripts/run_benchmarks.py [--minilm minilm_kd_t1] [--threads 1 4]
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "models" / "baselines"
F = ROOT / "models" / "final"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minilm", default="minilm_kd_t1", help="the MiniLM student chosen on the dev set")
    parser.add_argument("--threads", nargs="*", type=int, default=[1, 4])
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    models = {
        "teacher": "teacher-pytorch",
        "teacher_onnx_int8": "teacher-onnx-int8",
        "distilmbert_kd": f"onnx:{B / 'distilmbert_kd'}",
        "minilm_kd": f"onnx:{B / args.minilm}",
        "statickd": f"static:{F / 'statickd'}",
        "statickd_compact": f"static:{F / 'statickd_compact'}",
        "ft_kd_subword_q": f"fasttext:{B / 'ft_kd_subword_q.ftz'}",
        "ft_kd_subword": f"fasttext:{B / 'ft_kd_subword.bin'}",
        "tfidf_svm_kd": f"tfidf:{B / 'tfidf_svm_kd.joblib'}",
    }
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}
    for name, spec in models.items():
        if args.only and name not in args.only:
            continue
        for threads in args.threads:
            print(f"== {name} ({threads} thread)", flush=True)
            subprocess.run([sys.executable, "-m", "statickd.benchmark", "--model", spec, "--name", name,
                            "--threads", str(threads)], cwd=ROOT, env=env, check=True)
    for name in ("statickd", "statickd_compact"):   # where does the StaticKD time go?
        if not args.only or name in args.only:
            subprocess.run([sys.executable, "-m", "statickd.benchmark", "--model", models[name], "--name", name,
                            "--threads", "1", "--profile"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
