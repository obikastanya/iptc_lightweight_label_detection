"""Review round 5, part C: CPU benchmarks with the exact protocol of statickd.benchmark (scripts/run_benchmarks.py).

One fresh process per (model, threads), pinned P-cores, thread limits set in the environment before the child starts,
1 and 4 threads, three repetitions (--run 1, 2, 3) in this session. StaticKD, the fp32 Model2Vec classifier and the
fp32 ONNX transformers are rerun as same-session references. Results are appended to
results/review_experiments/r5_cpu_benchmark.jsonl (results/final/cpu_benchmark.jsonl is not touched).

Run from the project root on an otherwise idle machine:
    env/Scripts/python.exe scripts/review_experiments/r5_c_benchmark.py --runs 1 2 3
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from run_benchmarks import thread_env  # noqa: E402

B, F, R = ROOT / "models" / "baselines", ROOT / "models" / "final", ROOT / "models" / "review_r5"
OUT = ROOT / "results" / "review_experiments" / "r5_cpu_benchmark.jsonl"
MODELS = {
    "statickd": f"static:{F / 'statickd'}",
    "model2vec_clf": f"model2vec:{B / 'model2vec_clf'}",
    "model2vec_clf_fp16": f"model2vec:{R / 'model2vec_clf_fp16'}",
    "model2vec_clf_int8": f"model2vec:{R / 'model2vec_clf_int8'}",
    "m2v_lin": f"model2vec:{R / 'm2v_lin'}",
    "m2v_lin_nonorm_folded": f"static:{R / 'm2v_lin_nonorm_folded'}",
    "minilm_kd": f"onnx:{B / 'minilm_kd_t1'}",
    "minilm_kd_int8": f"onnx:{R / 'minilm_kd_int8'}",
    "minilm_kd_int8_gather_only": f"onnx:{R / 'minilm_kd_int8_gather_only_pc_rr'}",   # diagnostic variant (r5_c_minilm_diag.py)
    "distilmbert_kd": f"onnx:{B / 'distilmbert_kd'}",
    "distilmbert_kd_int8": f"onnx:{R / 'distilmbert_kd_int8'}",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--threads", nargs="*", type=int, default=[1, 4])
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    for run in args.runs:
        for name, spec in MODELS.items():
            if args.only and name not in args.only:
                continue
            path = Path(spec.partition(":")[2])
            if not path.exists():
                print(f"!! skip {name}: {path} missing", flush=True)
                continue
            for threads in args.threads:
                t0 = time.time()
                print(f"== run {run} {name} ({threads} thread)", flush=True)
                subprocess.run([sys.executable, "-m", "statickd.benchmark", "--model", spec, "--name", name,
                                "--threads", str(threads), "--run", str(run), "--out", str(OUT)],
                               cwd=ROOT, env=thread_env(threads), check=True)
                print(f"   {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
