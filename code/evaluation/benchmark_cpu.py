"""CPU-only efficiency benchmark of one model, run in a fresh process for clean memory numbers.

Simulates a small server: the thread count is fixed (default 1 and 4 are reported) and the
GPU is hidden. Appends one JSON line per run to results/cpu_benchmark.jsonl.

Usage:
    python evaluation/benchmark_cpu.py --model teacher-onnx-int8 --threads 1
"""
import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, whatever the machine has


def _set_thread_env(threads: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "RAYON_NUM_THREADS"):  # RAYON: the HF tokenizers thread pool
        os.environ[var] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if threads > 1 else "false"


# One logical CPU per physical P-core of the benchmark machine (Intel Core Ultra 7 155H):
# 6 hyper-threaded P-cores = logical CPUs 0-1 and 10-19, 8 E-cores = 2-9 (~2.5x slower on a
# single-threaded matmul probe), 2 low-power E-cores = 20-21 (~3.7x slower). Measured once with
# a pinned NumPy matmul; override with BENCH_PCORES="a,b,c" on other machines.
DEFAULT_PCORES = "0,10,12,14,16,18"


def _performance_cores() -> list[int]:
    return [int(c) for c in os.environ.get("BENCH_PCORES", DEFAULT_PCORES).split(",")]


def _pin_to_performance_cores(threads: int) -> list[int]:
    """Pin the process to `threads` distinct P-cores: emulates a small server with uniform
    cores. Without pinning, Windows migrates threads onto slow E-cores and multi-threaded
    results vary by an order of magnitude."""
    import psutil

    cpus = _performance_cores()[:threads]
    psutil.Process().cpu_affinity(cpus)
    return cpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="predictor spec, see evaluation/predictors.py")
    parser.add_argument("--name", default=None, help="display name in the results table")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-latency", type=int, default=100)
    parser.add_argument("--n-throughput", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    _set_thread_env(args.threads)
    pinned = _pin_to_performance_cores(args.threads)

    import json
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import RESULTS_DIR
    from data.emmediatopic import load_emmediatopic
    from evaluation.predictors import load_predictor
    from evaluation.speed import measure_latency, measure_throughput, rss_mb

    # Fixed benchmark sample: dev documents (4 languages, median ~220 words), same for all models.
    texts = load_emmediatopic("dev").sample(frac=1.0, random_state=0)["text"].tolist()
    texts = (texts * 10)[:max(args.n_latency, args.n_throughput)]

    rss_before = rss_mb()
    t0 = time.perf_counter()
    predictor = load_predictor(args.model, args.threads)
    load_seconds = time.perf_counter() - t0
    rss_loaded = rss_mb()

    latency = measure_latency(predictor.predict, texts[:args.n_latency])
    throughput = measure_throughput(predictor.predict, texts[:args.n_throughput], args.batch_size)
    rss_peak = rss_mb()

    result = {
        "model": args.name or args.model,
        "spec": args.model,
        "threads": args.threads,
        "pinned_cpus": pinned,
        "size_mb": round(predictor.size_mb, 2),
        "num_parameters": predictor.num_parameters,
        "load_seconds": round(load_seconds, 2),
        "rss_baseline_mb": round(rss_before, 1),
        "rss_model_mb": round(rss_loaded - rss_before, 1),
        "rss_peak_mb": round(rss_peak, 1),
        **{k: round(v, 3) if isinstance(v, float) else v for k, v in latency.items() if k != "n_docs"},
        "docs_per_sec": round(throughput["docs_per_sec"], 2),
        "batch_size": args.batch_size,
    }
    print(json.dumps(result))
    with open(RESULTS_DIR / "cpu_benchmark.jsonl", "a", encoding="utf8") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
