"""CPU-only efficiency benchmark (latency, throughput, memory), one model per fresh process.

Protocol (simulates a small server without GPU):
  * the GPU is hidden (CUDA_VISIBLE_DEVICES=""), thread count fixed for every library;
  * the process is pinned to physical performance cores (hybrid Intel CPU: without pinning Windows
    migrates threads to the ~2.5x slower E-cores and multi-threaded numbers become unstable);
  * latency = one article per call (streaming); throughput = batches of 64 (8 for transformers);
  * RAM = resident-set growth caused by loading the model;
  * sample = a fixed random sample of the unified test set (47 languages, median ~310 words).

Usage (from the project root):
    python -m statickd.benchmark --model static:models/final/statickd --name statickd --threads 1
"""
import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# One logical CPU per physical P-core of the benchmark machine (Intel Core Ultra 7 155H: logical CPUs
# 0-1 and 10-19 are the 6 hyper-threaded P-cores, 2-9 E-cores, 20-21 low-power E-cores).
DEFAULT_PCORES = "0,10,12,14,16,18"
HEAVY_PREFIXES = ("teacher", "onnx")


def _set_threads(threads: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "RAYON_NUM_THREADS"):  # RAYON = the HF tokenizers thread pool
        os.environ[var] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if threads > 1 else "false"


def _pin(threads: int) -> list[int]:
    import psutil

    cpus = [int(c) for c in os.environ.get("BENCH_PCORES", DEFAULT_PCORES).split(",")][:threads]
    psutil.Process().cpu_affinity(cpus)
    return cpus


def benchmark_texts(n: int, seed: int = 0) -> list[str]:
    from statickd.data import load_unified_test

    texts = load_unified_test().sample(frac=1.0, random_state=seed)["text"].tolist()
    return (texts * (1 + n // len(texts)))[:n]


def run(spec: str, name: str, threads: int, n_latency: int, n_throughput: int, batch_size: int) -> dict:
    import gc
    import time

    import numpy as np
    import psutil

    from statickd.baselines import load_predictor

    proc = psutil.Process(os.getpid())
    rss = lambda: proc.memory_info().rss / 2**20  # noqa: E731
    texts = benchmark_texts(max(n_latency, n_throughput))
    rss_before, t0 = rss(), time.perf_counter()
    model = load_predictor(spec, threads)
    load_seconds, rss_loaded = time.perf_counter() - t0, rss()
    predict = getattr(model, "predict", None) or (lambda batch: model.logits(batch).argmax(1))

    for t in texts[:3]:  # warm-up
        predict([t])
    times = []
    for t in texts[:n_latency]:
        start = time.perf_counter()
        predict([t])
        times.append((time.perf_counter() - start) * 1000)
    predict(texts[:batch_size])
    gc.collect()
    start = time.perf_counter()
    for i in range(0, n_throughput, batch_size):
        predict(texts[i:i + batch_size])
    docs_per_sec = n_throughput / (time.perf_counter() - start)
    times = np.array(times)
    return {"model": name, "spec": spec, "threads": threads, "size_mb": round(model.size_mb, 2),
            "num_parameters": model.num_parameters, "load_seconds": round(load_seconds, 2),
            "rss_model_mb": round(rss_loaded - rss_before, 1), "rss_peak_mb": round(rss(), 1),
            "latency_ms_mean": round(float(times.mean()), 3), "latency_ms_p50": round(float(np.percentile(times, 50)), 3),
            "latency_ms_p95": round(float(np.percentile(times, 95)), 3), "docs_per_sec": round(docs_per_sec, 3),
            "n_latency": n_latency, "n_throughput": n_throughput, "batch_size": batch_size}


def profile_static(path: str, name: str, n: int = 1280, batch_size: int = 64) -> dict:
    """Split the cost of a StaticKD prediction into tokenisation and table pooling (single thread)."""
    import time

    import numpy as np

    from statickd.student import StaticKDModel, encode

    model = StaticKDModel(path)
    texts = benchmark_texts(n)
    batches = [texts[i:i + batch_size] for i in range(0, n, batch_size)]
    encode(model.tokenizer, batches[0], model.max_tokens)  # warm-up
    t0 = time.perf_counter()
    ids = [encode(model.tokenizer, b, model.max_tokens) for b in batches]
    t_tok = time.perf_counter() - t0
    t0 = time.perf_counter()
    for b in ids:
        np.stack([model.table[i].mean(axis=0, dtype=np.float32) for i in b]) + model.bias
    t_pool = time.perf_counter() - t0
    tokens = [len(i) for b in ids for i in b]
    return {"model": name, "profile": True, "threads": 1, "n_docs": n, "mean_tokens": float(np.mean(tokens)),
            "tokenize_docs_per_sec": n / t_tok, "pool_docs_per_sec": n / t_pool,
            "tokenize_share": t_tok / (t_tok + t_pool)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--profile", action="store_true", help="StaticKD only: tokenisation vs pooling cost")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-latency", type=int, default=None)
    parser.add_argument("--n-throughput", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    heavy = args.model.startswith(HEAVY_PREFIXES)
    _set_threads(args.threads)
    pinned = _pin(args.threads)

    import json
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from statickd.config import FINAL_DIR

    if args.profile:
        result = profile_static(args.model.partition(":")[2], args.name)
    else:
        result = run(args.model, args.name, args.threads,
                     args.n_latency or (30 if heavy else 500), args.n_throughput or (48 if heavy else 3000),
                     args.batch_size or (8 if heavy else 64))
    result["pinned_cpus"] = pinned
    print(json.dumps(result), flush=True)
    out = Path(args.out) if args.out else FINAL_DIR / ("cpu_profile.jsonl" if args.profile else "cpu_benchmark.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf8") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
