"""Throughput / latency / memory measurement used for every model in the comparison."""
import gc
import os
import time
from typing import Callable

import numpy as np
import psutil


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def measure_throughput(predict_fn: Callable[[list[str]], object], texts: list[str],
                       batch_size: int, warmup: int = 1) -> dict:
    """Documents per second when predicting `texts` in batches of `batch_size`."""
    for _ in range(warmup):
        predict_fn(texts[:batch_size])
    gc.collect()
    start = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        predict_fn(texts[i:i + batch_size])
    elapsed = time.perf_counter() - start
    return {"docs_per_sec": len(texts) / elapsed, "n_docs": len(texts), "batch_size": batch_size}


def measure_latency(predict_fn: Callable[[list[str]], object], texts: list[str], warmup: int = 3) -> dict:
    """Per-document latency (batch size 1), the relevant number for a streaming pipeline."""
    for t in texts[:warmup]:
        predict_fn([t])
    times = []
    for t in texts:
        start = time.perf_counter()
        predict_fn([t])
        times.append((time.perf_counter() - start) * 1000)
    times = np.array(times)
    return {
        "latency_ms_mean": float(times.mean()),
        "latency_ms_p50": float(np.percentile(times, 50)),
        "latency_ms_p95": float(np.percentile(times, 95)),
        "n_docs": len(texts),
    }


def dir_size_mb(path) -> float:
    path = str(path)
    if os.path.isfile(path):
        return os.path.getsize(path) / 2**20
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total / 2**20
