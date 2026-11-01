"""Run the CPU benchmark for a list of models at 1 and 4 threads, each in a fresh process.

Run this on an otherwise idle machine. The heavy teacher runs get smaller sample sizes because
a single XLM-R-large prediction on one CPU thread takes seconds.

Usage:
    python evaluation/run_cpu_benchmarks.py [--threads=1,4] name=spec [name=spec ...]
"""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("benchmark_cpu.py")
THREADS = (1, 4)


def run(name: str, spec: str, thread_counts=THREADS) -> None:
    heavy = spec.startswith("teacher") or spec.startswith("onnx")
    for threads in thread_counts:
        cmd = [sys.executable, str(SCRIPT), "--model", spec, "--name", name, "--threads", str(threads),
               "--n-latency", "20" if heavy else "300",
               "--n-throughput", "40" if heavy else "2000",
               "--batch-size", "8" if heavy else "64"]
        print(" ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    thread_counts = THREADS
    for arg in sys.argv[1:]:
        if arg.startswith("--threads="):
            thread_counts = tuple(int(t) for t in arg.split("=", 1)[1].split(","))
            continue
        name, spec = arg.split("=", 1)
        run(name, spec, thread_counts)
