"""One benchmark process (statickd.benchmark protocol) that also understands the spec `sdstyle:<dir>`.

statickd.benchmark.run() builds its predictor with statickd.baselines.load_predictor; this wrapper adds
the Sparse-Distillation-style predictor to that function for this process only and then runs the
unchanged statickd.benchmark.main() (same pinning, thread limits, sample, warm-up and timing code).

Usage: see e1_benchmark.py (always started in a fresh process with the thread variables set).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import statickd.benchmark as bench  # noqa: E402  (hides the GPU at import)
import statickd.baselines as baselines  # noqa: E402

_original = baselines.load_predictor


def load_predictor(spec: str, threads: int = 1):
    kind, _, path = spec.partition(":")
    if kind == "sdstyle":
        from statickd.sdstyle import SDStyleModel
        return SDStyleModel(path)
    return _original(spec, threads)


baselines.load_predictor = load_predictor

if __name__ == "__main__":
    bench.main()
