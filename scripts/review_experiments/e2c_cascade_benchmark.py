"""E2 step 3: end-to-end CPU throughput of the StaticKD -> ONNX int8 teacher cascade.

The cascade runner takes a batch of articles, runs StaticKD, routes every article whose top softmax
probability is < theta to the ONNX int8 teacher (in sub-batches of 8, the transformer batch size of the
benchmark protocol), and returns the final labels. The timed loop therefore contains StaticKD,
softmax/routing and all teacher calls.

Protocol (as statickd.benchmark / scripts/run_benchmarks.py): GPU hidden, thread limits set in the
environment of a fresh child process, process pinned to one physical P-core (logical CPU 0), same text
sample (unified test set shuffled with random_state=0, first n articles), batches of 64, warm-up first.

One session (--session k) runs three fresh processes, one after the other:
  1. StaticKD component benchmark  (statickd.benchmark, default sizes: 3000 docs, batch 64)
  2. ONNX int8 teacher component   (statickd.benchmark, default sizes: 96 docs, batch 8)
  3. the cascade                   (this file with --child, n articles, batch 64, teacher sub-batch 8)
Results are appended to results/review_experiments/cascade_benchmark.jsonl and
cascade_components.jsonl.

Usage (from the project root):
    env/Scripts/python.exe scripts/review_experiments/e2c_cascade_benchmark.py --session 1
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "review_experiments"
STATIC_SPEC = f"static:{ROOT / 'models' / 'final' / 'statickd'}"


def thread_env(threads: int) -> dict:
    """Same as scripts/run_benchmarks.thread_env: thread limits must exist before NumPy is imported."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "TOKENIZERS_PARALLELISM": "true" if threads > 1 else "false",
           "CUDA_VISIBLE_DEVICES": ""}
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        env[var] = str(threads)
    return env


def child(theta: float, n: int, batch_size: int, teacher_batch: int, session: int) -> dict:
    import gc
    import time

    import numpy as np
    import psutil

    from statickd.baselines import load_predictor
    from statickd.benchmark import _pin, _set_threads, benchmark_texts
    from statickd.data import load_unified_test, softmax

    _set_threads(1)
    pinned = _pin(1)
    proc = psutil.Process(os.getpid())
    rss = lambda: proc.memory_info().rss / 2**20  # noqa: E731
    texts = benchmark_texts(n)
    rss0, t0 = rss(), time.perf_counter()
    student = load_predictor(STATIC_SPEC, 1)
    teacher = load_predictor("teacher-onnx-int8", 1)
    load_seconds, rss_loaded = time.perf_counter() - t0, rss()

    timers = {"statickd": 0.0, "teacher": 0.0}

    def cascade(batch: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Final labels of a batch and the routing mask (True = sent to the teacher)."""
        s0 = time.perf_counter()
        probs = softmax(student.logits(batch))
        labels = probs.argmax(1)
        routed = np.flatnonzero(probs.max(1) < theta)
        s1 = time.perf_counter()
        for j in range(0, len(routed), teacher_batch):
            idx = routed[j:j + teacher_batch]
            labels[idx] = teacher.logits([batch[i] for i in idx]).argmax(1)
        timers["statickd"] += s1 - s0
        timers["teacher"] += time.perf_counter() - s1
        mask = np.zeros(len(batch), dtype=bool)
        mask[routed] = True
        return labels, mask

    for t in texts[:3]:  # warm-up of both models (as statickd.benchmark.run)
        student.logits([t]); teacher.logits([t])
    student.logits(texts[:batch_size]); teacher.logits(texts[:teacher_batch])
    timers.update(statickd=0.0, teacher=0.0)
    gc.collect()
    labels, masks = [], []
    start = time.perf_counter()
    for i in range(0, n, batch_size):
        lab, m = cascade(texts[i:i + batch_size])
        labels.append(lab); masks.append(m)
    elapsed = time.perf_counter() - start
    labels, routed = np.concatenate(labels), np.concatenate(masks)

    # outside the timing: agreement with the cascade rebuilt from the stored test predictions
    idx = load_unified_test().sample(frac=1.0, random_state=0).index.to_numpy()[:n]
    from statickd.evaluation import load_logits
    ps = softmax(load_logits("statickd", "test")[idx])
    pt = load_logits("teacher_onnx_int8", "test")[idx].argmax(1)
    stored = np.where(ps.max(1) >= theta, ps.argmax(1), pt)
    n_routed = int(routed.sum())
    return {"model": "cascade", "theta": theta, "session": session, "threads": 1, "pinned_cpus": pinned,
            "n_docs": n, "batch_size": batch_size, "teacher_batch": teacher_batch,
            "n_routed": n_routed, "routed_share": n_routed / n, "seconds": elapsed, "docs_per_sec": n / elapsed,
            "statickd_seconds": timers["statickd"], "teacher_seconds": timers["teacher"],
            "statickd_docs_per_sec_in_cascade": n / timers["statickd"],
            "teacher_docs_per_sec_on_routed": n_routed / timers["teacher"] if n_routed else None,
            "load_seconds": load_seconds, "rss_models_mb": rss_loaded - rss0,
            "routing_same_as_stored": float((routed == (ps.max(1) < theta)).mean()),
            "labels_same_as_stored_cascade": float((labels == stored).mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--n", type=int, default=448, help="cascade sample size (7 batches of 64)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--teacher-batch", type=int, default=8)
    parser.add_argument("--child", action="store_true", help="internal: run the cascade in this process")
    parser.add_argument("--skip-components", action="store_true")
    args = parser.parse_args()
    theta = json.loads((RES / "cascade_threshold.json").read_text(encoding="utf8"))["chosen_theta"]

    if args.child:
        res = child(theta, args.n, args.batch_size, args.teacher_batch, args.session)
        print(json.dumps(res), flush=True)
        with open(RES / "cascade_benchmark.jsonl", "a", encoding="utf8") as f:
            f.write(json.dumps(res) + "\n")
        return

    env = thread_env(1)
    if not args.skip_components:
        for name, spec in (("statickd", STATIC_SPEC), ("teacher_onnx_int8", "teacher-onnx-int8")):
            print(f"== session {args.session}: {name} component (1 thread)", flush=True)
            subprocess.run([sys.executable, "-m", "statickd.benchmark", "--model", spec, "--name", name,
                            "--threads", "1", "--run", str(args.session),
                            "--out", str(RES / "cascade_components.jsonl")], cwd=ROOT, env=env, check=True)
    print(f"== session {args.session}: cascade theta={theta} (1 thread)", flush=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", "--session", str(args.session),
                    "--n", str(args.n), "--batch-size", str(args.batch_size),
                    "--teacher-batch", str(args.teacher_batch)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
