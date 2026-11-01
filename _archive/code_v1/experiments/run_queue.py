"""Phase 2 of the accuracy exploration: train StaticKD variants from a queue, then fold and evaluate.

Each job = extra CLI flags on top of the final recipe. Jobs run `--workers` at a time on the GPU;
after training every model is folded into a token table and evaluated on dev / llm_test / ccnews.

Usage:
    python experiments/run_queue.py --jobs tau1:"--temperature 1" gpt1:"--gpt-weight 1.0"
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
LOGS = CODE.parent / "scratch" / "explore_logs"
BASE = ["--sources", "emmediatopic", "ccnews", "ccnews_extra", "--epochs", "6",
        "--token-dropout", "0.2", "--keep-all-vocab"]


def run(job: str) -> str:
    name, _, flags = job.partition(":")
    name = f"x_{name}"
    log = LOGS / f"{name}.log"
    with open(log, "w", encoding="utf8") as f:
        cmd = [sys.executable, "-u", "students/static_student.py", "--name", name, *BASE, *flags.split()]
        if subprocess.run(cmd, cwd=CODE, stdout=f, stderr=subprocess.STDOUT).returncode:
            return f"{name}: TRAINING FAILED (see {log})"
        subprocess.run([sys.executable, "-c", f"from students.static_student import export_table; "
                        f"export_table('../models/static/{name}')"], cwd=CODE, stdout=f, stderr=subprocess.STDOUT)
        subprocess.run([sys.executable, "evaluation/evaluate_accuracy.py", "--model",
                        f"static:../models/static/{name}_table", "--name", f"{name}_table"],
                       cwd=CODE, stdout=f, stderr=subprocess.STDOUT)
    summary = [l.strip() for l in log.read_text(encoding="utf8").splitlines() if l.startswith("[")]
    line = f"{name}: " + " | ".join(summary)
    print(line, flush=True)
    return line


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(run, args.jobs))
    print("DONE\n" + "\n".join(results), flush=True)


if __name__ == "__main__":
    main()
