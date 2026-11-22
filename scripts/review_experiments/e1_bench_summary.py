"""E1: median over the benchmark sessions (as part5.py) and the session spread per model.

Spread = largest relative deviation of a session from the median (1-core throughput and 1-core p50
latency); the paper accepts <= 7%.

Usage (from the project root):
    env/Scripts/python.exe scripts/review_experiments/e1_bench_summary.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
E1 = ROOT / "results" / "review_experiments" / "e1"
METRICS = ["latency_ms_p50", "latency_ms_p95", "docs_per_sec", "rss_model_mb", "size_mb", "load_seconds"]
DEPS = {"statickd": "numpy, tokenizers", "sdstyle_small": "numpy, tokenizers", "sdstyle_large": "numpy, tokenizers"}


def main() -> None:
    runs = pd.read_json(E1 / "cpu_benchmark.jsonl", lines=True).drop_duplicates(["model", "threads", "run"], keep="last")
    med = runs.groupby(["model", "threads"])[METRICS].median().reset_index()
    out = {}
    for m, g in med.groupby("model"):
        r1, r4 = g[g.threads == 1].iloc[0], g[g.threads == 4].iloc[0]
        s1 = runs[(runs.model == m) & (runs.threads == 1)]
        s4 = runs[(runs.model == m) & (runs.threads == 4)]
        dev = lambda s, col, medv: float((s[col] - medv).abs().max() / medv)  # noqa: E731
        out[m] = {"latency_p50_1c_ms": r1.latency_ms_p50, "latency_p95_1c_ms": r1.latency_ms_p95,
                  "docs_per_sec_1c": r1.docs_per_sec, "docs_per_sec_4c": r4.docs_per_sec,
                  "rss_model_mb": r1.rss_model_mb, "size_mb": r1.size_mb, "load_seconds": r1.load_seconds,
                  "num_parameters": int(runs[runs.model == m].num_parameters.iloc[0]),
                  "n_sessions": int(s1.run.nunique()), "deps": DEPS.get(m),
                  "maxdev_dps_1c": dev(s1, "docs_per_sec", r1.docs_per_sec),
                  "maxdev_dps_4c": dev(s4, "docs_per_sec", r4.docs_per_sec),
                  "maxdev_lat_p50_1c": dev(s1, "latency_ms_p50", r1.latency_ms_p50),
                  "per_session_dps_1c": s1.sort_values("run").docs_per_sec.tolist(),
                  "per_session_dps_4c": s4.sort_values("run").docs_per_sec.tolist(),
                  "per_session_lat_p50_1c": s1.sort_values("run").latency_ms_p50.tolist()}
    if "statickd" in out:
        for m in out:
            out[m]["speed_ratio_vs_statickd_1c"] = out[m]["docs_per_sec_1c"] / out["statickd"]["docs_per_sec_1c"]
    paper = pd.read_json(ROOT / "results" / "final" / "cpu_benchmark.jsonl", lines=True)
    paper = paper[paper.model == "statickd"].drop_duplicates(["threads", "run"], keep="last").groupby("threads")[METRICS].median()
    result = {"sessions": out, "paper_statickd_median": json.loads(paper.to_json(orient="index"))}
    (E1 / "cpu_benchmark_summary.json").write_text(json.dumps(result, indent=1), encoding="utf8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
