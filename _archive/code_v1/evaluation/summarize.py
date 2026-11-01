"""Join accuracy results and CPU benchmarks into one comparison table (results/summary.md/.csv)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR  # noqa: E402

import pandas as pd  # noqa: E402


def accuracy_table() -> pd.DataFrame:
    rows = []
    for f in sorted((RESULTS_DIR / "accuracy").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf8"))
        row = {"model": r["model"]}
        for split in ("dev", "llm_test", "ccnews"):
            if split in r:
                row[f"{split}_macro_f1"] = r[split]["macro_f1"]
                row[f"{split}_micro_f1"] = r[split]["micro_f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def benchmark_table() -> pd.DataFrame:
    path = RESULTS_DIR / "cpu_benchmark.jsonl"
    if not path.exists():
        return pd.DataFrame(columns=["model"])
    df = pd.read_json(path, lines=True).drop_duplicates(["model", "threads"], keep="last")
    wide = df.pivot(index="model", columns="threads",
                    values=["latency_ms_p50", "docs_per_sec", "rss_peak_mb"])
    wide.columns = [f"{metric}_{threads}t" for metric, threads in wide.columns]
    size = df.groupby("model")["size_mb"].first()
    return wide.join(size).reset_index()


def main() -> None:
    table = accuracy_table().merge(benchmark_table(), on="model", how="outer")
    table = table.sort_values("dev_macro_f1", ascending=False)
    table.to_csv(RESULTS_DIR / "summary.csv", index=False)
    (RESULTS_DIR / "summary.md").write_text(table.round(3).to_markdown(index=False), encoding="utf8")
    print(table.round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
