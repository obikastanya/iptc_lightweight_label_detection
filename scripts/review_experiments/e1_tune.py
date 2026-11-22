"""E1 tuning: small grid for the Sparse-Distillation-style baseline, development set only.

Grid (fixed before running): lr_emb {1e-3, 1e-2} x hidden {256, 1024} x max epochs {6, 10}.
Every (lr_emb, hidden) pair is trained once for 10 epochs with seed 0. The configuration "max epochs
= k" keeps the epoch with the highest internal validation agreement among epochs 1..k (the StaticKD
selection rule), and its dev macro-F1 (GPU fp32, recorded during training) is the tuning score. A
10-epoch run with a constant learning rate contains the 6-epoch run as its prefix, so one run scores
both epoch settings. The test set is never loaded.

Usage (from the project root):
    env/Scripts/python.exe scripts/review_experiments/e1_tune.py --size small --lr 1e-3 1e-2 --hidden 256 1024
    env/Scripts/python.exe scripts/review_experiments/e1_tune.py --select
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from statickd.evaluation import eval_set  # noqa: E402
from statickd.sdstyle import SDConfig, train  # noqa: E402
from statickd.student import encode, load_base_embeddings  # noqa: E402

OUT_DIR = ROOT / "results" / "review_experiments" / "e1"
RUNS = OUT_DIR / "tuning_runs.jsonl"
SIZES = {"small": {"dim": 128, "buckets": 524_288}, "large": {"dim": 128, "buckets": 1_600_000}}
GRID = {"lr_emb": [1e-3, 1e-2], "hidden": [256, 1024], "max_epochs": [6, 10]}


def done() -> list[dict]:
    if not RUNS.exists():
        return []
    return [json.loads(line) for line in RUNS.read_text(encoding="utf8").splitlines() if line.strip()]


def run(size: str, lr: float, hidden: int) -> None:
    if any(r["size"] == size and r["lr_emb"] == lr and r["hidden"] == hidden for r in done()):
        print(f"{size} lr={lr} hidden={hidden}: done, skipped", flush=True)
        return
    tokenizer, _ = load_base_embeddings()
    dev = eval_set("dev")
    monitor = {"ids": encode(tokenizer, dev["text"].tolist()), "y": dev["y"].to_numpy()}
    cfg = SDConfig(name=f"tune_{size}_lr{lr:g}_h{hidden}", seed=0, hidden=hidden, lr_emb=lr, epochs=max(GRID["max_epochs"]),
                   **SIZES[size])
    meta = train(cfg, monitor=monitor, save=False)
    row = {"size": size, "lr_emb": lr, "hidden": hidden, **SIZES[size], "seed": 0,
           "embedding_parameters": meta["embedding_parameters"], "head_parameters": meta["head_parameters"],
           "peak_gpu_mb": meta["peak_gpu_mb"], "train_seconds": meta["train_seconds"], "history": meta["history"]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS, "a", encoding="utf8") as f:
        f.write(json.dumps(row) + "\n")


def configs(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        for k in GRID["max_epochs"]:
            h = r["history"][:k]
            e = int(np.argmax([x["val_agreement"] for x in h]))   # first maximum, as `agreement > best`
            out.append({"size": r["size"], "lr_emb": r["lr_emb"], "hidden": r["hidden"], "max_epochs": k,
                        "selected_epoch": e + 1, "val_agreement": h[e]["val_agreement"],
                        "dev_macro_f1": round(h[e]["monitor_macro_f1"], 5), "dev_accuracy": round(h[e]["monitor_accuracy"], 5)})
    return out


def select() -> dict:
    rows = done()
    table = configs(rows)
    result = {"protocol": __doc__.split("Usage")[0].strip(), "grid": GRID, "sizes": SIZES,
              "fixed": {k: v for k, v in SDConfig(name="x").__dict__.items() if k not in ("name", "lr_emb", "hidden", "epochs", "dim", "buckets")},
              "runs": [{k: v for k, v in r.items() if k != "history"} | {"history": r["history"]} for r in rows],
              "configs": table, "selected": {}}
    for size in SIZES:
        cand = [c for c in table if c["size"] == size]
        if cand:
            best = max(cand, key=lambda c: c["dev_macro_f1"])
            result["selected"][size] = best
    (OUT_DIR / "tuning.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf8")
    for c in table:
        print(c)
    print("selected:", json.dumps(result["selected"], indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(SIZES))
    parser.add_argument("--lr", nargs="*", type=float, default=GRID["lr_emb"])
    parser.add_argument("--hidden", nargs="*", type=int, default=GRID["hidden"])
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    if args.select:
        select()
        return
    for lr in args.lr:
        for hidden in args.hidden:
            run(args.size, lr, hidden)


if __name__ == "__main__":
    main()
