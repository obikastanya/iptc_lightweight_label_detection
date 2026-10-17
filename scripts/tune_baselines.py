"""Hyper-parameter grid for the non-transformer KD baselines, selected on the development set only.

The grid was fixed before running (research_final/paper_quality_check_filled.md, decision D3):
  * fastText + KD (word and XLM-R subword tokens): lr {0.1, 0.5, 1.0} x dim {100, 300} x epoch {5, 15, 25}
  * TF-IDF + SVM + KD: C {0.1, 0.25, 0.5, 1, 2}
Selection criterion: macro-F1 on the EMMediaTopic dev split, the same criterion used to fix the
StaticKD recipe. The test set is never used. All configurations are trained on the same distillation
pool with the teacher arg-max labels; models are kept in memory and only their dev scores are stored.

The run can be stopped and restarted: configurations already in baseline_grid.jsonl are not trained again.
fastText + word tokens is slow (character n-grams): about 7 min for 5 epochs and 22 min for 15 epochs at
dim 100 with 8 threads, so the whole grid takes many hours.

Outputs (appended / overwritten):
  results/final/baseline_grid.jsonl      one row per configuration
  results/final/baseline_grid_best.json  chosen configuration per baseline (read by train_baselines.py)

Usage (from the project root):
    python scripts/tune_baselines.py [--only fasttext tfidf] [--threads 8]
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from statickd.baselines import (fasttext_predict, fasttext_train_file, fit_fasttext, subword_tokens,  # noqa: E402
                                tfidf_vectorizer, tokenize)
from statickd.config import FINAL_DIR  # noqa: E402
from statickd.data import load_distillation_pool, teacher_probs  # noqa: E402
from statickd.evaluation import eval_set, scores  # noqa: E402

FT_GRID = {"lr": [0.1, 0.5, 1.0], "dim": [100, 300], "epoch": [5, 15, 25]}
TFIDF_GRID = {"c": [0.1, 0.25, 0.5, 1.0, 2.0]}
GRID_FILE = FINAL_DIR / "baseline_grid.jsonl"
BEST_FILE = FINAL_DIR / "baseline_grid_best.json"


def log(row: dict) -> None:
    print(json.dumps(row), flush=True)
    with open(GRID_FILE, "a", encoding="utf8") as f:
        f.write(json.dumps(row) + "\n")


def done_rows() -> list[dict]:
    if not GRID_FILE.exists():
        return []
    return [json.loads(line) for line in GRID_FILE.read_text(encoding="utf8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=["fasttext", "tfidf"])
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    pool = load_distillation_pool()
    y_train = teacher_probs(pool).argmax(1)
    dev = eval_set("dev")
    y_dev = dev["y"].to_numpy()
    best = json.loads(BEST_FILE.read_text()) if BEST_FILE.exists() else {}
    finished = done_rows()

    if "fasttext" in args.only:
        for tokens in ("word", "subword"):
            train_path = fasttext_train_file(tokenize(pool["text"].tolist(), tokens), y_train)
            dev_docs = tokenize(dev["text"].tolist(), tokens)
            rows = [r for r in finished if r["baseline"] == f"ft_kd_{tokens}"]
            for lr, dim, epoch in itertools.product(FT_GRID["lr"], FT_GRID["dim"], FT_GRID["epoch"]):
                if any((r["lr"], r["dim"], r["epoch"]) == (lr, dim, epoch) for r in rows):
                    continue   # resumed run: already trained
                t0 = time.perf_counter()
                model = fit_fasttext(train_path, tokens, dim=dim, epoch=epoch, lr=lr, threads=args.threads)
                s = scores(y_dev, fasttext_predict(model, dev_docs))
                row = {"baseline": f"ft_kd_{tokens}", "tokens": tokens, "lr": lr, "dim": dim, "epoch": epoch,
                       "dev_macro_f1": round(s["macro_f1"], 5), "dev_accuracy": round(s["accuracy"], 5),
                       "train_seconds": round(time.perf_counter() - t0, 1)}
                log(row)
                rows.append(row)
                del model
            os.unlink(train_path)
            top = max(rows, key=lambda r: r["dev_macro_f1"])
            best[f"ft_kd_{tokens}"] = {k: top[k] for k in ("lr", "dim", "epoch", "dev_macro_f1")}
            BEST_FILE.write_text(json.dumps(best, indent=1))

    if "tfidf" in args.only:
        from sklearn.svm import LinearSVC

        vec = tfidf_vectorizer()
        x_train = vec.fit_transform(subword_tokens(pool["text"].tolist()))
        x_dev = vec.transform(subword_tokens(dev["text"].tolist()))
        rows = [r for r in finished if r["baseline"] == "tfidf_svm_kd"]
        for c in TFIDF_GRID["c"]:
            if any(r["c"] == c for r in rows):
                continue   # resumed run: already trained
            t0 = time.perf_counter()
            clf = LinearSVC(C=c, random_state=0).fit(x_train, y_train)
            s = scores(y_dev, clf.classes_[np.asarray(clf.decision_function(x_dev)).argmax(1)])
            row = {"baseline": "tfidf_svm_kd", "c": c, "dev_macro_f1": round(s["macro_f1"], 5),
                   "dev_accuracy": round(s["accuracy"], 5), "train_seconds": round(time.perf_counter() - t0, 1)}
            log(row)
            rows.append(row)
        top = max(rows, key=lambda r: r["dev_macro_f1"])
        best["tfidf_svm_kd"] = {"c": top["c"], "dev_macro_f1": top["dev_macro_f1"]}
        BEST_FILE.write_text(json.dumps(best, indent=1))
    print("best:", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
