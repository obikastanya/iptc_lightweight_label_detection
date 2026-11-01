"""Accuracy vs coverage of a static student when only confident predictions are kept.

Also simulates a cascade: documents below the confidence threshold are sent to the teacher
(using the teacher predictions stored in results/accuracy/<teacher>.json), which shows how
much teacher compute is needed to close the accuracy gap.

Usage:
    python evaluation/confidence_analysis.py --model static:models/static/x --name x
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABEL2ID, PROCESSED_DIR, RESULTS_DIR  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from data.emmediatopic import load_emmediatopic  # noqa: E402
from evaluation.predictors import load_predictor  # noqa: E402

THRESHOLDS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def analyse(predictor, df: pd.DataFrame, teacher_pred: np.ndarray) -> list[dict]:
    texts = df["text"].tolist()
    probs = softmax(np.concatenate([predictor.logits(texts[i:i + 64]) for i in range(0, len(texts), 64)]))
    pred, conf, y = probs.argmax(1), probs.max(1), df["label_id"].to_numpy()
    rows = []
    for t in THRESHOLDS:
        keep = conf >= t
        cascade = np.where(keep, pred, teacher_pred)
        rows.append({
            "threshold": t,
            "coverage": float(keep.mean()),
            "accuracy_on_kept": float((pred[keep] == y[keep]).mean()) if keep.any() else None,
            "cascade_macro_f1": float(f1_score(y, cascade, average="macro")),
            "cascade_micro_f1": float(f1_score(y, cascade, average="micro")),
            "teacher_share": float(1 - keep.mean()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--teacher", default="teacher_xlmr_large")
    args = parser.parse_args()
    predictor = load_predictor(args.model, threads=4)
    teacher = json.loads((RESULTS_DIR / "accuracy" / f"{args.teacher}.json").read_text(encoding="utf8"))

    out = {}
    sets = {"dev": load_emmediatopic("dev"), "llm_test": pd.read_json(PROCESSED_DIR / "llm_test.jsonl", lines=True)}
    for name, df in sets.items():
        teacher_pred = np.array([LABEL2ID[p] for p in teacher[name]["predictions"]])
        out[name] = analyse(predictor, df, teacher_pred)
        print(f"== {name}\n{pd.DataFrame(out[name]).round(3).to_string(index=False)}")
    (RESULTS_DIR / f"confidence_{args.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
