"""Accuracy evaluation of any predictor on the project's evaluation sets.

Evaluation sets
  dev            EMMediaTopic dev split (1,000 docs, hr/sl/ca/el), GPT-4o labels. Never seen by
                 the teacher, so teacher and students are compared on equal footing.
  ccnews         CC-News held-out documents in ~70 languages from news sites unseen in training;
                 the reference is the teacher's prediction (measures distillation fidelity).
  llm_test       Multilingual CC-News sample annotated by an LLM with Kuzman's GPT-4o prompt
                 (independent of the teacher), if data/processed/llm_test.jsonl exists.

Usage:
    python evaluation/evaluate_accuracy.py --model fasttext:models/ft.bin --name fastText
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABELS, PROCESSED_DIR, RESULTS_DIR  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.emmediatopic import load_emmediatopic  # noqa: E402
from evaluation.metrics import (bootstrap_macro_f1_ci, confusion, f1_scores, per_group_f1,  # noqa: E402
                                per_label_f1)
from evaluation.predictors import load_predictor  # noqa: E402

TEACHER_CCNEWS_HELDOUT = PROCESSED_DIR / "teacher_labels" / "ccnews_heldout.parquet"
LLM_TEST = PROCESSED_DIR / "llm_test.jsonl"


def predict_all(predictor, texts: list[str], batch_size: int = 32) -> np.ndarray:
    return np.concatenate([predictor.predict(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)])


def evaluate_set(predictor, df: pd.DataFrame, label_col: str) -> dict:
    y_pred = predict_all(predictor, df["text"].tolist())
    y_true = df[label_col].to_numpy()
    lo, hi = bootstrap_macro_f1_ci(y_true, y_pred)
    return {
        **f1_scores(y_true, y_pred),
        "macro_f1_ci95": [lo, hi],
        "per_language": per_group_f1(y_true, y_pred, df["lang"].to_numpy()),
        "per_label_f1": per_label_f1(y_true, y_pred),
        "confusion": confusion(y_true, y_pred),
        "predictions": [LABELS[i] for i in y_pred],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--sets", nargs="*", default=["dev", "ccnews", "llm_test"])
    args = parser.parse_args()

    predictor = load_predictor(args.model, args.threads)
    results = {"model": args.name, "spec": args.model}

    if "dev" in args.sets:
        results["dev"] = evaluate_set(predictor, load_emmediatopic("dev"), "label_id")
        print(f"[dev] macro-F1={results['dev']['macro_f1']:.3f} micro-F1={results['dev']['micro_f1']:.3f}")
    if "ccnews" in args.sets and TEACHER_CCNEWS_HELDOUT.exists():
        df = pd.read_parquet(TEACHER_CCNEWS_HELDOUT)
        results["ccnews"] = evaluate_set(predictor, df, "teacher_label_id")
        print(f"[ccnews vs teacher] macro-F1={results['ccnews']['macro_f1']:.3f} "
              f"accuracy={results['ccnews']['accuracy']:.3f}")
    if "llm_test" in args.sets and LLM_TEST.exists():
        df = pd.read_json(LLM_TEST, lines=True)
        results["llm_test"] = evaluate_set(predictor, df, "label_id")
        print(f"[llm_test] macro-F1={results['llm_test']['macro_f1']:.3f} "
              f"micro-F1={results['llm_test']['micro_f1']:.3f}")

    out_dir = RESULTS_DIR / "accuracy"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{args.name}.json", "w", encoding="utf8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
