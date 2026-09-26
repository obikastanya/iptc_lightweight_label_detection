"""Classification metrics shared by the teacher and student evaluations."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABELS  # noqa: E402


def f1_scores(y_true, y_pred) -> dict:
    """Macro/micro F1 over the 17 IPTC labels, identical to Kuzman's evaluation code."""
    ids = list(range(len(LABELS)))
    present = sorted(set(y_true))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, labels=ids, average="micro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def per_label_f1(y_true, y_pred) -> dict:
    scores = f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average=None, zero_division=0)
    return {LABELS[i]: float(s) for i, s in enumerate(scores)}


def per_group_f1(y_true, y_pred, groups) -> dict:
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    return {g: f1_scores(y_true[groups == g], y_pred[groups == g]) for g in sorted(set(groups))}


def confusion(y_true, y_pred) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist()


def bootstrap_macro_f1_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap confidence interval of macro-F1 (the dev set has only 1,000 documents)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        stats.append(f1_score(y_true[idx], y_pred[idx], labels=sorted(set(y_true[idx])),
                              average="macro", zero_division=0))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)
