"""Evaluation sets, cached predictions, metrics and statistical tests.

Every model is run once per evaluation set; its 17 logits per document are stored in
results/final/predictions/<model>__<set>.npz, and all tables of the paper are computed from these
files. Evaluation sets:
    dev      EMMediaTopic dev (1,000 docs, 4 languages, GPT-4o labels)       -> development
    heldout  CC-News held-out (12,554 docs from unseen sites, 63 languages)  -> fidelity
    test     unified test set (12,744 docs, 47 languages, independent labels) -> final test
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score

from statickd.config import LABELS, NUM_LABELS, PREDICTIONS_DIR
from statickd.data import (load_ccnews_heldout, load_distillation_pool, load_emmediatopic, load_unified_test,
                           site_clusters, softmax)

EVAL_SETS = ("dev", "heldout", "test")


# ----------------------------------------------------------------------------- data
@lru_cache(maxsize=None)
def eval_set(name: str) -> pd.DataFrame:
    """Evaluation set with the columns text, lang, y (reference label id) and cluster."""
    if name == "dev":
        df = load_emmediatopic("dev")
        df = df.assign(y=df["label_id"], cluster="emmediatopic")
    elif name == "heldout":
        df = load_ccnews_heldout()
        # the held-out split is domain-disjoint per language; a few multilingual sites (e.g. bbc.com,
        # dw.com) also occur in other languages of the distillation data -> excluded by eval_view()
        seen = set(load_distillation_pool()["domain"])
        df = df.assign(y=df["teacher_label_id"], cluster=df["domain"], site_unseen=~df["domain"].isin(seen))
    elif name == "test":
        df = load_unified_test()
        df = df.assign(y=df["label_id"], cluster=site_clusters(df))
    else:
        raise ValueError(name)
    return df.reset_index(drop=True)


def eval_view(name: str) -> pd.DataFrame:
    """The documents that are scored. For `heldout` only articles from sites that never occur in the
    distillation data (in any language); the other sets are used completely. The index gives the
    position of each document in the stored predictions."""
    df = eval_set(name)
    return df[df["site_unseen"]] if name == "heldout" else df


def pred_view(model: str, name: str) -> np.ndarray:
    return load_pred(model, name)[eval_view(name).index.to_numpy()]


def logits_view(model: str, name: str) -> np.ndarray:
    return load_logits(model, name)[eval_view(name).index.to_numpy()]


# ----------------------------------------------------------------------------- predictions
def predict_logits(predictor, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """`predictor` exposes .logits(texts) (preferred) or .predict(texts)."""
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if hasattr(predictor, "logits"):
            out.append(np.asarray(predictor.logits(batch), dtype=np.float32))
        else:  # label-only predictors (fastText returns probabilities through .logits too)
            onehot = np.full((len(batch), NUM_LABELS), -1e4, dtype=np.float32)
            onehot[np.arange(len(batch)), predictor.predict(batch)] = 0.0
            out.append(onehot)
    return np.concatenate(out)


def prediction_path(model: str, set_name: str):
    return PREDICTIONS_DIR / f"{model}__{set_name}.npz"


def save_predictions(model: str, set_name: str, logits: np.ndarray) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_path(model, set_name), logits=logits.astype(np.float32))


def has_predictions(model: str, set_name: str) -> bool:
    return prediction_path(model, set_name).exists()


def load_logits(model: str, set_name: str) -> np.ndarray:
    return np.load(prediction_path(model, set_name))["logits"]


def load_pred(model: str, set_name: str) -> np.ndarray:
    return load_logits(model, set_name).argmax(1)


def evaluate_model(model_name: str, predictor, sets=EVAL_SETS, overwrite: bool = False,
                   batch_size: int = 64) -> dict:
    """Store the logits of `predictor` on every evaluation set and return a small summary."""
    summary = {}
    for s in sets:
        if overwrite or not has_predictions(model_name, s):
            save_predictions(model_name, s, predict_logits(predictor, eval_set(s)["text"].tolist(), batch_size))
        summary[s] = scores(eval_view(s)["y"].to_numpy(), pred_view(model_name, s))
    return summary


# ----------------------------------------------------------------------------- metrics
def macro_f1(y, p, labels=None) -> float:
    """Macro-F1 over the labels present in the reference (as in Kuzman & Ljubesic's code).

    Same result as sklearn f1_score(average="macro", labels=present, zero_division=0), computed from
    a confusion matrix with NumPy because the bootstrap calls it hundreds of thousands of times."""
    y, p = np.asarray(y, dtype=np.int64), np.asarray(p, dtype=np.int64)
    cm = np.bincount(y * NUM_LABELS + p, minlength=NUM_LABELS * NUM_LABELS).reshape(NUM_LABELS, NUM_LABELS)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(NUM_LABELS), where=denom > 0)
    labels = np.flatnonzero(support) if labels is None else np.asarray(labels)
    return float(f1[labels].mean())


def scores(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return {"macro_f1": macro_f1(y, p), "accuracy": float((y == p).mean()), "n": int(len(y))}


def per_label_f1(y, p) -> np.ndarray:
    return f1_score(y, p, labels=list(range(NUM_LABELS)), average=None, zero_division=0)


def confusion(y, p, normalise: bool = False) -> np.ndarray:
    cm = confusion_matrix(y, p, labels=list(range(NUM_LABELS))).astype(float)
    if normalise:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    return cm


def grouped_macro_f1(y, p, groups) -> float:
    """Mean over groups (e.g. languages) of the macro-F1 over each group's present labels."""
    y, p, groups = map(np.asarray, (y, p, groups))
    return float(np.mean([macro_f1(y[groups == g], p[groups == g]) for g in np.unique(groups)]))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of the top-1 probability (Guo et al., 2017)."""
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    rows = [{"bin": b, "confidence": conf[bins == b].mean(), "accuracy": (pred[bins == b] == y[bins == b]).mean(),
             "n": int((bins == b).sum())} for b in range(n_bins) if (bins == b).any()]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- statistics
def _cluster_index(clusters) -> list[np.ndarray]:
    clusters = np.asarray(clusters)
    _, inverse = np.unique(clusters, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.cumsum(np.bincount(inverse))[:-1]
    return np.split(order, bounds)


def bootstrap_samples(n: int, clusters=None, n_boot: int = 1000, seed: int = 0):
    """Yield index arrays of bootstrap resamples. With `clusters`, whole clusters (news sites) are
    resampled with replacement, which keeps articles of the same site together (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    if clusters is None:
        for _ in range(n_boot):
            yield rng.integers(0, n, n)
        return
    groups = _cluster_index(clusters)
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        yield np.concatenate([groups[g] for g in pick])


def bootstrap_ci(y, p, metric=macro_f1, clusters=None, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of `metric` (document- or cluster-level)."""
    y, p = np.asarray(y), np.asarray(p)
    vals = [metric(y[i], p[i]) for i in bootstrap_samples(len(y), clusters, n_boot, seed)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap(y, p_a, p_b, metric=macro_f1, clusters=None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Difference metric(a) - metric(b) on the same documents, its 95% interval and a two-sided
    bootstrap p-value (share of resamples whose difference has the opposite sign, doubled)."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    delta = metric(y, p_a) - metric(y, p_b)
    diffs = np.array([metric(y[i], p_a[i]) - metric(y[i], p_b[i])
                      for i in bootstrap_samples(len(y), clusters, n_boot, seed)])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta": float(delta), "ci_low": float(lo), "ci_high": float(hi), "p": float(min(1.0, p))}


def mcnemar(y, p_a, p_b) -> dict:
    """Exact McNemar test on the documents where exactly one of the two models is correct."""
    a, b = np.asarray(p_a) == np.asarray(y), np.asarray(p_b) == np.asarray(y)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a": only_a, "only_b": only_b, "p": float(p)}


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def welch(a, b) -> dict:
    """Welch t-test between two groups of seed-level scores."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = stats.ttest_ind(a, b, equal_var=False)
    return {"delta": float(a.mean() - b.mean()), "t": float(t.statistic), "p": float(t.pvalue)}


def probs_from(model: str, set_name: str) -> np.ndarray:
    return softmax(load_logits(model, set_name))
