"""Phase 1 of the accuracy exploration: inference-time variants of a folded StaticKD model.

No retraining. Every variant keeps the deployment format (token table + bias) and the
O(tokens) cost:
  * truncation      only the first L tokens are used (the teacher itself reads 512 tokens)
  * position weight tokens among the first k get weight (1 + alpha)  (title / lead emphasis)
  * idf weight      token weight = idf^p from document frequencies of the training corpus
  * bias calibration 17 additive class offsets tuned for macro-F1 on dev with 5-fold CV
                    (reported score is out-of-fold, so it is not optimistically biased)

Usage:
    python experiments/explore_inference.py --model ../models/static/static_final_table
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR, RESULTS_DIR  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from data.emmediatopic import load_emmediatopic  # noqa: E402
from evaluation.evaluate_accuracy import LLM_TEST, TEACHER_CCNEWS_HELDOUT  # noqa: E402
from students.static_student import _unk_id  # noqa: E402

TRAIN_TOKEN_CACHE = PROCESSED_DIR / "token_cache" / "3d5e782f77f1.npz"  # 286k training docs


def macro(y, p):
    return f1_score(y, p, average="macro", labels=np.unique(y), zero_division=0)


def load_sets():
    dev = load_emmediatopic("dev")
    cc = pd.read_parquet(TEACHER_CCNEWS_HELDOUT)
    llm = pd.read_json(LLM_TEST, lines=True)
    return {"dev": (dev["text"].tolist(), dev["label_id"].to_numpy()),
            "llm_test": (llm["text"].tolist(), llm["label_id"].to_numpy()),
            "ccnews": (cc["text"].tolist(), cc["teacher_label_id"].to_numpy())}


def encode_all(tokenizer, texts, unk, max_tokens=2048):
    out = []
    for e in tokenizer.encode_batch(texts, add_special_tokens=False):
        ids = np.asarray(e.ids[:max_tokens], dtype=np.int64)
        if unk is not None:
            ids = ids[ids != unk]
        out.append(ids if len(ids) else np.zeros(1, dtype=np.int64))
    return out


def idf_from_training(vocab_size: int) -> np.ndarray:
    data = np.load(TRAIN_TOKEN_CACHE)
    docs = np.split(data["flat"], np.cumsum(data["lengths"])[:-1])
    df = np.zeros(vocab_size, dtype=np.int64)
    for d in docs:
        df[np.unique(d)] += 1
    return np.log((1 + len(docs)) / (1 + df)) + 1.0


def doc_logits(table, bias, docs, max_len=1024, k=0, alpha=0.0, token_weight=None):
    out = np.empty((len(docs), table.shape[1]), dtype=np.float32)
    for j, ids in enumerate(docs):
        ids = ids[:max_len]
        rows = table[ids].astype(np.float32)
        w = np.ones(len(ids), dtype=np.float32)
        if token_weight is not None:
            w = w * token_weight[ids]
        if alpha:
            w[:k] *= 1.0 + alpha
        out[j] = (w[:, None] * rows).sum(0) / w.sum()
    return out + bias


def fit_bias(logits, y, n_rounds=3, grid=np.linspace(-1.5, 1.5, 31)):
    """Coordinate ascent on 17 additive class offsets maximising macro-F1."""
    delta = np.zeros(logits.shape[1], dtype=np.float32)
    best = macro(y, (logits + delta).argmax(1))
    for _ in range(n_rounds):
        for c in range(logits.shape[1]):
            keep = delta[c]
            for g in grid:
                delta[c] = g
                s = macro(y, (logits + delta).argmax(1))
                if s > best + 1e-9:
                    best, keep = s, g
            delta[c] = keep
    return delta


def cv_bias(logits, y, folds=5, seed=0):
    """Out-of-fold predictions with class offsets fitted on the other folds."""
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % folds
    pred = np.empty(len(y), dtype=np.int64)
    for f in range(folds):
        tr, te = fold != f, fold == f
        delta = fit_bias(logits[tr], y[tr])
        pred[te] = (logits[te] + delta).argmax(1)
    return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../models/static/static_final_table")
    args = parser.parse_args()

    path = Path(args.model)
    data = np.load(path / "model.npz")
    table, bias = data["token_table"], data["bias"]
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    unk = _unk_id(tokenizer)
    sets = {name: (encode_all(tokenizer, texts, unk), y) for name, (texts, y) in load_sets().items()}

    def score_all(**kw):
        res = {}
        for name, (docs, y) in sets.items():
            p = doc_logits(table, bias, docs, **kw).argmax(1)
            res[name] = float((p == y).mean()) if name == "ccnews" else float(macro(y, p))
        return res

    rows = []

    def add(group, config, **kw):
        r = score_all(**kw)
        rows.append({"group": group, "config": config, **r})
        print(f"{group:10s} {config:28s} dev={r['dev']:.4f} llm={r['llm_test']:.4f} cc={r['ccnews']:.4f}", flush=True)

    add("baseline", "L=1024")
    for L in (64, 128, 256, 384, 512, 768, 2048):
        add("truncate", f"L={L}", max_len=L)
    for k, alpha in itertools.product((16, 32, 64), (0.5, 1.0, 2.0)):
        add("position", f"k={k} alpha={alpha}", k=k, alpha=alpha)
    idf = idf_from_training(len(table)).astype(np.float32)
    for p in (0.5, 1.0, 2.0):
        add("idf", f"idf^{p}", token_weight=idf ** p)

    # bias calibration: 5-fold CV on dev (out-of-fold macro-F1); offsets fitted on the whole of
    # dev are then applied unchanged to the other two sets.
    docs, y = sets["dev"]
    dev_logits = doc_logits(table, bias, docs)
    oof = cv_bias(dev_logits, y)
    delta = fit_bias(dev_logits, y)
    other = {}
    for name in ("llm_test", "ccnews"):
        d, yy = sets[name]
        p = (doc_logits(table, bias, d) + delta).argmax(1)
        other[name] = float((p == yy).mean()) if name == "ccnews" else float(macro(yy, p))
    rows.append({"group": "bias", "config": "5-fold CV on dev", "dev": float(macro(y, oof)), **other})
    print(f"bias       5-fold CV on dev             dev={rows[-1]['dev']:.4f} llm={other['llm_test']:.4f} "
          f"cc={other['ccnews']:.4f}  offsets={np.round(delta, 2).tolist()}")

    out = RESULTS_DIR / "exploration"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"phase1_inference_{path.name}.csv", index=False)
    (out / f"phase1_bias_offsets_{path.name}.json").write_text(json.dumps(delta.tolist()))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
