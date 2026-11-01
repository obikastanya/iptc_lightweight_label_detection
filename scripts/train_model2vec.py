"""Model2Vec classifier baseline on the same teacher labels as the other + KD baselines.

The Model2Vec library trains a classifier on top of a static model (StaticModelForClassification):
mean pooling with learned token weights, one 512-unit hidden layer, fine-tuned embeddings, hard labels,
and early stopping on a 10% validation split of the training data. Every setting is the library default;
only the starting embeddings are fixed to potion-multilingual-128M, the same as StaticKD. The trained model
is exported with to_pipeline() (NumPy + scikit-learn head) and evaluated on dev, held-out, and test.

Outputs:
  models/baselines/model2vec_clf/            the pipeline + train_meta.json
  results/final/predictions/model2vec_clf__{dev,heldout,test}.npz

Usage (from the project root):
    python scripts/train_model2vec.py [--seed 0] [--device auto]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

import statickd.config as config  # noqa: E402,F401  (sets HF_HOME)
from statickd.baselines import Model2VecPredictor  # noqa: E402
from statickd.config import BASE_EMBEDDINGS, MODELS_DIR  # noqa: E402
from statickd.data import load_distillation_pool, teacher_probs  # noqa: E402
from statickd.evaluation import evaluate_model  # noqa: E402

NAME = "model2vec_clf"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import model2vec
    import torch
    from model2vec.train import StaticModelForClassification

    torch.manual_seed(args.seed)
    pool = load_distillation_pool()
    texts = pool["text"].tolist()
    y = teacher_probs(pool).argmax(1).astype(int).tolist()   # top teacher label, as for fastText and TF-IDF + SVM

    out = MODELS_DIR / "baselines" / NAME
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    clf = StaticModelForClassification.from_pretrained(BASE_EMBEDDINGS)
    clf = clf.fit(texts, y, device=args.device, random_seed=args.seed)
    train_seconds = time.perf_counter() - t0
    clf.to_pipeline().save_pretrained(str(out))

    meta = {"name": NAME, "base_embeddings": BASE_EMBEDDINGS, "model2vec_version": model2vec.__version__,
            "labels": "teacher arg-max (hard)", "train_docs": len(texts), "seed": args.seed,
            "settings": "library defaults: n_layers=1, hidden_dim=512, learned token weights, fine-tuned embeddings, "
                        "lr=1e-3, early stopping (patience 5) on a 10% validation split, max 512 tokens",
            "train_seconds": round(train_seconds, 1), "device": args.device}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)

    summary = evaluate_model(NAME, Model2VecPredictor(out), overwrite=True, batch_size=1024)
    meta["eval"] = {s: {k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()} for s, r in summary.items()}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["eval"], indent=1))


if __name__ == "__main__":
    main()
