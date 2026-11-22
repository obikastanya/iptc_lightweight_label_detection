"""Review round 5, B1: Model2Vec classifier with a linear head (n_layers=0), folded into a token table.

Training is identical to scripts/train_model2vec.py (same 285,999 distillation documents, hard teacher labels,
library defaults, seed 0, potion-multilingual-128M start), except n_layers=0 (the library then builds a single
nn.Linear(256, 17) head). Two variants:
  m2v_lin        n_layers=0, every other setting the library default (normalize=True: the pooled vector is
                 L2-normalised before the head, as in the existing baseline)
  m2v_lin_nonorm n_layers=0 and normalize=False. Needed because with normalize=True the logit of a document is
                 W s / ||s|| + b (s = weighted token sum): the per-document norm needs the 256-dim embeddings, so the
                 head cannot be folded into a 17-logit table exactly. Without normalisation logits = mean_t(T[t]) + b
                 with T = (sigmoid(w_t) e_t) W^T, which is exactly the StaticKD deployment form.

Outputs (nothing in models/final or results/final is touched):
  models/review_r5/<variant>/                 model2vec pipeline (to_pipeline().save_pretrained) + train_meta.json
  models/review_r5/<variant>_folded/          folded fp16 token table in StaticKD format (model.npz, tokenizer.json, meta.json)
  results/review_experiments/r5_b1_predictions/<name>__<set>.npz   logits (pseudo-logits = log-probs for the pipelines)
  results/review_experiments/r5_b1.json

Run from the project root:  env/Scripts/python.exe scripts/review_experiments/r5_b1_model2vec_linear.py [--variants ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import statickd.config as config  # noqa: E402,F401
from statickd import evaluation as ev  # noqa: E402
from statickd.baselines import Model2VecPredictor, dir_size_mb  # noqa: E402
from statickd.config import BASE_EMBEDDINGS, LABELS, MODELS_DIR, NUM_LABELS  # noqa: E402
from statickd.data import load_distillation_pool, softmax, teacher_probs  # noqa: E402

OUT_MODELS = MODELS_DIR / "review_r5"
PRED_DIR = ROOT / "results/review_experiments/r5_b1_predictions"
OUT_JSON = ROOT / "results/review_experiments/r5_b1.json"
VARIANTS = {"m2v_lin": {"n_layers": 0}, "m2v_lin_nonorm": {"n_layers": 0, "normalize": False}}


def save_logits(name: str, s: str, L: np.ndarray) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRED_DIR / f"{name}__{s}.npz", logits=L.astype(np.float32))


def train_variant(name: str, kwargs: dict, texts, y, seed: int = 0) -> dict:
    import model2vec
    import torch
    from model2vec.train import StaticModelForClassification

    torch.manual_seed(seed)
    out = OUT_MODELS / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    clf = StaticModelForClassification.from_pretrained(BASE_EMBEDDINGS, **kwargs)
    assert clf.n_layers == 0 and len([m for m in clf.head if isinstance(m, torch.nn.Linear)]) == 1
    clf = clf.fit(texts, y, device="auto", random_seed=seed)
    train_seconds = time.perf_counter() - t0
    pipe = clf.to_pipeline()
    pipe.save_pretrained(str(out))
    meta = {"name": name, "base_embeddings": BASE_EMBEDDINGS, "model2vec_version": model2vec.__version__,
            "labels": "teacher arg-max (hard)", "train_docs": len(texts), "seed": seed, "overrides": kwargs,
            "normalize": bool(clf.normalize), "n_layers": clf.n_layers, "head": str(clf.head),
            "train_seconds": round(train_seconds, 1), "device": "auto"}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    return meta


def fold(name: str) -> dict:
    """Fold the linear head into T = E_w W^T (E_w = embeddings with the learned token weights baked in)."""
    from model2vec.inference import StaticModelPipeline
    from statickd.student import StaticKDModel

    src = OUT_MODELS / name
    pipe = StaticModelPipeline.from_pretrained(str(src))
    sm, head = pipe.model, pipe.head
    assert len(head.layers) == 1, "not a linear head"
    assert list(np.asarray(head.classes_).astype(int)) == list(range(NUM_LABELS))
    info = {"pipeline_weights_separate": sm.weights is not None, "token_mapping": sm.token_mapping is not None,
            "normalize": bool(sm.normalize), "embedding_shape": list(sm.embedding.shape), "embedding_dtype": str(sm.embedding.dtype)}
    E = sm.embedding.astype(np.float32)
    if sm.token_mapping is not None:
        E = E[sm.token_mapping]
    if sm.weights is not None:  # learned token weights kept separately (only when the base was quantized)
        E = E * sm.weights[:, None]
    W, b = head.layers[0].weight.astype(np.float32), head.layers[0].bias.astype(np.float32)
    T32 = E @ W.T
    out = OUT_MODELS / f"{name}_folded"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "model.npz", token_table=T32.astype(np.float16), bias=b)
    sm.tokenizer.save(str(out / "tokenizer.json"), pretty=False)
    (out / "meta.json").write_text(json.dumps({"format": "token_table_fp16", "labels": LABELS, "max_tokens": 512,
                                               "folded_from": str(src.relative_to(ROOT)), "normalize_in_source": bool(sm.normalize)}, indent=1),
                                   encoding="utf8")
    info["folded_dir"] = str(out.relative_to(ROOT))
    info["folded_files_mb"] = {f.name: round(f.stat().st_size / 2**20, 2) for f in out.iterdir()}
    info["folded_size_mb"] = round(dir_size_mb(out), 2)
    info["pipeline_size_mb"] = round(dir_size_mb(src), 2)
    info["bias_abs_max"] = float(np.abs(b).max())

    # verification on dev + test texts
    folded = StaticKDModel(out)
    checks = {}
    for s in ("dev", "test"):
        texts = ev.eval_set(s)["text"].tolist()
        # (1) the pipeline's own path
        L_pipe = np.concatenate([np.log(np.clip(pipe.predict_proba(texts[i:i + 1024], max_length=512, use_multiprocessing=False), 1e-12, 1))
                                 for i in range(0, len(texts), 1024)])
        # (2) table applied to the pipeline's own token ids (isolates the fold from tokenisation differences)
        ids = sm.tokenize(texts, max_length=512)
        H = np.stack([E[i].mean(0) if len(i) else np.zeros(E.shape[1], np.float32) for i in ids])
        z_unfolded = (H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-32) if sm.normalize else H) @ W.T + b
        z_t32 = np.stack([T32[i].mean(0) if len(i) else np.zeros(NUM_LABELS, np.float32) for i in ids]) + b
        z_t16 = np.stack([folded.table[i].mean(0, dtype=np.float32) if len(i) else np.zeros(NUM_LABELS, np.float32) for i in ids]) + b
        if sm.normalize:  # exact only with the per-document norm of the 256-dim embedding
            norms = np.linalg.norm(H, axis=1, keepdims=True) + 1e-32
            z_t32_exact = (z_t32 - b) / norms + b
            info.setdefault("median_doc_norm", {})[s] = float(np.median(norms))
        else:
            z_t32_exact = z_t32
        # (3) the StaticKD deployment code (its own tokenisation: truncate at 512 ids then drop [UNK])
        L_dep = np.concatenate([folded.logits(texts[i:i + 512]) for i in range(0, len(texts), 512)])
        p_pipe = L_pipe.argmax(1)
        checks[s] = {
            "unfolded_recomputed_vs_pipeline_argmax_agreement": float((z_unfolded.argmax(1) == p_pipe).mean()),
            "table_fp32_exact_form_vs_pipeline_argmax_agreement": float((z_t32_exact.argmax(1) == p_pipe).mean()),
            "table_fp32_exact_form_max_abs_logit_diff_vs_unfolded": float(np.abs(z_t32_exact - z_unfolded).max()),
            "table_fp16_plain_mean_vs_pipeline_argmax_agreement": float((z_t16.argmax(1) == p_pipe).mean()),
            "table_fp16_plain_mean_max_abs_logit_diff_vs_unfolded": float(np.abs(z_t16 - z_unfolded).max()),
            "statickd_code_vs_pipeline_argmax_agreement": float((L_dep.argmax(1) == p_pipe).mean()),
            "n": len(texts)}
        print(f"[fold {name}] {s}", checks[s], flush=True)
    info["checks"] = checks
    return info


def evaluate(name: str, predictor, batch_size: int) -> dict:
    out = {}
    for s in ev.EVAL_SETS:
        texts = ev.eval_set(s)["text"].tolist()
        t0 = time.time()
        L = ev.predict_logits(predictor, texts, batch_size)
        save_logits(name, s, L)
        view = ev.eval_view(s)
        p = L[view.index.to_numpy()].argmax(1)
        out[s] = {"macro_f1": ev.macro_f1(view.y.to_numpy(), p), "accuracy": float((view.y.to_numpy() == p).mean()),
                  "seconds": round(time.time() - t0, 1)}
        print(f"[eval {name}] {s} {out[s]}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    res = json.loads(OUT_JSON.read_text(encoding="utf8")) if OUT_JSON.exists() else {}
    if not args.skip_train:
        pool = load_distillation_pool()
        texts = pool["text"].tolist()
        y = teacher_probs(pool).argmax(1).astype(int).tolist()
    from statickd.student import StaticKDModel
    for v in args.variants:
        r = res.get(v, {})
        if not args.skip_train:
            r["train"] = train_variant(v, VARIANTS[v], texts, y)
        r["eval_pipeline"] = evaluate(v, Model2VecPredictor(OUT_MODELS / v), batch_size=1024)
        r["fold"] = fold(v)
        r["eval_folded_statickd_code"] = evaluate(f"{v}_folded", StaticKDModel(OUT_MODELS / f"{v}_folded"), batch_size=512)
        res[v] = r
        OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf8")
    print("written", OUT_JSON)


if __name__ == "__main__":
    main()
