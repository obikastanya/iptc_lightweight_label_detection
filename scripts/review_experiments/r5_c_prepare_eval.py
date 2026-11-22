"""Review round 5, part C (preparation + accuracy): reduced-precision variants and their predictions.

C1  Model2Vec classifier (models/baselines/model2vec_clf, MLP head) with the embedding matrix stored in fp16, and in
    int8 with model2vec's own quantize_embeddings (DType.Int8: one global scale, not stored; the pipeline loader
    StaticModelPipeline.from_pretrained has no quantize_to argument, so the quantized matrix is saved and loaded as is).
C2  MiniLM-L6 + KD (minilm_kd_t1, 256 tokens) and DistilmBERT + KD (256 tokens): their fp32 ONNX exports quantized with
    onnxruntime.quantization.quantize_dynamic. The teacher int8 model was not quantized in this project: it is
    onnx-community's model_quantized.onnx (transformers.js conversion; its quantize_config.json: per_channel=true,
    reduce_range=true; the graph has MatMulInteger nodes with per-channel int8 weights and dynamically quantized uint8
    activations). The same settings are used here: per_channel=True, reduce_range=True, weight_type=QInt8, extra_options
    EnableSubgraph/MatMulConstBOnly. The fp32 ONNX models are also evaluated through the same CPU code path so that the
    int8 effect is not mixed with the PyTorch-GPU vs ONNX difference of the stored predictions.

Outputs: models/review_r5/{model2vec_clf_fp16, model2vec_clf_int8, minilm_kd_int8, distilmbert_kd_int8}/,
         results/review_experiments/r5_c_predictions/<name>__<set>.npz, results/review_experiments/r5_c_prepare.json
Run from the project root:  env/Scripts/python.exe scripts/review_experiments/r5_c_prepare_eval.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import statickd.config as config  # noqa: E402,F401
from statickd import evaluation as ev  # noqa: E402
from statickd.baselines import Model2VecPredictor, OnnxClassifier, dir_size_mb  # noqa: E402
from statickd.config import MODELS_DIR  # noqa: E402

B = MODELS_DIR / "baselines"
OUT_MODELS = MODELS_DIR / "review_r5"
PRED_DIR = ROOT / "results/review_experiments/r5_c_predictions"
OUT_JSON = ROOT / "results/review_experiments/r5_c_prepare.json"
SETS = ("dev", "test")


def save(name, s, L):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRED_DIR / f"{name}__{s}.npz", logits=L.astype(np.float32))


def summary(name, L, s):
    view = ev.eval_view(s)
    y = view.y.to_numpy()
    p = L[view.index.to_numpy()].argmax(1)
    return {"macro_f1": ev.macro_f1(y, p), "accuracy": float((y == p).mean())}


def c1() -> dict:
    from model2vec.inference import StaticModelPipeline
    from model2vec.quantization import DType, quantize_embeddings

    out = {}
    src = B / "model2vec_clf"
    base = StaticModelPipeline.from_pretrained(str(src))
    out["fp32_embedding_dtype"] = str(base.model.embedding.dtype)
    for tag, dt in (("fp16", DType.Float16), ("int8", DType.Int8)):
        name = f"model2vec_clf_{tag}"
        pipe = StaticModelPipeline.from_pretrained(str(src))
        pipe.model.embedding = quantize_embeddings(pipe.model.embedding, dt)
        dst = OUT_MODELS / name
        if dst.exists():
            shutil.rmtree(dst)
        pipe.save_pretrained(str(dst))
        if (src / "train_meta.json").exists():
            shutil.copy2(src / "train_meta.json", dst / "train_meta.json")
        r = {"saved_dir": str(dst.relative_to(ROOT)), "size_mb": round(dir_size_mb(dst), 2),
             "files_mb": {f.name: round(f.stat().st_size / 2**20, 2) for f in dst.iterdir()}}
        try:
            pred = Model2VecPredictor(dst)
            r["loaded_embedding_dtype"] = str(pred.pipeline.model.embedding.dtype)
            r["loads"] = True
        except Exception as e:  # noqa: BLE001
            r["loads"] = False
            r["error"] = repr(e)
            out[tag] = r
            continue
        for s in SETS:
            t0 = time.time()
            L = ev.predict_logits(pred, ev.eval_set(s)["text"].tolist(), 1024)
            save(name, s, L)
            ref = ev.load_logits("model2vec_clf", s)
            r[s] = {**summary(name, L, s), "fp32": summary("model2vec_clf", ref, s),
                    "argmax_changes_vs_fp32": int((L.argmax(1) != ref.argmax(1)).sum()), "n": len(L),
                    "max_abs_logprob_diff": float(np.abs(L - ref).max()), "seconds": round(time.time() - t0, 1)}
            print(f"[C1] {name} {s}", r[s], flush=True)
        out[tag] = r
    out["fp32_size_mb"] = round(dir_size_mb(src), 2)
    return out


def onnx_logits(model_dir: Path, onnx_file: str, texts: list[str], max_length: int, threads: int = 8, batch: int = 32) -> np.ndarray:
    clf = OnnxClassifier(model_dir, onnx_file, threads=threads, max_length=max_length)
    order = np.argsort([len(t) for t in texts], kind="stable")   # length-sorted batches: less padding, same logits
    out = np.empty((len(texts), 17), np.float32)
    for i in range(0, len(texts), batch):
        idx = order[i:i + batch]
        out[idx] = clf.logits([texts[j] for j in idx])
    return out


def c2() -> dict:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = {}
    for name, src_name in (("minilm_kd", "minilm_kd_t1"), ("distilmbert_kd", "distilmbert_kd")):
        src = B / src_name
        dst = OUT_MODELS / f"{name}_int8"
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "train_meta.json", "special_tokens_map.json"):
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        t0 = time.time()
        quantize_dynamic(str(src / "model.onnx"), str(dst / "model.onnx"), per_channel=True, reduce_range=True,
                         weight_type=QuantType.QInt8, extra_options={"EnableSubgraph": True, "MatMulConstBOnly": True})
        g = onnx.load(str(dst / "model.onnx"), load_external_data=False)
        ops = {}
        for n in g.graph.node:
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
        max_len = int(json.loads((src / "train_meta.json").read_text())["max_length"])
        r = {"quantize_seconds": round(time.time() - t0, 1), "fp32_onnx_mb": round(dir_size_mb(src / "model.onnx"), 2),
             "int8_onnx_mb": round(dir_size_mb(dst / "model.onnx"), 2), "max_length": max_len,
             "int8_ops": {k: ops[k] for k in sorted(ops, key=ops.get, reverse=True)[:12]}}
        print(f"[C2] {name} quantized", r, flush=True)
        for s in SETS:
            texts = ev.eval_set(s)["text"].tolist()
            res = {}
            for tag, d in (("onnx_fp32", src), ("onnx_int8", dst)):
                t0 = time.time()
                L = onnx_logits(d, "model.onnx", texts, max_len)
                save(f"{name}_{tag}", s, L)
                res[tag] = {**summary(name, L, s), "seconds": round(time.time() - t0, 1)}
                print(f"[C2] {name} {tag} {s}", res[tag], flush=True)
            stored = ev.load_logits(src_name, s)
            Lf = np.load(PRED_DIR / f"{name}_onnx_fp32__{s}.npz")["logits"]
            Li = np.load(PRED_DIR / f"{name}_onnx_int8__{s}.npz")["logits"]
            res["stored_pytorch_gpu"] = summary(src_name, stored, s)
            res["argmax_changes_int8_vs_onnx_fp32"] = int((Li.argmax(1) != Lf.argmax(1)).sum())
            res["argmax_changes_onnx_fp32_vs_stored"] = int((Lf.argmax(1) != stored.argmax(1)).sum())
            res["argmax_changes_int8_vs_stored"] = int((Li.argmax(1) != stored.argmax(1)).sum())
            res["n"] = len(texts)
            r[s] = res
        out[name] = r
        prev = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
        OUT_JSON.write_text(json.dumps({**prev, "c2_partial": out}, indent=1))
    return out


def main() -> None:
    t0 = time.time()
    res = {"c1_model2vec": c1()}
    OUT_JSON.write_text(json.dumps(res, indent=1))
    res["c2_onnx_int8"] = c2()
    res["seconds"] = round(time.time() - t0, 1)
    res.pop("c2_partial", None)
    OUT_JSON.write_text(json.dumps(res, indent=1))
    print("written", OUT_JSON)


if __name__ == "__main__":
    main()
