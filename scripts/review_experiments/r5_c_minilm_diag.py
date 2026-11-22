"""Review round 5, C2 diagnostic: why does MiniLM-L6 + KD lose accuracy under quantize_dynamic?

The teacher-style setting (per_channel, reduce_range, QInt8 weights, all quantizable ops incl. the embedding Gather)
drops MiniLM-L6 + KD from 0.783 to 0.535 dev macro-F1, while DistilmBERT keeps its score. This script quantizes
MiniLM with a few alternative settings, scores them on dev (1,000 docs), and evaluates the best one that differs from
the teacher setting on the test set as an extra (clearly labelled) row. Nothing in models/baselines is changed.

Outputs: models/review_r5/minilm_kd_int8_<variant>/, results/review_experiments/r5_c_predictions/minilm_kd_<variant>__<set>.npz,
         results/review_experiments/r5_c_minilm_diag.json
Run from the project root:  env/Scripts/python.exe scripts/review_experiments/r5_c_minilm_diag.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "review_experiments"))

import numpy as np  # noqa: E402

from statickd import evaluation as ev  # noqa: E402
from statickd.baselines import dir_size_mb  # noqa: E402
from statickd.config import MODELS_DIR  # noqa: E402

from r5_c_prepare_eval import PRED_DIR, onnx_logits, save, summary  # noqa: E402

SRC = MODELS_DIR / "baselines" / "minilm_kd_t1"
OUT = ROOT / "results/review_experiments/r5_c_minilm_diag.json"
VARIANTS = {
    "matmul_pc_rr": dict(op_types_to_quantize=["MatMul"], per_channel=True, reduce_range=True),
    "all_tensor": dict(per_channel=False, reduce_range=False),
    "matmul_tensor": dict(op_types_to_quantize=["MatMul"], per_channel=False, reduce_range=False),
    "gather_only_pc_rr": dict(op_types_to_quantize=["Gather"], per_channel=True, reduce_range=True),
}


def main() -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    res = {"teacher_setting_dev": json.loads((ROOT / "results/review_experiments/r5_c_prepare.json").read_text())["c2_onnx_int8"]["minilm_kd"]["dev"]}
    texts = ev.eval_set("dev")["text"].tolist()
    fp32 = np.load(PRED_DIR / "minilm_kd_onnx_fp32__dev.npz")["logits"]
    for name, kw in VARIANTS.items():
        dst = MODELS_DIR / "review_r5" / f"minilm_kd_int8_{name}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "train_meta.json", "special_tokens_map.json"):
            if (SRC / f).exists():
                shutil.copy2(SRC / f, dst / f)
        quantize_dynamic(str(SRC / "model.onnx"), str(dst / "model.onnx"), weight_type=QuantType.QInt8,
                         extra_options={"EnableSubgraph": True, "MatMulConstBOnly": True}, **kw)
        t0 = time.time()
        L = onnx_logits(dst, "model.onnx", texts, 256)
        save(f"minilm_kd_int8_{name}", "dev", L)
        res[name] = {"settings": {k: v for k, v in kw.items()}, "onnx_mb": round(dir_size_mb(dst / "model.onnx"), 2),
                     "dev": {**summary("x", L, "dev"), "argmax_changes_vs_onnx_fp32": int((L.argmax(1) != fp32.argmax(1)).sum()),
                             "seconds": round(time.time() - t0, 1)}}
        print(name, res[name], flush=True)
        OUT.write_text(json.dumps(res, indent=1))
    # best alternative on dev -> test
    best = max(VARIANTS, key=lambda n: res[n]["dev"]["macro_f1"])
    texts_t = ev.eval_set("test")["text"].tolist()
    dst = MODELS_DIR / "review_r5" / f"minilm_kd_int8_{best}"
    t0 = time.time()
    L = onnx_logits(dst, "model.onnx", texts_t, 256)
    save(f"minilm_kd_int8_{best}", "test", L)
    f32t = np.load(PRED_DIR / "minilm_kd_onnx_fp32__test.npz")["logits"]
    res["best_on_dev"] = best
    res["best_test"] = {**summary("x", L, "test"), "argmax_changes_vs_onnx_fp32": int((L.argmax(1) != f32t.argmax(1)).sum()),
                        "n": len(L), "seconds": round(time.time() - t0, 1)}
    print("best", best, res["best_test"], flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
