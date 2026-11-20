"""E3: do the stored StaticKD test predictions come from the saved fp16 final model?

Reloads models/final/statickd (fp16 token_table + fp32 bias), predicts the unified test set with the
repo's own path (evaluation.eval_set("test") texts -> evaluation.predict_logits(batch 64) ->
StaticKDModel.logits), and compares with results/final/predictions/statickd__test.npz.
Nothing is written to results/final.

Usage (from the project root):
    env/Scripts/python.exe scripts/review_experiments/e3_fp16_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from statickd import evaluation as ev  # noqa: E402
from statickd.student import StaticKDModel  # noqa: E402

MODEL_DIR = ROOT / "models" / "final" / "statickd"
OUT = ROOT / "results" / "review_experiments" / "e3_fp16_check.json"


def main() -> None:
    model = StaticKDModel(MODEL_DIR)
    data = np.load(MODEL_DIR / "model.npz")
    stored_file = ev.prediction_path("statickd", "test")
    stored = ev.load_logits("statickd", "test")
    test = ev.eval_set("test")
    fresh = ev.predict_logits(model, test["text"].tolist(), batch_size=64)
    y = test["y"].to_numpy()
    diff = np.abs(fresh - stored)
    res = {
        "model_dir": str(MODEL_DIR.relative_to(ROOT)),
        "arrays_in_model_npz": {k: [str(data[k].dtype), list(data[k].shape)] for k in data.files},
        "stored_predictions": str(stored_file.relative_to(ROOT)),
        "n_docs": int(len(stored)),
        "argmax_differences": int((fresh.argmax(1) != stored.argmax(1)).sum()),
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "macro_f1_stored": ev.macro_f1(y, stored.argmax(1)),
        "macro_f1_fresh": ev.macro_f1(y, fresh.argmax(1)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
