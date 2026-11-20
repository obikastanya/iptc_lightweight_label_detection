"""E2 step 1: choose the cascade threshold on the development set only (no test data is loaded).

Rule: the smallest theta in {0.5, 0.6, 0.7, 0.8, 0.9} whose dev cascade macro-F1 is >= the dev macro-F1
of the teacher alone (the ONNX int8 teacher if its dev predictions exist, else the PyTorch teacher).
If no theta qualifies, the theta with the highest dev cascade macro-F1. Ties in the fallback go to the
smaller theta. Cascade: StaticKD decides when its top softmax probability >= theta, otherwise the teacher.

Usage (from the project root):
    env/Scripts/python.exe scripts/review_experiments/e2a_select_threshold.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from statickd import evaluation as ev  # noqa: E402
from statickd.data import softmax  # noqa: E402

THETAS = (0.5, 0.6, 0.7, 0.8, 0.9)
OUT = ROOT / "results" / "review_experiments" / "cascade_threshold.json"
RULE = ("smallest theta in {0.5, 0.6, 0.7, 0.8, 0.9} whose dev cascade macro-F1 >= dev macro-F1 of the "
        "teacher alone; if none qualifies, the theta with the highest dev cascade macro-F1")


def main() -> None:
    if OUT.exists():
        sys.exit(f"{OUT} already exists; the threshold is fixed once (delete it deliberately to redo)")
    teacher = "teacher_onnx_int8" if ev.has_predictions("teacher_onnx_int8", "dev") else "teacher"
    y = ev.eval_view("dev")["y"].to_numpy()
    probs = softmax(ev.logits_view("statickd", "dev"))
    conf, pred_s = probs.max(1), probs.argmax(1)
    pred_t = ev.pred_view(teacher, "dev")
    f1_teacher = ev.macro_f1(y, pred_t)
    rows = []
    for theta in THETAS:
        keep = conf >= theta
        pred = np.where(keep, pred_s, pred_t)
        rows.append({"theta": theta, "statickd_share": float(keep.mean()), "macro_f1": ev.macro_f1(y, pred),
                     "accuracy": float((pred == y).mean()), "meets_teacher": bool(ev.macro_f1(y, pred) >= f1_teacher)})
    qualifying = [r for r in rows if r["meets_teacher"]]
    chosen = qualifying[0] if qualifying else max(rows, key=lambda r: (r["macro_f1"], -r["theta"]))
    res = {"rule": RULE, "teacher_used": teacher, "set": "dev", "n_docs": int(len(y)),
           "teacher_dev_macro_f1": f1_teacher, "teacher_dev_accuracy": float((pred_t == y).mean()),
           "statickd_dev_macro_f1": ev.macro_f1(y, pred_s), "grid": rows,
           "rule_branch": "first qualifying theta" if qualifying else "fallback: highest dev cascade macro-F1",
           "chosen_theta": chosen["theta"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
