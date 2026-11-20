"""E2 step 2: test-set accuracy of the cascade at the dev-chosen threshold (from stored predictions).

Reads the threshold fixed by e2a_select_threshold.py. Paired site-cluster bootstrap (2,000 resamples,
seed 0) of macro-F1(cascade) - macro-F1(ONNX int8 teacher), and bootstrap TOST with margin 0.01 (repo
tost()). Also re-computes the paper's cascade grid as a sanity check.

Usage (from the project root):
    env/Scripts/python.exe scripts/review_experiments/e2b_cascade_test_accuracy.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from statickd import evaluation as ev  # noqa: E402
from statickd.data import softmax  # noqa: E402

RES = ROOT / "results" / "review_experiments"
N_BOOT_PAIRED = 2000
MARGIN = 0.01


def main() -> None:
    sel = json.loads((RES / "cascade_threshold.json").read_text(encoding="utf8"))
    theta, teacher = sel["chosen_theta"], sel["teacher_used"]
    view = ev.eval_view("test")
    y, clusters = view["y"].to_numpy(), view["cluster"].to_numpy()
    probs = softmax(ev.logits_view("statickd", "test"))
    conf, pred_s = probs.max(1), probs.argmax(1)
    pred_t = ev.pred_view(teacher, "test")

    grid = []
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        keep = conf >= th
        pred = np.where(keep, pred_s, pred_t)
        grid.append({"theta": th, "statickd_share": float(keep.mean()), "macro_f1": ev.macro_f1(y, pred),
                     "accuracy": float((pred == y).mean())})

    keep = conf >= theta
    pred_c = np.where(keep, pred_s, pred_t)
    # paired_bootstrap_seeds with one prediction per side = paired_bootstrap, but it also returns the diffs
    pb = ev.paired_bootstrap_seeds(y, [pred_c], [pred_t], clusters=clusters, n_boot=N_BOOT_PAIRED, seed=0)
    eq = ev.tost(pb.pop("diffs"), MARGIN)
    res = {"theta": theta, "teacher": teacher, "n_docs": int(len(y)), "bootstrap": "paired site-cluster",
           "n_boot": N_BOOT_PAIRED, "seed": 0, "statickd_share": float(keep.mean()),
           "routed_share": float(1 - keep.mean()),
           "cascade": {"macro_f1": ev.macro_f1(y, pred_c), "accuracy": float((pred_c == y).mean())},
           "teacher_alone": {"macro_f1": ev.macro_f1(y, pred_t), "accuracy": float((pred_t == y).mean())},
           "delta_macro_f1": pb, "tost_margin": MARGIN, "tost": eq, "grid_sanity_check": grid}
    (RES / "e2_cascade_test_accuracy.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
