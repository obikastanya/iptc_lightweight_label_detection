"""E4: 95% bootstrap interval of the retention ratio macro-F1(StaticKD) / macro-F1(teacher).

Same resampling as the main table (analysis.score_row): site-cluster bootstrap on the test set,
document bootstrap on dev, 1,000 resamples, seed 0. The ratio is computed inside every resample on
the same resampled documents for both models.

Usage (from the project root):
    env/Scripts/python.exe scripts/review_experiments/e4_retention_ci.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from statickd import evaluation as ev  # noqa: E402

N_BOOT = 1000
OUT = ROOT / "results" / "review_experiments" / "e4_retention_ci.json"


def retention_ci(set_name: str, student: str = "statickd", teacher: str = "teacher", n_boot: int = N_BOOT,
                 seed: int = 0) -> dict:
    view = ev.eval_view(set_name)
    y = view["y"].to_numpy()
    p_s, p_t = ev.pred_view(student, set_name), ev.pred_view(teacher, set_name)
    clusters = view["cluster"].to_numpy() if set_name != "dev" else None   # as in analysis.score_row
    f1_s, f1_t = ev.macro_f1(y, p_s), ev.macro_f1(y, p_t)
    ratios = np.array([ev.macro_f1(y[i], p_s[i]) / ev.macro_f1(y[i], p_t[i])
                       for i in ev.bootstrap_samples(len(y), clusters, n_boot, seed)])
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return {"set": set_name, "bootstrap": "site-cluster" if clusters is not None else "document",
            "n_boot": n_boot, "seed": seed, "n_docs": int(len(y)),
            "macro_f1_student": f1_s, "macro_f1_teacher": f1_t, "retention": f1_s / f1_t,
            "ci_low": float(lo), "ci_high": float(hi), "boot_median": float(np.median(ratios))}


def main() -> None:
    res = {s: retention_ci(s) for s in ("test", "dev")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf8")
    for s, r in res.items():
        print(f"{s:4s} StaticKD {r['macro_f1_student']:.4f} / teacher {r['macro_f1_teacher']:.4f} = "
              f"{r['retention']:.2%}  95% CI [{r['ci_low']:.2%}, {r['ci_high']:.2%}]  ({r['bootstrap']} bootstrap)")


if __name__ == "__main__":
    main()
