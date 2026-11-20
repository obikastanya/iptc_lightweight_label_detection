"""E2 step 3 summary: median over sessions, session spread, and the formula estimate.

Formula (paper, Section 10.4): r_cascade = 1 / (1/r_S + (1-s)/r_T), with r_S, r_T the measured single-core
throughputs of StaticKD and the ONNX int8 teacher (component benchmarks of the same sessions, median over
sessions) and s the StaticKD share measured on the cascade sample.

Usage (from the project root):
    env/Scripts/python.exe scripts/review_experiments/e2d_summarize.py [--sessions 1 2 3]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "review_experiments"
MAX_DEV = 0.07


def spread(values: pd.Series) -> dict:
    med = float(values.median())
    return {"median": med, "per_session": [round(float(v), 4) for v in values],
            "max_dev_from_median": float(max(abs(values - med)) / med)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", nargs="*", type=int, default=None)
    args = parser.parse_args()
    casc = pd.read_json(RES / "cascade_benchmark.jsonl", lines=True).drop_duplicates("session", keep="last")
    comp = pd.read_json(RES / "cascade_components.jsonl", lines=True).drop_duplicates(["model", "run"], keep="last")
    if args.sessions:
        casc, comp = casc[casc.session.isin(args.sessions)], comp[comp.run.isin(args.sessions)]
    casc, comp = casc.sort_values("session"), comp.sort_values("run")
    r_s = spread(comp[comp.model == "statickd"].docs_per_sec)
    r_t = spread(comp[comp.model == "teacher_onnx_int8"].docs_per_sec)
    e2e = spread(casc.docs_per_sec)
    routed = float(casc.routed_share.iloc[0])
    assert casc.routed_share.nunique() == 1, "the sample is fixed, so the routed share must not change"

    def formula(rs, rt):
        return 1.0 / (1.0 / rs + routed / rt)

    est = formula(r_s["median"], r_t["median"])
    per_session_est = [formula(a, b) for a, b in zip(comp[comp.model == "statickd"].docs_per_sec,
                                                     comp[comp.model == "teacher_onnx_int8"].docs_per_sec)]
    # same formula with the teacher's speed on the routed articles themselves (from the cascade timers)
    rt_routed = float(casc.teacher_docs_per_sec_on_routed.median())
    est_routed = formula(r_s["median"], rt_routed)
    res = {"theta": float(casc.theta.iloc[0]), "sessions": casc.session.tolist(), "n_docs": int(casc.n_docs.iloc[0]),
           "routed_share": routed, "statickd_share": 1 - routed,
           "end_to_end_docs_per_sec": e2e, "r_statickd_docs_per_sec": r_s, "r_teacher_onnx_int8_docs_per_sec": r_t,
           "formula_estimate_docs_per_sec": est, "formula_per_session": per_session_est,
           "gap_measured_vs_formula": e2e["median"] / est - 1,
           "teacher_docs_per_sec_on_routed_in_cascade": spread(casc.teacher_docs_per_sec_on_routed),
           "formula_with_teacher_speed_on_routed": est_routed,
           "gap_measured_vs_formula_routed_speed": e2e["median"] / est_routed - 1,
           "time_share_teacher": float((casc.teacher_seconds / casc.seconds).median()),
           "max_dev_allowed": MAX_DEV,
           "spread_ok": all(x["max_dev_from_median"] <= MAX_DEV for x in (e2e, r_s, r_t)),
           "labels_same_as_stored_cascade": casc.labels_same_as_stored_cascade.tolist(),
           "routing_same_as_stored": casc.routing_same_as_stored.tolist()}
    (RES / "e2_cascade_benchmark_summary.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
