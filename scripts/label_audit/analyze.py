"""Step 8: analysis of the article-level label audit (protocol: results/label_audit/protocol.md).

Reads the audit files in results/label_audit/, treats empty human decisions in verify_human.csv as
pending, and writes results.json and report.md. Re-run after the human fills verify_human.csv:
    python scripts/label_audit/analyze.py
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import beta

from common import LABELS, OUT, STRATA, TEST_PATH, AUDIT_ORIGINS, ROOT, read_jsonl

N_BOOT, SEED_BOOT = 2000, 20261122
MODELS = {"teacher": "teacher", "StaticKD (final)": "statickd", "Model2Vec classifier + KD": "model2vec_clf",
          "MiniLM-L6 + KD": "minilm_kd_t1", "DistilmBERT + KD": "distilmbert_kd",
          "TF-IDF + SVM + KD": "tfidf_svm_kd", "fastText + KD": "ft_kd_subword"}


def clopper_pearson(k: int, n: int, a: float = 0.05):
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return [float(lo), float(hi)]


# ----------------------------------------------------------------------------- per-item verdicts
def build_items() -> pd.DataFrame:
    key = pd.read_csv(OUT / "sample_key.csv", dtype=str)
    a = pd.DataFrame(read_jsonl(OUT / "llm_labels_A.jsonl")).add_prefix("A_").rename(columns={"A_item_id": "item_id"})
    df = key.merge(a, on="item_id").sort_values("item_id").reset_index(drop=True)
    df["agree"] = (df["A_primary"] == df["label"]) & ~df["A_unusable"].astype(bool)
    df["lenient"] = (df["A_primary"] == df["label"]) | (df["A_secondary"] == df["label"])

    adj = pd.DataFrame(read_jsonl(OUT / "adjudication.jsonl")).add_prefix("adj_").rename(columns={"adj_item_id": "item_id"})
    akey = pd.read_csv(OUT / "adjudication_key.csv", dtype=str).fillna("")
    adj = adj.merge(akey, on="item_id")

    def adj_outcome(r):
        d = r["adj_decision"]
        if d in ("option_1", "option_2"):
            origin = r[f"{d}_origin"]
            return {"section": "section_wins", "llm_A": "llm_wins", "both": "section_wins"}[origin]
        return {"both_acceptable": "both_acceptable", "neither": "neither", "unusable": "unusable"}[d]

    adj["adj_outcome"] = adj.apply(adj_outcome, axis=1)
    df = df.merge(adj, on="item_id", how="left")

    b = pd.DataFrame(read_jsonl(OUT / "llm_labels_B.jsonl")).add_prefix("B_").rename(columns={"B_item_id": "item_id"})
    df = df.merge(b, on="item_id", how="left")
    df["spot"] = df["B_primary"].notna()
    df["B_confirm"] = np.where(df["spot"], (df["B_primary"] == df["label"]) | (df["B_secondary"] == df["label"]), np.nan)

    hum = pd.read_csv(OUT / "verify_human.csv", dtype=str, encoding="utf-8-sig").fillna("")
    hkey = pd.read_csv(OUT / "human_options_key.csv", dtype=str)
    hum = hum.merge(hkey, on="item_id", how="left")
    route = pd.read_csv(OUT / "human_routing.csv", dtype=str)
    hum = hum.merge(route[["item_id", "reason"]], on="item_id")
    hum = hum[["item_id", "task", "reason", "option_A", "option_B", "option_A_origin", "option_B_origin",
               "decision", "other_label"]].rename(columns={"decision": "h_decision", "other_label": "h_other"})
    df = df.merge(hum, on="item_id", how="left")
    df["routed"] = df["task"].notna()

    out = df.apply(verdict, axis=1, result_type="expand")
    df["status"], df["verified"], df["status_prov"], df["verified_prov"] = out[0], out[1], out[2], out[3]
    return df


def _human(r):
    """Map a human decision to (status, verified label set). status None = pending."""
    d, o = r["h_decision"].strip(), r["h_other"].strip().lower()
    if not d:
        return None, None
    if d.lower() == "unusable":
        return "unusable", None
    if r["task"] == "choose_between":
        d = d.upper()
        if d in ("A", "B"):
            origin = r[f"option_{d}_origin"]
            lab = r[f"option_{d}"]
            return ("correct" if origin == "section" else "wrong"), [lab]
        if d == "BOTH":
            return "both_acceptable", [r["option_A"], r["option_B"]]
        if d == "NEITHER":
            return "wrong", ([o] if o in LABELS else None)
        raise ValueError(f"{r['item_id']}: unknown decision {d!r}")
    d = d.lower()
    if d not in LABELS:
        raise ValueError(f"{r['item_id']}: label {d!r} not one of the 17 labels")
    labs = [d] + ([o] if o in LABELS and o != d else [])
    if d == r["label"]:
        return "correct", labs
    if o == r["label"]:
        return ("correct" if r["reason"] == "spotcheck" else "both_acceptable"), labs
    return "wrong", labs


def _adjudicator(r):
    oc = r["adj_outcome"]
    if oc == "section_wins":
        return "correct", [r["label"]]
    if oc == "llm_wins":
        return "wrong", [r["A_primary"]]
    if oc == "both_acceptable":
        return "both_acceptable", [r["label"], r["A_primary"]]
    if oc == "neither":
        return "wrong", [r["adj_correct_label"]]
    return "unusable", None


def verdict(r):
    """(status, verified labels) final and provisional (provisional: the LLM verdict stands in for a
    pending human decision: B's label for spot-check items, the adjudicator for adjudicated items)."""
    if r["agree"]:
        if not r["spot"] or r["B_confirm"]:
            return "correct", [r["label"]], "correct", [r["label"]]
        s, v = _human(r)
        bl = [x for x in (r["B_primary"], r["B_secondary"]) if isinstance(x, str)]
        prov = (s, v) if s else ("wrong", bl)
        return s or "pending", v, prov[0], prov[1]
    if not r["routed"]:
        s, v = _adjudicator(r)
        return s, v, s, v
    s, v = _human(r)
    ps, pv = (s, v) if s else _adjudicator(r)
    return s or "pending", v, ps, pv


# ----------------------------------------------------------------------------- estimates
def estimate(df: pd.DataFrame, status_col: str, pooled_cond_fallback: bool = True) -> dict:
    """Weighted editorial-label accuracy (strict, lenient) following protocol Section 10.4."""
    spot = df[df["spot"]]
    sp_res = spot[spot[status_col].isin(["correct", "wrong", "both_acceptable"])]
    e = float((sp_res[status_col] == "wrong").mean()) if len(sp_res) else 0.0
    dis_all = df[~df["agree"]]
    res_all = dis_all[dis_all[status_col].isin(["correct", "wrong", "both_acceptable"])]
    pooled = ((res_all[status_col] == "correct").mean() if len(res_all) else np.nan,
              res_all[status_col].isin(["correct", "both_acceptable"]).mean() if len(res_all) else np.nan)
    per, tot_w = {}, sum(POP.values())
    strict = lenient = 0.0
    for s in STRATA:
        d = df[df["source"] == s]
        if not len(d):
            continue
        pa = d["agree"].mean()
        dis = d[~d["agree"]]
        res = dis[dis[status_col].isin(["correct", "wrong", "both_acceptable"])]
        if len(res):
            cs, cl, fb = (res[status_col] == "correct").mean(), res[status_col].isin(["correct", "both_acceptable"]).mean(), False
        else:
            cs, cl, fb = pooled[0], pooled[1], True
        if len(dis) == 0:
            cs = cl = 0.0
        acc_s, acc_l = pa * (1 - e) + (1 - pa) * cs, pa * (1 - e) + (1 - pa) * cl
        per[s] = {"p_agree": float(pa), "p_section_correct_given_disagree_strict": float(cs),
                  "p_section_correct_given_disagree_lenient": float(cl), "n_resolved_disagree": int(len(res)),
                  "pooled_fallback": fb, "acc_strict": float(acc_s), "acc_lenient": float(acc_l)}
        strict += POP[s] / tot_w * acc_s
        lenient += POP[s] / tot_w * acc_l
    return {"spot_error": e, "strata": per, "acc_strict": float(strict), "acc_lenient": float(lenient)}


def bootstrap(df: pd.DataFrame, status_col: str) -> dict:
    rng = np.random.default_rng(SEED_BOOT)
    groups = {s: df[df["source"] == s] for s in STRATA}
    st, le = [], []
    for _ in range(N_BOOT):
        parts = [g.iloc[rng.integers(0, len(g), len(g))] for g in groups.values() if len(g)]
        r = estimate(pd.concat(parts), status_col)
        st.append(r["acc_strict"])
        le.append(r["acc_lenient"])
    q = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
    return {"acc_strict_ci": q(st), "acc_lenient_ci": q(le)}


def model_check(df: pd.DataFrame) -> dict:
    test = pd.read_json(TEST_PATH, lines=True, dtype={"domain": str, "url": str})
    pos = pd.Series(np.arange(len(test)), index=test["doc_id"].astype(str))
    rows = pos.loc[df["doc_id"].astype(str)].to_numpy()
    y_sec = df["label"].map(LABELS.index).to_numpy()
    out = {}
    for name, f in MODELS.items():
        logits = np.load(ROOT / "results" / "final" / "predictions" / f"{f}__test.npz")["logits"]
        if logits.shape[0] != len(test):
            raise SystemExit(f"{f}: row count mismatch")
        pred = logits[rows].argmax(1)
        lab = [LABELS[i] for i in pred]
        rec = {"acc_vs_section": float((pred == y_sec).mean()), "n_section": int(len(pred))}
        for col, tag in (("verified", "verified"), ("verified_prov", "provisional")):
            ok = [p in v for p, v in zip(lab, df[col]) if isinstance(v, list) and v]
            rec[f"acc_vs_{tag}"] = float(np.mean(ok)) if ok else None
            rec[f"n_{tag}"] = len(ok)
        sec_sub = [p == s for p, s, v in zip(lab, df["label"], df["verified"]) if isinstance(v, list) and v]
        rec["acc_vs_section_same_items_as_verified"] = float(np.mean(sec_sub)) if sec_sub else None
        out[name] = rec
    return out


# ----------------------------------------------------------------------------- main
def main():
    global POP
    full = pd.read_json(TEST_PATH, lines=True, dtype={"domain": str, "url": str})
    POP = full[full["label_origin"].isin(AUDIT_ORIGINS)]["source"].value_counts().to_dict()
    df = build_items()
    n_pending = int((df["status"] == "pending").sum())

    by = df.groupby("source")
    ann = {s: {"n": int(len(g)), "agree_strict": int(g["agree"].sum()), "agree_lenient": int(g["lenient"].sum()),
               "A_unusable": int(g["A_unusable"].astype(bool).sum()), "disagree": int((~g["agree"]).sum())}
           for s, g in by}
    ann["all"] = {"n": len(df), "agree_strict": int(df["agree"].sum()), "agree_lenient": int(df["lenient"].sum()),
                  "A_unusable": int(df["A_unusable"].astype(bool).sum()), "disagree": int((~df["agree"]).sum())}
    for v in ann.values():
        v["rate_strict"], v["rate_lenient"] = v["agree_strict"] / v["n"], v["agree_lenient"] / v["n"]

    dis = df[~df["agree"]]
    adj_counts = dis["adj_outcome"].value_counts().to_dict()
    adj_conf = pd.crosstab(dis["adj_outcome"], dis["adj_confidence"]).to_dict(orient="index")
    final_dis = dis["status"].value_counts().to_dict()
    final_dis_by = {s: g["status"].value_counts().to_dict() for s, g in dis.groupby("source")}

    spot = df[df["spot"]]
    k = int(spot["B_confirm"].astype(bool).sum())
    sp_res = spot[spot["status"].isin(["correct", "wrong", "both_acceptable"])]
    k_err = int((sp_res["status"] == "wrong").sum())
    spot_res = {"n": int(len(spot)), "by_source": spot["source"].value_counts().to_dict(),
                "B_confirmed": k, "B_confirm_rate": k / len(spot), "B_confirm_ci": clopper_pearson(k, len(spot)),
                "resolved": int(len(sp_res)), "pending": int((spot["status"] == "pending").sum()),
                "error_after_human": k_err, "error_rate": (k_err / len(sp_res)) if len(sp_res) else None,
                "error_ci": clopper_pearson(k_err, len(sp_res))}

    est = estimate(df, "status")
    est.update(bootstrap(df, "status"))
    prov = estimate(df, "status_prov")
    prov.update(bootstrap(df, "status_prov"))
    models = model_check(df)
    routed = df[df["routed"]]
    results = {"date": str(date.today()), "n_items": len(df), "population": POP,
               "human_pending": n_pending, "human_routed": int(len(routed)),
               "human_routed_by_reason": routed["reason"].value_counts().to_dict(),
               "annotator_A_vs_section": ann, "adjudicator_outcomes": adj_counts,
               "adjudicator_outcome_x_confidence": adj_conf, "disagree_final_status": final_dis,
               "disagree_final_status_by_source": final_dis_by, "spot_check": spot_res,
               "estimate_interim": est, "estimate_provisional_llm_for_pending": prov, "model_check": models}
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    item_cols = ["item_id", "source", "lang", "agree", "lenient", "spot", "routed", "status", "status_prov"]
    df[item_cols].to_csv(OUT / "item_status.csv", index=False, encoding="utf-8")
    write_report(results)
    print(f"wrote results.json, report.md, item_status.csv | pending human items: {n_pending}")


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def ci(c):
    return "n/a" if c is None or c[0] is None else f"{100 * c[0]:.1f}-{100 * c[1]:.1f}%"


def write_report(R: dict) -> None:
    L = []
    interim = R["human_pending"] > 0
    L.append("# Article-level audit of editorial labels" + (" (INTERIM)" if interim else ""))
    L.append("")
    L.append(f"Generated {R['date']} by `scripts/label_audit/analyze.py`. Protocol: `protocol.md`.")
    if interim:
        L.append(f"**{R['human_pending']} of {R['human_routed']} human-routed items are still pending** "
                 "(`verify_human.csv`). Pending items are excluded from the interim estimate; a second, "
                 "provisional estimate uses the LLM verdict (adjudicator or Annotator B) in their place.")
    L.append("")
    L.append(f"Sample: {R['n_items']} articles, stratified by source; population "
             + ", ".join(f"{k} {v:,}" for k, v in R["population"].items()) + ".")
    L.append("")
    L.append("## 1. Annotator A (blind LLM) vs editorial label")
    L.append("")
    L.append("| stratum | n | agree (strict) | agree (lenient) | A unusable | disagree |")
    L.append("|---|---|---|---|---|---|")
    for s, v in R["annotator_A_vs_section"].items():
        L.append(f"| {s} | {v['n']} | {v['agree_strict']} ({pct(v['rate_strict'])}) | {v['agree_lenient']} "
                 f"({pct(v['rate_lenient'])}) | {v['A_unusable']} | {v['disagree']} |")
    L.append("")
    L.append("## 2. Adjudication of disagreements (blind LLM adjudicator)")
    L.append("")
    L.append("| outcome | n | high | medium | low |")
    L.append("|---|---|---|---|---|")
    for oc, n in sorted(R["adjudicator_outcomes"].items(), key=lambda x: -x[1]):
        c = R["adjudicator_outcome_x_confidence"].get(oc, {})
        L.append(f"| {oc} | {n} | {c.get('high', 0)} | {c.get('medium', 0)} | {c.get('low', 0)} |")
    L.append("")
    L.append("Final status of disagree items (human decision where routed; pending = not yet decided): "
             + ", ".join(f"{k} {v}" for k, v in R["disagree_final_status"].items()) + ".")
    L.append("")
    sp = R["spot_check"]
    L.append("## 3. Spot-check of agreed items (blind Annotator B)")
    L.append("")
    L.append(f"{sp['n']} agreed items ({', '.join(f'{k} {v}' for k, v in sp['by_source'].items())}). "
             f"B confirmed {sp['B_confirmed']}/{sp['n']} = {pct(sp['B_confirm_rate'])} "
             f"(Clopper-Pearson 95% CI {ci(sp['B_confirm_ci'])}). After human review: "
             f"{sp['error_after_human']} errors in {sp['resolved']} resolved items "
             f"(error rate {pct(sp['error_rate'])}, CI {ci(sp['error_ci'])}); {sp['pending']} pending.")
    L.append("")
    L.append("## 4. Estimated editorial-label accuracy (population-weighted)")
    L.append("")
    L.append("acc_s = P_s(agree)(1 - e) + P_s(disagree) P_s(section correct | disagree); strict = section label "
             "is the main topic; lenient also counts both_acceptable. 95% CI: stratified bootstrap, 2,000 resamples.")
    L.append("")
    L.append("| estimate | strict | 95% CI | lenient | 95% CI | spot-check error e |")
    L.append("|---|---|---|---|---|---|")
    for tag, k in (("interim (pending excluded)", "estimate_interim"),
                   ("provisional (LLM verdict for pending)", "estimate_provisional_llm_for_pending")):
        e = R[k]
        L.append(f"| {tag} | {pct(e['acc_strict'])} | {ci(e['acc_strict_ci'])} | {pct(e['acc_lenient'])} | "
                 f"{ci(e['acc_lenient_ci'])} | {pct(e['spot_error'])} |")
    L.append("")
    L.append("Per stratum (interim):")
    L.append("")
    L.append("| stratum | P(agree) | P(correct given disagree) strict / lenient | resolved disagree | acc strict | acc lenient |")
    L.append("|---|---|---|---|---|---|")
    for s, v in R["estimate_interim"]["strata"].items():
        fb = " (pooled fallback)" if v["pooled_fallback"] else ""
        L.append(f"| {s} | {pct(v['p_agree'])} | {pct(v['p_section_correct_given_disagree_strict'])} / "
                 f"{pct(v['p_section_correct_given_disagree_lenient'])}{fb} | {v['n_resolved_disagree']} | "
                 f"{pct(v['acc_strict'])} | {pct(v['acc_lenient'])} |")
    L.append("")
    L.append("## 5. Descriptive model check on the audited items")
    L.append("")
    L.append("Accuracy (argmax of stored test logits) against the editorial label on all audited items, against "
             "the editorial label on the verified subset, and against the verified label (both_acceptable: either "
             "label counts; unusable and pending excluded). Provisional = LLM verdicts for pending items.")
    L.append("")
    L.append("| model | vs section (all) | vs section (verified subset) | vs verified | n verified | vs provisional | n prov. |")
    L.append("|---|---|---|---|---|---|---|")
    for m, v in R["model_check"].items():
        L.append(f"| {m} | {pct(v['acc_vs_section'])} | {pct(v['acc_vs_section_same_items_as_verified'])} | "
                 f"{pct(v['acc_vs_verified'])} | {v['n_verified']} | {pct(v['acc_vs_provisional'])} | {v['n_provisional']} |")
    L.append("")
    L.append("## Notes")
    L.append("")
    L.append("- All LLM roles (Annotator A, adjudicator, Annotator B, translator) were fresh subagents that saw only "
             "item_id, language, headline and text (plus two unlabelled candidates for the adjudicator). The "
             "agent tool log in `agent_tool_log.csv` shows each subagent read only its own batch file.")
    L.append("- Annotator A, the adjudicator and Annotator B are the same underlying LLM; their agreement is not "
             "independent human agreement.")
    L.append("- The agreed items outside the spot-check are assumed correct; the spot-check error rate e is pooled "
             "over strata.")
    if interim:
        L.append("- While items are pending, the interim estimate is biased: only high-confidence adjudications are "
                 "resolved, and these are mostly cases where the LLM label won. The provisional row is the better "
                 "interim guide; the final number needs the human decisions.")
    L.append("- The teacher was fine-tuned on GPT-4o labels, so LLM-verified labels may share its view of the "
             "taxonomy. The model check against verified labels can favour the teacher and its students.")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


POP: dict = {}

if __name__ == "__main__":
    main()
