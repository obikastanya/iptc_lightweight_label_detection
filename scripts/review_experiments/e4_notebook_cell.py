# E4 snippet for src/notebook_build/part4.py (NOT inserted; proposal only).
# Place it as a new code(r"""...""") cell right after the cell that fills NUM from main_table
# (Section 8.1). It uses the names already defined there: ev, np, key, NUM, fmt, N_BOOT, T_REF.
# The resampling is the one of analysis.score_row: site-cluster bootstrap on test, document bootstrap
# on dev, N_BOOT resamples, seed 0; the ratio is taken inside each resample on the same documents.
CELL = r'''
# 95% bootstrap interval of the retention ratio macro-F1(StaticKD) / macro-F1(teacher)
for s in ("dev", "test"):
    view = ev.eval_view(s)
    y = view.y.to_numpy(); clusters = view.cluster.to_numpy() if s != "dev" else None
    p_s, p_t = ev.pred_view(key("statickd"), s), ev.pred_view(key("teacher"), s)
    ratios = np.array([ev.macro_f1(y[i], p_s[i]) / ev.macro_f1(y[i], p_t[i])
                       for i in ev.bootstrap_samples(len(y), clusters, N_BOOT, 0)])
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    NUM.add(f"statickd.{s}.pct_teacher.ci", f"{fmt(lo, 1, percent=True)}--{fmt(hi, 1, percent=True)}")
    print(f"{s}: retensi {ev.macro_f1(y, p_s) / ev.macro_f1(y, p_t):.1%} [{lo:.1%}, {hi:.1%}]")
'''
