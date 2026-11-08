"""Extra figures for the Indonesian paper (paper_final_id/main_id.tex). Text inside the figures is English so the
same files can be reused when the Indonesian version is carried back into the English paper.

  fig_tradeoff_memory.pdf  macro-F1 against single-core throughput (a) and against model RAM (b)
  fig_testset_composition.pdf  test articles per language by label origin, grouped by distillation data

Inputs are the notebook outputs (results/final/paper_generated/tab_eff.tex, tab_main.tex) and the unified test set.
Run from the project root: env/Scripts/python.exe paper_final_id/make_figures.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

from statickd import analysis as an  # noqa: E402
from statickd import data, viz  # noqa: E402

GEN = ROOT / "results" / "final" / "paper_generated"
OUT = Path(__file__).resolve().parent / "figures"

# display name in the generated tables -> model key used by viz
NAME_TO_KEY = {
    "Teacher XLM-R-large": "teacher", "Teacher XLM-R-large, ONNX int8": "teacher_onnx_int8",
    "DistilmBERT + KD": "distilmbert_kd", "MiniLM-L6 + KD": "minilm_kd", "StaticKD": "statickd",
    "StaticKD-compact": "statickd_compact", "fastText + KD, subword, quantized": "ft_kd_subword_q",
    "fastText + KD, subword": "ft_kd_subword", "TF-IDF + SVM + KD": "tfidf_svm_kd",
    "Model2Vec classifier + KD": "model2vec_clf",
}
MARKERS = {"teacher": "s", "teacher_onnx_int8": "D", "distilmbert_kd": "^", "minilm_kd": "v", "statickd": "*",
           "statickd_compact": "P", "ft_kd_subword_q": "o", "ft_kd_subword": "X", "tfidf_svm_kd": "h", "model2vec_clf": "D"}


def num(s: str) -> float:
    return float(s.replace(",", "").strip())


def table_rows(name: str) -> dict:
    rows = {}
    for line in (GEN / name).read_text(encoding="utf8").splitlines():
        if "&" not in line or line.startswith("%"):
            continue
        cells = [re.sub(r"\\textbf\{([^}]*)\}", r"\1", c).strip() for c in line.rstrip("\\ ").split("&")]
        label = cells[0].replace(" (proposed)", "").strip()
        if label in NAME_TO_KEY:
            rows[NAME_TO_KEY[label]] = cells
    return rows


def fig_tradeoff_memory() -> None:
    eff = table_rows("tab_eff.tex")      # cols: name, latency, dps 1c, dps 4c, RAM MB, file MB, deps
    main = table_rows("tab_main.tex")    # cols: name, dev F1 [ci], dev acc, agreement, test F1 [ci], ...
    test_f1 = {k: float(re.match(r"[\d.]+", c[4]).group(0)) for k, c in main.items()}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), sharey=True)
    for key, c in eff.items():
        if key not in test_f1:
            continue
        kw = dict(s=150 if key == "statickd" else 55, marker=MARKERS[key], color=viz.color(key), zorder=3,
                  edgecolor="white", linewidth=0.6)
        if key == "model2vec_clf":   # open marker: it sits almost on top of StaticKD in panel (a)
            kw.update(facecolor="none", edgecolor=viz.color(key), linewidth=1.3, zorder=5)
            del kw["color"]
        axes[0].scatter(num(c[2]), test_f1[key], **kw)
        axes[1].scatter(num(c[4]), test_f1[key], **kw)
    axes[0].set_xlabel("throughput, 1 CPU core (documents/s, log)")
    axes[1].set_xlabel("model RAM (MB, log)")
    axes[0].set_ylabel("macro-F1, unified test set")
    for ax, title in zip(axes, ("(a) accuracy vs. speed", "(b) accuracy vs. memory")):
        ax.set_xscale("log")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xticks([300, 1000, 3000])
    handles = [Line2D([], [], ls="", marker=MARKERS[k], ms=9 if k == "statickd" else 6, color=viz.color(k),
                      markerfacecolor="none" if k == "model2vec_clf" else viz.color(k),
                      label=next(n for n, kk in NAME_TO_KEY.items() if kk == k)) for k in eff if k in test_f1]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.savefig(OUT / "fig_tradeoff_memory.pdf")
    plt.close(fig)


def fig_testset_composition() -> None:
    test = an.eval_set("test")
    pool_langs = data.load_distillation_pool()["lang"].value_counts()
    info = an.language_info(test, pool_langs)
    tiers = [("≥500 docs", lambda n: n >= 500), ("1–499 docs", lambda n: 0 < n < 500), ("none", lambda n: n == 0)]
    origins = [("human_annotation", "human annotation"), ("publisher_category", "publisher category"),
               ("publisher_section_jsonld", "JSON-LD section")]
    counts = test.groupby(["lang", "label_origin"]).size().unstack(fill_value=0)
    order, bounds = [], []
    for _, keep in tiers:
        langs = [l for l in counts.index if keep(int(info.loc[l, "distil_docs"]))]
        langs.sort(key=lambda l: -counts.loc[l].sum())
        bounds.append((len(order), len(order) + len(langs)))
        order += langs
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    bottom = [0] * len(order)
    for col, label in origins:
        vals = [int(counts.loc[l, col]) if col in counts.columns else 0 for l in order]
        ax.bar(range(len(order)), vals, bottom=bottom, color=viz.ORIGIN_COLORS[col], width=0.8, label=label)
        bottom = [b + v for b, v in zip(bottom, vals)]
    top = max(bottom)
    for (name, _), (a, b) in zip(tiers, bounds):
        if a:
            ax.axvline(a - 0.5, color="#555555", lw=0.8)
        ax.text((a + b - 1) / 2, top * 1.03, f"{name}\n({b - a} languages)", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=6.5)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(0, top * 1.22)
    ax.set_title("distillation data per language", fontsize=8, pad=2)
    ax.set_ylabel("test articles")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=7, frameon=False, bbox_to_anchor=(0.22, 0.86))
    fig.savefig(OUT / "fig_testset_composition.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    viz.apply_style()
    fig_tradeoff_memory()
    fig_testset_composition()
    print("written to", OUT)
