"""One print-friendly matplotlib style for every figure of the paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from statickd.config import ROOT

FIG_DIR = ROOT / "results" / "final" / "figures"   # build_paper.py copies the figures the paper uses

# fixed colour per model family (colour-blind safe, Okabe-Ito)
COLORS = {
    "teacher": "#000000",
    "teacher_onnx_int8": "#555555",
    "distilmbert": "#CC79A7",
    "minilm": "#E69F00",
    "statickd": "#0072B2",
    "statickd_compact": "#56B4E9",
    "fasttext": "#D55E00",
    "tfidf": "#009E73",
    "model2vec": "#882255",
    "other": "#999999",
}
ORIGIN_COLORS = {"human_annotation": "#0072B2", "publisher_category": "#E69F00",
                 "publisher_section_jsonld": "#009E73"}


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 9,
        "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "pdf.fonttype": 42,
        "font.family": "DejaVu Sans",
    })


def family(model: str) -> str:
    for key in ("teacher_onnx_int8", "teacher", "distilmbert", "minilm", "compact", "statickd", "full",
                "ft_", "tfidf", "model2vec"):
        if key in model:
            return {"compact": "statickd_compact", "full": "statickd", "ft_": "fasttext"}.get(key, key)
    return "other"


def color(model: str) -> str:
    return COLORS[family(model)]


def forest(ax, labels, delta, ci95, ci90, significant, margin: float, xlabel: str, breaks=(), xmin=None) -> None:
    """Forest plot of paired differences, top row first: thin line = 95% interval, thick bar = 90% interval
    (TOST), filled marker = significant after Holm, shaded band = +-margin equivalence region.
    One hue only, so significance is read from the marker fill and survives grayscale printing.
    `breaks` are row indices above which a thin separator line is drawn (group boundaries).
    With `xmin`, a point left of it is drawn as an arrow at the axis edge with its value printed."""
    ink = COLORS["statickd"]
    y = list(range(len(labels)))[::-1]
    ax.axvspan(-margin, margin, color="#999999", alpha=0.22, lw=0, zorder=0)
    ax.axvline(0, color="#555555", lw=0.8, zorder=1)
    for yi, d, (l95, h95), (l90, h90), sig in zip(y, delta, ci95, ci90, significant):
        lo95, lo90 = (max(l95, xmin), max(l90, xmin)) if xmin is not None else (l95, l90)
        ax.plot([lo95, h95], [yi, yi], color=ink, lw=1.0, solid_capstyle="butt", zorder=2)
        ax.plot([lo90, h90], [yi, yi], color=ink, lw=3.2, solid_capstyle="butt", alpha=0.55, zorder=2)
        if xmin is not None and d < xmin:
            ax.plot(xmin, yi, "<", ms=6, mec=ink, mfc=ink if sig else "white", clip_on=False, zorder=3)
            ax.annotate(f"{d:+.3f}".replace("-", "−"), (xmin, yi), xytext=(4, 4), textcoords="offset points",
                        fontsize=6.5, color="#333333", va="bottom")
        else:
            ax.plot(d, yi, "o", ms=5, mec=ink, mew=1.2, mfc=ink if sig else "white", zorder=3)
    if xmin is not None:
        ax.set_xlim(left=xmin - 0.004)   # room so the arrow does not touch the tick labels
    for b in breaks:
        ax.axhline(len(labels) - b - 0.5, color="#bbbbbb", lw=0.6, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=ink, lw=1.0, label="95% interval"),
               Line2D([], [], color=ink, lw=3.2, alpha=0.55, label="90% interval"),
               Line2D([], [], ls="", marker="o", ms=5, mec=ink, mfc=ink, label="Holm $p<0.05$"),
               Line2D([], [], ls="", marker="o", ms=5, mec=ink, mfc="white", label="not significant"),
               Patch(color="#999999", alpha=0.22, label=f"$\\pm${margin:g} margin")]
    ax.legend(handles=handles, fontsize=6.5, ncol=3, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.0)


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    fig.savefig(FIG_DIR / f"{name}.png")
    return path


def show_and_save(fig, name: str) -> None:
    save(fig, name)
    plt.show()
