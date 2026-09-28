"""One print-friendly matplotlib style for every figure of the paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from statickd.config import ROOT

FIG_DIR = ROOT / "jurnal_latex" / "figures"

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
                "ft_", "tfidf"):
        if key in model:
            return {"compact": "statickd_compact", "full": "statickd", "ft_": "fasttext"}.get(key, key)
    return "other"


def color(model: str) -> str:
    return COLORS[family(model)]


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    fig.savefig(FIG_DIR / f"{name}.png")
    return path


def show_and_save(fig, name: str) -> None:
    save(fig, name)
    plt.show()
