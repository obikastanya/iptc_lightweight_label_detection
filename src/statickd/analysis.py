"""Model registry and the analysis helpers used by the notebook to build every table of the paper."""
from __future__ import annotations

import numpy as np
import pandas as pd

from statickd.config import MODELS_DIR
from statickd.evaluation import (bootstrap_ci, eval_set, eval_view, grouped_macro_f1, has_predictions, load_pred,
                                 macro_f1, pred_view, scores)
from statickd.experiments import ABLATIONS, run_name

FINAL_MODELS_DIR = MODELS_DIR / "final"
BASELINES_DIR = MODELS_DIR / "baselines"

# key -> display name, benchmark spec (None = accuracy only)
MODELS = {
    "teacher":           ("Teacher XLM-R-large", "teacher-pytorch"),
    "teacher_onnx_int8": ("Teacher XLM-R-large, ONNX int8", "teacher-onnx-int8"),
    "distilmbert_kd":    ("DistilmBERT + KD", f"onnx:{BASELINES_DIR / 'distilmbert_kd'}"),
    "minilm_kd":         ("MiniLM-L6 + KD", f"onnx:{BASELINES_DIR / 'minilm_kd'}"),
    "statickd":          ("StaticKD (usulan)", f"static:{FINAL_MODELS_DIR / 'statickd'}"),
    "statickd_compact":  ("StaticKD-compact", f"static:{FINAL_MODELS_DIR / 'statickd_compact'}"),
    "ft_kd_subword_q":   ("fastText + KD, subword, kuantisasi", f"fasttext:{BASELINES_DIR / 'ft_kd_subword_q.ftz'}"),
    "ft_kd_subword":     ("fastText + KD, subword", f"fasttext:{BASELINES_DIR / 'ft_kd_subword.bin'}"),
    "ft_kd_word":        ("fastText + KD, kata", f"fasttext:{BASELINES_DIR / 'ft_kd_word.bin'}"),
    "tfidf_svm_kd":      ("TF-IDF + SVM + KD", f"tfidf:{BASELINES_DIR / 'tfidf_svm_kd.joblib'}"),
    "ft_gpt_subword":    ("fastText, label GPT-4o, subword", f"fasttext:{BASELINES_DIR / 'ft_gpt_subword.bin'}"),
    "ft_gpt_word":       ("fastText, label GPT-4o, kata", f"fasttext:{BASELINES_DIR / 'ft_gpt_word.bin'}"),
}

REGION = {
    **{l: "Eropa" for l in "en es ru tr it fr pt de el pl uk ro nl bg ca cs".split()},
    **{l: "Asia Selatan" for l in "mr hi ta pa or te kn bn gu ml".split()},
    **{l: "Afrika" for l in "ha rn sw ig lg so yo xh ln am sn pcm ti om".split()},
    **{l: "Asia Timur/Tenggara & Timur Tengah" for l in "vi id ar zh fa ko th".split()},
}
SCRIPT = {
    **{l: "Latin" for l in "en es tr it fr pt de pl ro nl ca cs ha rn sw ig lg so yo xh ln sn pcm om vi id".split()},
    **{l: "Sirilik/Yunani" for l in "ru uk bg el".split()},
    **{l: "Arab" for l in "ar fa".split()},
    **{l: "Brahmi (India)" for l in "hi mr bn pa gu or ta te kn ml".split()},
    **{l: "Ge'ez" for l in "am ti".split()},
    **{l: "CJK/Thai" for l in "zh ko th".split()},
}
# the 100 languages of XLM-R pre-training (Conneau et al., 2020), i.e. of the teacher's backbone
XLMR_LANGS = set("""af am ar as az be bg bn br bs ca cs cy da de el en eo es et eu fa fi fr fy ga gd gl gu ha he hi hr hu
hy id is it ja jv ka kk km kn ko ku ky la lo lt lv mg mk ml mn mr ms my ne nl no om or pa pl ps pt ro ru sa sd si sk
sl so sq sr su sv sw ta te th tl tr ug uk ur uz vi xh yi zh""".split())
ORIGIN_NAMES = {"human_annotation": "anotasi manusia", "publisher_category": "kategori penerbit",
                "publisher_section_jsonld": "rubrik JSON-LD"}


def available(models) -> list[str]:
    return [m for m in models if has_predictions(m, "test")]


def score_row(model: str, set_name: str, clusters: bool = True, n_boot: int = 1000) -> dict:
    df = eval_view(set_name)
    y, p = df["y"].to_numpy(), pred_view(model, set_name)
    # dev has no site information (one source): document bootstrap there, site-cluster bootstrap elsewhere
    use_clusters = clusters and set_name != "dev"
    lo, hi = bootstrap_ci(y, p, clusters=df["cluster"].to_numpy() if use_clusters else None, n_boot=n_boot)
    row = {"model": model, "set": set_name, **scores(y, p), "ci_low": lo, "ci_high": hi}
    if set_name == "test":
        row["lang_macro_f1"] = grouped_macro_f1(y, p, df["lang"].to_numpy())
    return row


def seed_table(codes=None, sets=("dev", "heldout", "test")) -> pd.DataFrame:
    """Scores of every trained StaticKD run (one row per code and seed)."""
    rows = []
    for code, (desc, _, seeds) in ABLATIONS.items():
        if codes and code not in codes:
            continue
        for seed in seeds:
            name = run_name(code, seed)
            if not all(has_predictions(name, s) for s in sets):
                continue
            row = {"code": code, "description": desc, "seed": seed}
            for s in sets:
                y, p = eval_view(s)["y"].to_numpy(), pred_view(name, s)
                row[f"{s}_macro"], row[f"{s}_acc"] = macro_f1(y, p), float((y == p).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def language_info(test: pd.DataFrame, pool_langs: pd.Series) -> pd.DataFrame:
    """Per test language: region, script, #distillation docs and a data tier."""
    rows = []
    for lang, g in test.groupby("lang"):
        n_distil = int(pool_langs.get(lang, 0))
        share = g["cluster"].value_counts(normalize=True)
        eff = 1.0 / float((share ** 2).sum())
        rows.append({"lang": lang, "region": REGION.get(lang, "?"), "script": SCRIPT.get(lang, "?"),
                     "n_test": len(g), "n_labels": g["label"].nunique(), "distil_docs": n_distil,
                     "distil_tier": "tidak ada (0)" if n_distil == 0 else ("sedikit (<500)" if n_distil < 500 else "cukup (≥500)"),
                     "xlmr": "tercakup pra-latih XLM-R" if lang in XLMR_LANGS else "di luar pra-latih XLM-R",
                     "clusters": len(share), "effective_clusters": eff, "top_cluster_share": float(share.iloc[0]),
                     "diversity_tier": "beragam (≥10)" if eff >= 10 else ("sedang (3–10)" if eff >= 3 else "satu penerbit (<3)")})
    return pd.DataFrame(rows).set_index("lang")


def subgroup_scores(models, column: str, test: pd.DataFrame | None = None, min_n: int = 1) -> pd.DataFrame:
    """Macro-F1 (present labels) and accuracy of each model within each value of a test column."""
    test = eval_set("test") if test is None else test
    rows = []
    for value, g in test.groupby(column):
        if len(g) < min_n:
            continue
        idx = g.index.to_numpy()
        y = g["y"].to_numpy()
        for m in models:
            p = load_pred(m, "test")[idx]
            rows.append({column: value, "model": m, "n": len(g), "macro_f1": macro_f1(y, p),
                         "accuracy": float((y == p).mean())})
    return pd.DataFrame(rows)


def cap_per_site(test: pd.DataFrame, per_site: int = 5, seed: int = 0) -> np.ndarray:
    """Indices of the sensitivity subset: at most `per_site` articles per site in each (lang, label) cell
    (only for sites with known domains, as the per-site cap in the first version of the test set)."""
    shuffled = test.sample(frac=1.0, random_state=seed)
    known = shuffled["domain"].ne("")
    capped = shuffled[known].groupby(["lang", "label", "domain"], group_keys=False).head(per_site)
    return np.sort(np.concatenate([capped.index.to_numpy(), shuffled[~known].index.to_numpy()]))
