"""Train the non-transformer baselines on the SAME distillation pool as StaticKD and store predictions.

| name              | training labels                         | tokens          |
|-------------------|-----------------------------------------|-----------------|
| ft_gpt_word       | GPT-4o labels, EMMediaTopic train only  | words (+char n) |
| ft_gpt_subword    | GPT-4o labels, EMMediaTopic train only  | XLM-R subwords  |
| ft_kd_word        | teacher arg-max, 285,999 docs, 70 langs | words (+char n) |
| ft_kd_subword     | teacher arg-max, 285,999 docs, 70 langs | XLM-R subwords  |
| ft_kd_subword_q   | as above, quantised (.ftz)              | XLM-R subwords  |
| tfidf_svm_kd      | teacher arg-max, 285,999 docs, 70 langs | XLM-R subwords  |

The KD baselines use the configuration chosen on the dev set by scripts/tune_baselines.py
(results/final/baseline_grid_best.json; ft_kd_subword_q is the quantised ft_kd_subword configuration).
Without that file, and always for the GPT-4o references, the fixed defaults are used: fastText dim 100,
lr 0.5, 15 epochs (25 for the small GPT-4o set, 20k documents); TF-IDF + SVM C = 0.5.

Usage (from the project root):
    python scripts/train_baselines.py [--only ft_kd_subword tfidf_svm_kd] [--threads 8]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statickd.baselines import (BASELINES_DIR, FastTextPredictor, TfidfPredictor, train_fasttext,  # noqa: E402
                                train_tfidf)
from statickd.config import FINAL_DIR  # noqa: E402
from statickd.data import load_distillation_pool, teacher_probs  # noqa: E402
from statickd.evaluation import evaluate_model  # noqa: E402

BEST_FILE = FINAL_DIR / "baseline_grid_best.json"


def grid_choice(name: str) -> dict:
    """Hyper-parameters chosen on dev by tune_baselines.py ({} = keep the defaults)."""
    best = json.loads(BEST_FILE.read_text()) if BEST_FILE.exists() else {}
    choice = best.get("ft_kd_subword" if name == "ft_kd_subword_q" else name, {})
    return {k: v for k, v in choice.items() if k in ("lr", "dim", "epoch", "c")}

FASTTEXT = {  # name: (labels, tokens, quantize, epochs) -- 25 epochs for the small GPT-4o set (20k docs)
    "ft_gpt_word": ("gpt", "word", False, 25),
    "ft_gpt_subword": ("gpt", "subword", False, 25),
    "ft_kd_word": ("teacher", "word", False, 15),
    "ft_kd_subword": ("teacher", "subword", False, 15),
    "ft_kd_subword_q": ("teacher", "subword", True, 15),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=[*FASTTEXT, "tfidf_svm_kd"])
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    pool = load_distillation_pool()
    hard = teacher_probs(pool).argmax(1)
    emm = pool[pool["source"] == "emmediatopic"]
    for name in args.only:
        t0 = time.perf_counter()
        if name in FASTTEXT:
            labels, tokens, quantize, epochs = FASTTEXT[name]
            texts, y = (emm["text"].tolist(), emm["gpt_label_id"].to_numpy()) if labels == "gpt" \
                else (pool["text"].tolist(), hard)
            params = {"epoch": epochs, **(grid_choice(name) if labels == "teacher" else {})}
            print(f"{name}: {params}", flush=True)
            path = train_fasttext(texts, y, BASELINES_DIR / name, tokens=tokens, quantize=quantize,
                                  threads=args.threads, **params)
            predictor = FastTextPredictor(path)
        else:
            params = grid_choice(name)
            print(f"{name}: {params}", flush=True)
            path = train_tfidf(pool["text"].tolist(), hard, BASELINES_DIR / name, **params)
            predictor = TfidfPredictor(path)
        summary = evaluate_model(name, predictor, overwrite=True)
        line = " ".join(f"{s}={r['macro_f1']:.4f}/{r['accuracy']:.4f}" for s, r in summary.items())
        print(f"{name}: {line} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
