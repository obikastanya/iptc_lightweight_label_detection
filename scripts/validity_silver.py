"""Same-language check of the silver labels: English CC-NEWS articles with JSON-LD section labels that are
NOT in the test set (English test cells are filled by human-labelled sources) are scored by the teacher and
the students. Their per-label accuracy is compared with that on the human-labelled English test articles.

Output: data/processed/testset_jsonld/validity_en_silver.jsonl and predictions "<model>__valid_en_silver".

Usage (from the project root):
    python scripts/validity_silver.py [--per-label 50] [--device cuda]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from statickd.config import LABEL2ID  # noqa: E402
from statickd.data import TESTSET_DIR, load_unified_test  # noqa: E402
from statickd.evaluation import save_predictions  # noqa: E402
from statickd.student import StaticKDModel  # noqa: E402

OUT = TESTSET_DIR / "validity_en_silver.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-label", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    pool = pd.read_parquet(TESTSET_DIR / "unified_pool.parquet")
    test = load_unified_test()
    cand = pool[(pool["lang"] == "en") & (pool["label_origin"] == "publisher_section_jsonld")
                & ~pool["url"].isin(set(test["url"].dropna()))]
    sample = (cand.sample(frac=1.0, random_state=0).groupby("label", group_keys=False).head(args.per_label)
              .reset_index(drop=True))
    sample["label_id"] = sample["label"].map(LABEL2ID)
    sample.to_json(OUT, orient="records", lines=True, force_ascii=False)
    print(f"{len(sample)} English silver articles from {sample['domain'].nunique()} sites, "
          f"{sample['label'].nunique()} labels -> {OUT}", flush=True)
    texts = sample["text"].tolist()
    for name, path in (("statickd", "models/final/statickd"),):
        save_predictions(name, "valid_en_silver", StaticKDModel(path).logits(texts))
    from statickd.teacher import Teacher
    teacher = Teacher(device=args.device)
    save_predictions("teacher", "valid_en_silver", teacher.logits(texts, batch_size=args.batch_size))
    print("done", flush=True)


if __name__ == "__main__":
    main()
