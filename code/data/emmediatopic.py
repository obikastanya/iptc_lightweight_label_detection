"""Loader for the EMMediaTopic 1.0 dataset (Kuzman & Ljubesic, 2025; CLARIN.SI 11356/1991)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EMMEDIATOPIC_PATH, LABEL2ID, truncate_words  # noqa: E402


def load_emmediatopic(split: str | None = None) -> pd.DataFrame:
    """Return EMMediaTopic with columns: document_id, lang, text, label, label_id, split.

    `label` is the GPT-4o annotation. The 1,000-document `dev` split was never used to train
    the released XLM-R teacher, so it is the clean held-out evaluation set of this project.
    """
    df = pd.read_json(EMMEDIATOPIC_PATH, lines=True)
    df = df.rename(columns={"GPT-IPTC-label": "label"})
    df["text"] = df["text"].map(truncate_words)
    df["label_id"] = df["label"].map(LABEL2ID)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df
