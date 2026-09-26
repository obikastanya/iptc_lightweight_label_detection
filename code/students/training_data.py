"""Training pools for the student models.

Every row carries the document text, its language, the GPT-4o label (EMMediaTopic only) and
the teacher's 17 logits. The EMMediaTopic dev split is never included here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TEACHER_DIR = PROCESSED_DIR / "teacher_labels"


def load_pool(sources: tuple[str, ...] = ("emmediatopic", "ccnews")) -> pd.DataFrame:
    """sources: 'emmediatopic' (20k train docs, 4 languages), 'ccnews' (110k docs, ~70 languages)
    and 'ccnews_extra' (156k additional CC-News docs for the data-scaling experiment)."""
    frames = []
    if "emmediatopic" in sources:
        emm = pd.read_parquet(TEACHER_DIR / "emmediatopic.parquet")
        emm = emm[emm["split"] == "train"]
        frames.append(pd.DataFrame({
            "text": emm["text"], "lang": emm["lang"], "source": "emmediatopic",
            "gpt_label_id": emm["label_id"], "teacher_logits": emm["teacher_logits"],
        }))
    for source, file in (("ccnews", "ccnews_train"), ("ccnews_extra", "ccnews_train_extra")):
        if source in sources:
            cc = pd.read_parquet(TEACHER_DIR / f"{file}.parquet")
            frames.append(pd.DataFrame({
                "text": cc["text"], "lang": cc["lang"], "source": source,
                "gpt_label_id": -1, "teacher_logits": cc["teacher_logits"],
            }))
    return pd.concat(frames, ignore_index=True)


def teacher_probs(df: pd.DataFrame, temperature: float = 1.0) -> np.ndarray:
    logits = np.stack(df["teacher_logits"].to_numpy()) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)
