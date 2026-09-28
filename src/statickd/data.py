"""Loaders for every dataset used in the study.

| name                | content                                                        | labels                    |
|---------------------|----------------------------------------------------------------|---------------------------|
| EMMediaTopic train  | 20,000 news articles (ca, el, hr, sl)                          | GPT-4o + teacher logits   |
| EMMediaTopic dev    | 1,000 articles, never used to train the teacher                | GPT-4o                    |
| CC-News train       | 109,842 articles, 69 languages (2019-2021)                     | teacher logits            |
| CC-News extra       | 156,157 further articles from the same training sites          | teacher logits            |
| CC-News held-out    | 13,549 articles (12,554 from sites absent from all training data) | teacher logits (fidelity) |
| unified test        | 12,744 articles, 47 languages, 2024-2026 + public datasets     | human / publisher labels  |
"""
import numpy as np
import pandas as pd

from statickd.config import LABEL2ID, PROCESSED_DIR, RAW_DIR, truncate_words

EMMEDIATOPIC_PATH = RAW_DIR / "EMMediaTopic-1.0.jsonl"
TEACHER_LABELS_DIR = PROCESSED_DIR / "teacher_labels"
CCNEWS_DIR = PROCESSED_DIR / "ccnews"
TESTSET_DIR = PROCESSED_DIR / "testset_jsonld"
UNIFIED_TEST_PATH = TESTSET_DIR / "unified_test.jsonl"

DISTILLATION_SOURCES = ("emmediatopic", "ccnews", "ccnews_extra")


def load_emmediatopic(split: str | None = None) -> pd.DataFrame:
    """EMMediaTopic 1.0 (Kuzman & Ljubesic, 2025; CLARIN.SI 11356/1991).

    Columns: document_id, lang, text, label (GPT-4o), label_id, split. Texts are cut to 512 words.
    """
    df = pd.read_json(EMMEDIATOPIC_PATH, lines=True).rename(columns={"GPT-IPTC-label": "label"})
    df["text"] = df["text"].map(truncate_words)
    df["label_id"] = df["label"].map(LABEL2ID)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def load_distillation_pool(sources: tuple[str, ...] = DISTILLATION_SOURCES) -> pd.DataFrame:
    """Unlabelled (teacher-labelled) training documents for the students.

    sources: 'emmediatopic' (EMMediaTopic train split), 'ccnews' (CC-News train),
    'ccnews_extra' (additional CC-News documents). The EMMediaTopic dev split is never included.
    Columns: text, lang, source, domain, gpt_label_id (-1 outside EMMediaTopic), teacher_logits.
    """
    frames = []
    if "emmediatopic" in sources:
        emm = pd.read_parquet(TEACHER_LABELS_DIR / "emmediatopic.parquet")
        emm = emm[emm["split"] == "train"]
        frames.append(pd.DataFrame({"text": emm["text"], "lang": emm["lang"], "source": "emmediatopic",
                                    "domain": "emmediatopic", "gpt_label_id": emm["label_id"],
                                    "teacher_logits": emm["teacher_logits"]}))
    for source, file in (("ccnews", "ccnews_train"), ("ccnews_extra", "ccnews_train_extra")):
        if source in sources:
            cc = pd.read_parquet(TEACHER_LABELS_DIR / f"{file}.parquet")
            frames.append(pd.DataFrame({"text": cc["text"], "lang": cc["lang"], "source": source,
                                        "domain": cc["domain"], "gpt_label_id": -1,
                                        "teacher_logits": cc["teacher_logits"]}))
    return pd.concat(frames, ignore_index=True)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def teacher_probs(df: pd.DataFrame, temperature: float = 1.0) -> np.ndarray:
    """Teacher distribution softened with temperature tau: softmax(z / tau)."""
    return softmax(np.stack(df["teacher_logits"].to_numpy()), temperature)


def load_ccnews_heldout() -> pd.DataFrame:
    """CC-News documents from news sites never seen in training; label = teacher prediction."""
    return pd.read_parquet(TEACHER_LABELS_DIR / "ccnews_heldout.parquet")


def load_unified_test() -> pd.DataFrame:
    """The unified multilingual IPTC test set (see statickd.testset.unified).

    Columns: doc_id, source, label_origin, lang, label, label_id, source_label, headline, text,
    url, domain, lid_checked, train_overlap.
    """
    df = pd.read_json(UNIFIED_TEST_PATH, lines=True, dtype={"domain": str, "url": str})
    df["domain"] = df["domain"].fillna("")
    return df


def site_clusters(df: pd.DataFrame) -> np.ndarray:
    """Sampling cluster of every test article: its news site, or its source dataset when the dataset
    has no URLs. Used by the site-cluster bootstrap."""
    domain = df["domain"].fillna("").astype(str)
    return np.where(domain.eq(""), "src:" + df["source"].astype(str), domain)
