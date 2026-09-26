"""Shared paths and constants for the lightweight IPTC news topic classification research.

Import this module before transformers / huggingface_hub so the model cache lives in the project.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf_cache"))

EMMEDIATOPIC_PATH = RAW_DIR / "EMMediaTopic-1.0.jsonl"
TEACHER_MODEL_ID = "classla/multilingual-IPTC-news-topic-classifier"

# Same preprocessing as Kuzman & Ljubesic (2025): first 512 words of the article.
MAX_WORDS = 512
# The model card recommends texts of at least 75 words.
MIN_WORDS = 75

# The 17 top-level IPTC Media Topic labels, in a fixed order used by every student model.
LABELS = [
    "arts, culture, entertainment and media",
    "conflict, war and peace",
    "crime, law and justice",
    "disaster, accident and emergency incident",
    "economy, business and finance",
    "education",
    "environment",
    "health",
    "human interest",
    "labour",
    "lifestyle and leisure",
    "politics",
    "religion",
    "science and technology",
    "society",
    "sport",
    "weather",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

# Languages of the EMMediaTopic training data (seen by the teacher during fine-tuning).
TEACHER_TRAIN_LANGS = ["ca", "el", "hr", "sl"]


def truncate_words(text: str, max_words: int = MAX_WORDS) -> str:
    """Keep the first `max_words` whitespace-separated words, as in the original paper."""
    words = text.split()
    return " ".join(words[:max_words])
