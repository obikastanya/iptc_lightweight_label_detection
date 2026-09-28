"""Paths, labels and constants shared by every module.

The project root is the folder that contains `data/`, `models/` and `results/`. It defaults to the
parent of `src/` and can be moved with the environment variable STATICKD_ROOT. Import this module
before `transformers` / `huggingface_hub`, because it points the Hugging Face cache into the project.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("STATICKD_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FINAL_DIR = RESULTS_DIR / "final"          # everything reported in the paper lives here
PREDICTIONS_DIR = FINAL_DIR / "predictions"  # one .npz of logits per (model, evaluation set)

os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf_cache"))

TEACHER_MODEL_ID = "classla/multilingual-IPTC-news-topic-classifier"
TEACHER_ONNX_ID = "onnx-community/multilingual-IPTC-news-topic-classifier-ONNX"
BASE_EMBEDDINGS = "minishlab/potion-multilingual-128M"

# Same preprocessing as Kuzman & Ljubesic (2025): title + body, first 512 words.
MAX_WORDS = 512
# The teacher's model card recommends at least 75 words.
MIN_WORDS = 75
# Scripts written without spaces: length is measured in characters instead of words.
NO_SPACE_LANGS = {"ja", "zh", "th", "my", "km", "lo"}
MIN_CHARS_NO_SPACE = 200
MAX_CHARS_NO_SPACE = 2000

# The 17 top-level IPTC Media Topics, in the fixed order used by every model and table.
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
SHORT_LABELS = ["arts", "conflict", "crime", "disaster", "economy", "education", "environment",
                "health", "human int.", "labour", "lifestyle", "politics", "religion", "sci-tech",
                "society", "sport", "weather"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

# Languages of the teacher's own fine-tuning data (EMMediaTopic).
TEACHER_TRAIN_LANGS = ("ca", "el", "hr", "sl")


def truncate_words(text: str, max_words: int = MAX_WORDS) -> str:
    """Keep the first `max_words` whitespace-separated words, as in the original paper."""
    return " ".join(text.split()[:max_words])


def truncate(text: str, lang: str) -> str:
    """Language-aware truncation: characters for scripts without spaces, words otherwise."""
    if lang in NO_SPACE_LANGS:
        return text[:MAX_CHARS_NO_SPACE]
    return truncate_words(text)


def long_enough(text: str, lang: str) -> bool:
    if lang in NO_SPACE_LANGS:
        return len(text) >= MIN_CHARS_NO_SPACE
    return len(text.split()) >= MIN_WORDS
