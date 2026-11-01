"""Tokenisation shared by training and inference of the bag-of-tokens students."""
import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402,F401  (sets HF_HOME)

WORD_RE = re.compile(r"\w+", re.UNICODE)


def word_tokens(text: str) -> str:
    """Lower-cased word tokens (the classic fastText setting; weak for scripts without spaces)."""
    return " ".join(WORD_RE.findall(text.lower()))


@lru_cache(maxsize=None)
def subword_tokenizer(name: str = "xlm-roberta-base"):
    """XLM-R SentencePiece tokenizer: language-agnostic, also segments Chinese, Japanese, Thai."""
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download

    return Tokenizer.from_file(hf_hub_download(name, "tokenizer.json"))


def subword_tokens(texts: list[str], name: str = "xlm-roberta-base") -> list[str]:
    tok = subword_tokenizer(name)
    encs = tok.encode_batch([t.lower() for t in texts], add_special_tokens=False)
    return [" ".join(e.tokens) for e in encs]
