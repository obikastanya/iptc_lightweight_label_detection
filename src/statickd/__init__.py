"""StaticKD: distilling a multilingual IPTC news topic classifier into a static token-logit table for lightweight CPU inference.

A multilingual XLM-R-large teacher is distilled into a static-embedding student whose linear head is
folded into one table of 17 logits per token. Inference needs only NumPy and `tokenizers`.

    from statickd import StaticKDModel
    model = StaticKDModel.load("models/final/statickd")
    model.predict_labels(["The central bank raised interest rates by 50 basis points ..."])
"""
from statickd.config import LABELS, NUM_LABELS  # noqa: F401
from statickd.student import StaticKDModel  # noqa: F401

__version__ = "1.0.0"
