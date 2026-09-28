"""The teacher: Kuzman & Ljubesic's XLM-RoBERTa-large IPTC classifier
(classla/multilingual-IPTC-news-topic-classifier, 559.9M parameters, max 512 tokens).

It is used offline, once, to label the unlabelled distillation corpus with its full 17-logit
vector (the soft labels). A GPU only speeds this up; the logits do not depend on the device.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import statickd.config as config  # noqa: F401  (sets HF_HOME before transformers is imported)
from statickd.config import LABEL2ID, LABELS, TEACHER_MODEL_ID


class Teacher:
    """Returns logits in the project-wide label order (config.LABELS)."""

    def __init__(self, device: str | None = None, fp16: bool = True, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
        dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL_ID, dtype=dtype)
        self.model.to(self.device).eval()
        id2label = self.model.config.id2label
        # permutation from the model's output index to the project label index
        self.order = np.array([LABEL2ID[id2label[i]] for i in range(len(LABELS))])
        self.num_parameters = sum(p.numel() for p in self.model.parameters())

    def logits(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """(n, 17) logits. Texts are sorted by length so that batches need little padding."""
        order = np.argsort([len(t) for t in texts])[::-1]
        out = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                idx = order[start:start + batch_size]
                enc = self.tokenizer([texts[i] for i in idx], truncation=True, max_length=self.max_length,
                                     padding=True, return_tensors="pt").to(self.device)
                model_logits = self.model(**enc).logits.float().cpu().numpy()
                out[idx[:, None], self.order[None, :]] = model_logits
        return out

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.logits(texts).argmax(axis=1)


def label_corpus(teacher: Teacher, df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    """Add the columns teacher_logits (17 floats) and teacher_label_id to a corpus."""
    start = time.perf_counter()
    logits = teacher.logits(df["text"].tolist(), batch_size=batch_size)
    elapsed = time.perf_counter() - start
    print(f"labelled {len(df)} docs in {elapsed / 60:.1f} min ({len(df) / elapsed:.1f} docs/s)", flush=True)
    return df.assign(teacher_logits=list(logits), teacher_label_id=logits.argmax(axis=1))
