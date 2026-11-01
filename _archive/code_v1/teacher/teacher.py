"""Wrapper around Kuzman & Ljubesic's XLM-R-large IPTC classifier (the teacher model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABEL2ID, LABELS, TEACHER_MODEL_ID  # noqa: E402  (sets HF_HOME)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402


class Teacher:
    """Returns logits in the project-wide label order (config.LABELS)."""

    def __init__(self, device: str = "cuda", fp16: bool = True, max_length: int = 512):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
        dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL_ID, torch_dtype=dtype)
        self.model.to(device).eval()
        id2label = self.model.config.id2label
        # Permutation from the model's output index to the project label index.
        self.order = np.array([LABEL2ID[id2label[i]] for i in range(len(LABELS))])

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def logits(self, texts: list[str], batch_size: int = 32, sort_by_length: bool = True) -> np.ndarray:
        """Logits of shape (n_texts, 17). Texts are length-sorted to minimise padding."""
        order = np.argsort([len(t) for t in texts])[::-1] if sort_by_length else np.arange(len(texts))
        out = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            idx = order[start:start + batch_size]
            enc = self.tokenizer([texts[i] for i in idx], truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt").to(self.device)
            model_logits = self.model(**enc).logits.float().cpu().numpy()
            reordered = np.zeros_like(model_logits)
            reordered[:, self.order] = model_logits
            out[idx] = reordered
        return out

    def predict(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.logits(texts, batch_size=batch_size).argmax(axis=1)
