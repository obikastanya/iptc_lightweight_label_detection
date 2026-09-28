"""Standalone inference for StaticKD models (only `numpy`, `tokenizers` and `safetensors` are needed).

    from statickd_inference import StaticKD
    model = StaticKD(".")                      # folder with model.safetensors, tokenizer.json, config.json
    model.predict(["The central bank raised interest rates ..."])
    # -> ['economy, business and finance']

A document's logits are the mean of the table rows of its tokens plus a bias:
    logits(d) = mean_i T[t_i] + b
"""
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer


class StaticKD:
    def __init__(self, path: str = "."):
        path = Path(path)
        tensors = load_file(str(path / "model.safetensors"))
        self.table, self.bias = tensors["token_table"], tensors["bias"]
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        config = json.loads((path / "config.json").read_text(encoding="utf8"))
        self.labels = [config["id2label"][str(i)] for i in range(config["num_labels"])]
        self.max_tokens = int(config.get("max_tokens", 1024))
        self.unk = next((self.tokenizer.token_to_id(t) for t in ("[UNK]", "<unk>")
                         if self.tokenizer.token_to_id(t) is not None), None)

    def logits(self, texts: list[str]) -> np.ndarray:
        """(n, 17) logits. Special tokens are not added and unknown pieces are ignored."""
        rows = []
        for enc in self.tokenizer.encode_batch(list(texts), add_special_tokens=False):
            ids = np.asarray(enc.ids[:self.max_tokens], dtype=np.int64)
            if self.unk is not None:
                ids = ids[ids != self.unk]
            if len(ids) == 0:
                ids = np.zeros(1, dtype=np.int64)
            rows.append(self.table[ids].mean(axis=0, dtype=np.float32))
        return np.stack(rows) + self.bias

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        z = self.logits(texts)
        z = np.exp(z - z.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)

    def predict(self, texts: list[str]) -> list[str]:
        return [self.labels[i] for i in self.logits(texts).argmax(axis=1)]


if __name__ == "__main__":
    model = StaticKD(Path(__file__).parent)
    for text in ["The national team won the World Cup final 2-1 after extra time.",
                 "Bank Indonesia menaikkan suku bunga acuan untuk menahan inflasi.",
                 "Heavy rain and strong winds are expected across the region tomorrow."]:
        print(model.predict([text])[0], "<-", text)
