"""Sparse-Distillation-style baseline (Ye et al., NAACL 2022), reviewer experiment E1.

A deep averaging network (DAN) whose embeddings are trained from scratch and distilled from the
teacher on the same data, targets and tokenizer as StaticKD, so the comparison isolates the
Sparse Distillation ingredients:

    tokens t_1..t_n (potion Unigram ids, same encode() as StaticKD, <= 1,024 tokens)
    ids    = unigrams  U  hashed bigrams (t_i, t_i+1)  U  hashed trigrams (t_i, t_i+1, t_i+2)
    h      = mean_{j in ids} E[j]                    E: (V + B) x d, random init, sparse gradients
    z      = W2 ReLU(W1 dropout(h) + b1) + b2       two-layer MLP head (not foldable)
    loss   = KL(softmax(z_teacher) || softmax(z))     tau = 1, stored teacher logits

The n-gram rows are embedding rows (averaged with the unigram rows into one document vector), not
the hashed bigram LOGIT table of statickd.student (bigram_bits). Hashing is pure uint64 NumPy
arithmetic shared by training and inference, so both see exactly the same ids.

Training: EmbeddingBag(mean, sparse) + SparseAdam for E, AdamW for the head, id dropout 0.2 (each
unigram / n-gram id is dropped from the bag, as word dropout in a DAN), dropout on the pooled vector
and the hidden layer, batch 64, the same 3% internal validation split as StaticKD (same seed ->
same split); the epoch with the highest validation agreement with the teacher is kept.
Inference: NumPy + `tokenizers` (embeddings stored in fp16).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from statickd.config import BASE_EMBEDDINGS, LABELS, MODELS_DIR, NUM_LABELS
from statickd.data import DISTILLATION_SOURCES, load_distillation_pool, softmax, teacher_probs
from statickd.student import MAX_TOKENS, encode, encode_cached, load_base_embeddings

SD_DIR = MODELS_DIR / "review_e1"
_P = np.uint64(1_000_003)
_MUL = np.uint64(0x9E3779B97F4A7C15)
_TRI_SALT = np.uint64(0x632BE59BD9B4E019)
_SHIFT = np.uint64(20)


# ----------------------------------------------------------------------------- hashing
def ngram_ids(ids: np.ndarray, vocab_size: int, buckets: int) -> np.ndarray:
    """Unigram ids followed by the hashed bigram and trigram ids (offset by vocab_size)."""
    ids = np.asarray(ids, dtype=np.int64)
    parts = [ids]
    u = ids.astype(np.uint64)
    if len(u) >= 2:
        key2 = u[:-1] * _P + u[1:]
        parts.append(((key2 * _MUL) >> _SHIFT) % np.uint64(buckets))
        if len(u) >= 3:
            key3 = (key2[:-1] * _P + u[2:]) ^ _TRI_SALT
            parts.append(((key3 * _MUL) >> _SHIFT) % np.uint64(buckets))
    out = [parts[0]] + [p.astype(np.int64) + vocab_size for p in parts[1:]]
    return np.concatenate(out)


# ----------------------------------------------------------------------------- configuration
@dataclass
class SDConfig:
    name: str
    seed: int = 0
    dim: int = 128
    buckets: int = 524_288           # hashed bigram + trigram rows (shared table)
    hidden: int = 1024
    init_std: float = 0.1            # N(0, init_std^2), fixed a priori (no potion vectors)
    dropout: float = 0.1             # on the pooled vector and the hidden layer
    token_dropout: float = 0.2       # each id (unigram or n-gram) dropped with this probability
    temperature: float = 1.0
    epochs: int = 10
    batch_size: int = 64
    lr_emb: float = 1e-3             # SparseAdam
    lr_head: float = 3e-3            # AdamW (as StaticKD)
    val_fraction: float = 0.03
    sources: tuple[str, ...] = DISTILLATION_SOURCES
    base: str = BASE_EMBEDDINGS      # only its tokenizer is used

    def num_embedding_parameters(self, vocab_size: int) -> int:
        return (vocab_size + self.buckets) * self.dim

    def num_head_parameters(self) -> int:
        return self.dim * self.hidden + self.hidden + self.hidden * NUM_LABELS + NUM_LABELS


def _flatten(seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return np.concatenate(seqs).astype(np.int32), np.stack([starts, lengths], 1)


# ----------------------------------------------------------------------------- training
def train(cfg: SDConfig, out_dir: Path | None = None, monitor: dict | None = None, verbose: bool = True,
          save: bool = True) -> dict:
    """Train one model. `monitor` = {"ids": [token id arrays], "y": labels}: macro-F1 of every epoch
    on that set (development set only) is recorded in the history (used for tuning, GPU fp32)."""
    import torch
    import torch.nn.functional as F

    from statickd.evaluation import macro_f1

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    tokenizer, base = load_base_embeddings(cfg.base)
    vocab = base.shape[0]
    del base

    df = load_distillation_pool(tuple(cfg.sources))
    tok = encode_cached(tokenizer, df["text"].tolist(), key=f"{cfg.base}|{list(cfg.sources)}")
    probs = teacher_probs(df, cfg.temperature).astype(np.float32)
    hard = probs.argmax(1)
    del df
    t_prep = time.perf_counter()
    flat, span = _flatten([ngram_ids(t, vocab, cfg.buckets) for t in tok])
    del tok
    if verbose:
        print(f"[{cfg.name}] n-gram ids: {len(flat):,} ({time.perf_counter() - t_prep:.0f}s)", flush=True)

    # same internal validation split as statickd.student.train (first use of the same rng)
    perm = rng.permutation(len(span))
    n_val = int(len(span) * cfg.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = vocab + cfg.buckets
    emb = torch.nn.EmbeddingBag(rows, cfg.dim, mode="mean", sparse=True)
    torch.nn.init.normal_(emb.weight, 0.0, cfg.init_std)
    emb = emb.to(device)
    head = torch.nn.Sequential(torch.nn.Dropout(cfg.dropout), torch.nn.Linear(cfg.dim, cfg.hidden), torch.nn.ReLU(),
                               torch.nn.Dropout(cfg.dropout), torch.nn.Linear(cfg.hidden, NUM_LABELS)).to(device)
    opt_head = torch.optim.AdamW(head.parameters(), lr=cfg.lr_head, weight_decay=1e-4)
    opt_emb = torch.optim.SparseAdam(list(emb.parameters()), lr=cfg.lr_emb)

    def batch_ids(idx, augment):
        seqs = [flat[s:s + n] for s, n in span[idx]]
        if augment and cfg.token_dropout > 0:
            seqs = [s[rng.random(len(s)) >= cfg.token_dropout] if len(s) > 4 else s for s in seqs]
            seqs = [s if len(s) else flat[st:st + 1] for s, (st, _) in zip(seqs, span[idx])]
        return seqs

    def pack(seqs):
        offsets = np.concatenate([[0], np.cumsum([len(s) for s in seqs[:-1]])]).astype(np.int64)
        return (torch.from_numpy(np.concatenate(seqs).astype(np.int64)).to(device),
                torch.from_numpy(offsets).to(device))

    def forward(seqs):
        return head(emb(*pack(seqs)))

    def predict(seqs_all, bs=512):
        emb.eval(), head.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(seqs_all), bs):
                out.append(forward(seqs_all[s:s + bs]).argmax(-1).cpu().numpy())
        emb.train(), head.train()
        return np.concatenate(out)

    mon_seqs = None
    if monitor is not None:
        mon_seqs = [ngram_ids(t, vocab, cfg.buckets) for t in monitor["ids"]]
    val_seqs = batch_ids(val_idx, augment=False)

    history, best, best_state = [], -1.0, None
    steps = len(train_idx) // cfg.batch_size
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        rng.shuffle(train_idx)
        total = 0.0
        for step in range(steps):
            idx = train_idx[step * cfg.batch_size:(step + 1) * cfg.batch_size]
            logits = forward(batch_ids(idx, augment=True))
            target = torch.from_numpy(probs[idx]).to(device)
            loss = F.kl_div(F.log_softmax(logits / cfg.temperature, -1), target,
                            reduction="none").sum(-1).mean() * cfg.temperature ** 2
            opt_head.zero_grad(), opt_emb.zero_grad()
            loss.backward()
            opt_head.step(), opt_emb.step()
            total += loss.item()
        agreement = float((predict(val_seqs) == hard[val_idx]).mean())
        row = {"epoch": epoch + 1, "loss": total / steps, "val_agreement": agreement,
               "seconds": time.perf_counter() - t0}
        if mon_seqs is not None:
            row["monitor_macro_f1"] = macro_f1(monitor["y"], predict(mon_seqs))
            row["monitor_accuracy"] = float((predict(mon_seqs) == np.asarray(monitor["y"])).mean())
        history.append(row)
        if verbose:
            extra = f" monitor_f1={row['monitor_macro_f1']:.4f}" if mon_seqs is not None else ""
            print(f"[{cfg.name}] epoch {epoch + 1}: loss={row['loss']:.4f} val_agreement={agreement:.4f}{extra} "
                  f"({row['seconds']:.0f}s)", flush=True)
        if agreement > best:
            best = agreement
            if save:
                linears = [m for m in head if isinstance(m, torch.nn.Linear)]
                best_state = {"emb": emb.weight.detach().cpu().numpy().astype(np.float16),
                              "layers": [(m.weight.detach().cpu().numpy().copy(), m.bias.detach().cpu().numpy().copy())
                                         for m in linears]}
    meta = {"config": asdict(cfg), "history": history, "best_val_agreement": best,
            "best_epoch": int(np.argmax([h["val_agreement"] for h in history]) + 1),
            "train_docs": int(len(train_idx)), "val_docs": int(n_val), "device": device,
            "vocab_size": int(vocab), "embedding_parameters": cfg.num_embedding_parameters(vocab),
            "head_parameters": cfg.num_head_parameters(),
            "peak_gpu_mb": (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None,
            "train_seconds": time.perf_counter() - t0}
    if save:
        save_model(out_dir or SD_DIR / cfg.name, tokenizer, best_state, meta)
    return meta


def save_model(out: Path, tokenizer: Tokenizer, state: dict, meta: dict) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"embeddings": state["emb"].astype(np.float16)}
    for i, (w, b) in enumerate(state["layers"]):
        arrays[f"W{i}"], arrays[f"b{i}"] = w.astype(np.float32), b.astype(np.float32)
    np.save(out / "embeddings.npy", arrays.pop("embeddings"))   # memory-mappable, no zip overhead
    np.savez(out / "head.npz", **arrays)
    tokenizer.save(str(out / "tokenizer.json"), pretty=False)
    cfg = meta["config"]
    meta = {**meta, "format": "sdstyle_dan_fp16", "labels": LABELS, "max_tokens": MAX_TOKENS,
            "vocab_size": meta["vocab_size"], "buckets": cfg["buckets"], "dim": cfg["dim"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf8")
    return out


# ----------------------------------------------------------------------------- inference
class SDStyleModel:
    """NumPy inference: tokenise -> unigram + hashed n-gram ids -> mean of fp16 rows -> ReLU MLP."""

    def __init__(self, path: str | Path):
        path = Path(path)
        self.path = path
        self.meta = json.loads((path / "meta.json").read_text(encoding="utf8"))
        self.table = np.load(path / "embeddings.npy")
        head = np.load(path / "head.npz")
        n = sum(1 for k in head.files if k.startswith("W"))
        self.layers = [(head[f"W{i}"], head[f"b{i}"]) for i in range(n)]
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.vocab_size = int(self.meta["vocab_size"])
        self.buckets = int(self.meta["buckets"])
        self.max_tokens = int(self.meta.get("max_tokens", MAX_TOKENS))
        self.size_mb = sum(f.stat().st_size for f in path.iterdir()) / 2**20
        self.num_parameters = int(self.table.size + sum(w.size + b.size for w, b in self.layers))

    def logits(self, texts: list[str]) -> np.ndarray:
        ids = encode(self.tokenizer, list(texts), self.max_tokens)
        x = np.stack([self.table[ngram_ids(i, self.vocab_size, self.buckets)].mean(axis=0, dtype=np.float32)
                      for i in ids])
        for k, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if k < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
        return x

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.logits(texts).argmax(axis=1)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return softmax(self.logits(texts))
