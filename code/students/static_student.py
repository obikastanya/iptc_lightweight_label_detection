"""Static-embedding student distilled from the XLM-R teacher.

Architecture (as cheap as fastText at inference time):
    subword tokenizer -> embedding lookup -> mean pooling -> [small MLP] -> 17 logits
The embedding table is initialised from multilingual static embeddings (model2vec
potion-multilingual-128M, 101 languages, aligned across languages; or embeddings distilled
from the teacher itself) and fine-tuned end-to-end with knowledge distillation: KL divergence
to the teacher's temperature-softened distribution.

Inference needs only `tokenizers` + NumPy (no PyTorch). Embeddings are stored as int8 with a
per-row scale. With a linear head, the head is folded into the embedding table so each token
contributes a precomputed 17-dim vector.

Usage:
    python students/static_student.py --name static_potion_kd --epochs 10 --keep-all-vocab
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABELS, MODELS_DIR, NUM_LABELS, PROCESSED_DIR  # noqa: E402  (sets HF_HOME)

import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

BASE_EMBEDDINGS = "minishlab/potion-multilingual-128M"
MAX_TOKENS = 1024  # the 512-word input rarely exceeds this; cost is linear and tiny
TOKEN_CACHE_DIR = PROCESSED_DIR / "token_cache"


def load_base(name: str = BASE_EMBEDDINGS) -> tuple[Tokenizer, np.ndarray]:
    from huggingface_hub import snapshot_download
    from safetensors.numpy import load_file

    path = Path(name) if Path(name).exists() else Path(snapshot_download(name))
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    tensors = load_file(str(path / "model.safetensors"))
    embeddings = next(iter(tensors.values())).astype(np.float32)
    return tokenizer, embeddings


def encode(tokenizer: Tokenizer, texts: list[str], unk_id: int | None) -> list[np.ndarray]:
    encs = tokenizer.encode_batch(texts, add_special_tokens=False)
    out = []
    for e in encs:
        ids = np.asarray(e.ids[:MAX_TOKENS], dtype=np.int64)
        if unk_id is not None:
            ids = ids[ids != unk_id]
        out.append(ids if len(ids) else np.zeros(1, dtype=np.int64))
    return out


def _unk_id(tokenizer: Tokenizer) -> int | None:
    for tok in ("[UNK]", "<unk>"):
        idx = tokenizer.token_to_id(tok)
        if idx is not None:
            return idx
    return None


def encode_cached(tokenizer: Tokenizer, texts: list[str], unk_id: int | None, key: str) -> list[np.ndarray]:
    """Tokenising ~130k documents takes minutes, so token ids are cached per tokenizer + corpus."""
    digest = hashlib.md5((key + str(len(texts)) + texts[0][:200] + texts[-1][:200]).encode()).hexdigest()[:12]
    cache = TOKEN_CACHE_DIR / f"{digest}.npz"
    if cache.exists():
        data = np.load(cache)
        return np.split(data["flat"], np.cumsum(data["lengths"])[:-1])
    ids = encode(tokenizer, texts, unk_id)
    TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, flat=np.concatenate(ids), lengths=np.array([len(i) for i in ids]))
    return ids


# ----------------------------------------------------------------------------- training
def build_head(torch, dim: int, hidden: int, dropout: float):
    layers = [torch.nn.Dropout(dropout)]
    if hidden > 0:
        layers += [torch.nn.Linear(dim, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout)]
        dim = hidden
    layers.append(torch.nn.Linear(dim, NUM_LABELS))
    return torch.nn.Sequential(*layers)


def train(args) -> Path:
    import torch
    import torch.nn.functional as F
    from students.training_data import load_pool, teacher_probs

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tokenizer, base = load_base(args.base)
    unk = _unk_id(tokenizer)

    df = load_pool(tuple(args.sources))
    probs = teacher_probs(df, args.temperature).astype(np.float32)
    gpt = df["gpt_label_id"].to_numpy()
    t0 = time.perf_counter()
    ids = encode_cached(tokenizer, df["text"].tolist(), unk, key=f"{args.base}|{args.sources}")
    print(f"tokenised {len(ids)} docs in {time.perf_counter() - t0:.0f}s", flush=True)

    perm = rng.permutation(len(ids))
    n_val = int(len(ids) * args.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = torch.nn.EmbeddingBag.from_pretrained(torch.from_numpy(base), freeze=args.freeze,
                                                mode="mean", sparse=True).to(device)
    head = build_head(torch, base.shape[1], args.hidden, args.dropout).to(device)
    opt_dense = torch.optim.AdamW(head.parameters(), lr=args.lr_head, weight_decay=1e-4)
    opt_sparse = None if args.freeze else torch.optim.SparseAdam(list(emb.parameters()), lr=args.lr_emb)

    def batch_tensors(idx, token_dropout=0.0):
        seqs = [ids[i] for i in idx]
        if token_dropout > 0:  # randomly drop tokens: cheap augmentation against over-fitting
            seqs = [s[rng.random(len(s)) >= token_dropout] if len(s) > 4 else s for s in seqs]
            seqs = [s if len(s) else ids[i][:1] for s, i in zip(seqs, idx)]
        offsets = np.cumsum([0] + [len(s) for s in seqs[:-1]])
        flat = torch.from_numpy(np.concatenate(seqs)).to(device)
        return flat, torch.from_numpy(offsets).to(device)

    def evaluate(idx):
        emb.eval(), head.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(idx), 1024):
                flat, off = batch_tensors(idx[s:s + 1024])
                preds.append(head(emb(flat, off)).argmax(-1).cpu().numpy())
        emb.train(), head.train()
        return float((np.concatenate(preds) == probs[idx].argmax(1)).mean())

    best_acc, best_state = -1.0, None
    steps_per_epoch = len(train_idx) // args.batch_size
    for epoch in range(args.epochs):
        rng.shuffle(train_idx)
        total = 0.0
        for step in range(steps_per_epoch):
            idx = train_idx[step * args.batch_size:(step + 1) * args.batch_size]
            flat, off = batch_tensors(idx, args.token_dropout)
            logits = head(emb(flat, off))
            target = torch.from_numpy(probs[idx]).to(device)
            loss = F.kl_div(F.log_softmax(logits / args.temperature, -1), target,
                            reduction="batchmean") * args.temperature ** 2
            if args.gpt_weight > 0:  # optional: also fit GPT-4o labels where available
                y = torch.from_numpy(gpt[idx]).to(device)
                mask = y >= 0
                if mask.any():
                    loss = loss + args.gpt_weight * F.cross_entropy(logits[mask], y[mask])
            opt_dense.zero_grad()
            if opt_sparse:
                opt_sparse.zero_grad()
            loss.backward()
            opt_dense.step()
            if opt_sparse:
                opt_sparse.step()
            total += loss.item()
        val_acc = evaluate(val_idx)
        print(f"epoch {epoch + 1}: loss={total / steps_per_epoch:.4f} val_agreement={val_acc:.4f}", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            linears = [m for m in head if isinstance(m, torch.nn.Linear)]
            best_state = (emb.weight.detach().cpu().numpy().copy(),
                          [(m.weight.detach().cpu().numpy().copy(), m.bias.detach().cpu().numpy().copy())
                           for m in linears])

    return save(args.name, tokenizer, best_state[0], best_state[1],
                meta={"base": args.base, "val_agreement_with_teacher": best_acc, **vars(args)},
                keep_ids=None if args.keep_all_vocab else _used_ids(ids, args.min_count))


def _used_ids(ids: list[np.ndarray], min_count: int) -> np.ndarray:
    counts = np.bincount(np.concatenate(ids))
    return np.flatnonzero(counts >= min_count)


def quantize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.abs(matrix).max(axis=1, keepdims=True) / 127.0
    scale[scale == 0] = 1.0
    return np.round(matrix / scale).astype(np.int8), scale.astype(np.float32).ravel()


def save(name, tokenizer, embeddings, layers, meta, keep_ids=None) -> Path:
    """Save int8 embeddings + head layers. With `keep_ids`, tokens never seen in training are
    mapped to a shared zero vector (vocabulary pruning) to shrink the model."""
    out = MODELS_DIR / "static" / name
    out.mkdir(parents=True, exist_ok=True)
    vocab_size = embeddings.shape[0]
    if keep_ids is not None:
        remap = np.full(vocab_size, len(keep_ids), dtype=np.int32)  # last row = unknown / pruned
        remap[keep_ids] = np.arange(len(keep_ids), dtype=np.int32)
        embeddings = np.vstack([embeddings[keep_ids], np.zeros((1, embeddings.shape[1]), np.float32)])
    else:
        remap = np.arange(vocab_size, dtype=np.int32)
    q, scale = quantize_rows(embeddings)
    head = {}
    for i, (w, b) in enumerate(layers):
        head[f"W{i}"], head[f"b{i}"] = w.astype(np.float32), b.astype(np.float32)
    np.savez(out / "model.npz", emb_q=q, emb_scale=scale, remap=remap, **head)
    tokenizer.save(str(out / "tokenizer.json"), pretty=False)
    meta = {k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool, list, type(None)))}
    meta.update({"labels": LABELS, "rows": int(q.shape[0]), "dim": int(q.shape[1]), "head_layers": len(layers)})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {out}: {q.shape[0]} rows x {q.shape[1]} dims, {len(layers)} head layer(s)")
    return out


def export_table(model_dir: str) -> Path:
    """Deployment format for a linear-head student: one float16 row of 17 logits per token.

    mean(E[ids]) @ W.T + b == mean(T[ids]) + b with T = E @ W.T, so the 256-dim embedding
    table and the head collapse into a V x 17 table (~8x smaller than the int8 embeddings).
    """
    src = Path(model_dir)
    data = np.load(src / "model.npz")
    weight = data["weight"] if "weight" in data else data["W0"]
    bias = data["bias"] if "bias" in data else data["b0"]
    if "W1" in data:
        raise ValueError("only linear-head students can be folded into a token table")
    emb = data["emb_q"].astype(np.float32) * data["emb_scale"][:, None]
    table = (emb[data["remap"]] @ weight.T).astype(np.float16)
    dst = src.with_name(src.name + "_table")
    dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / "model.npz", token_table=table, bias=bias.astype(np.float32))
    Tokenizer.from_file(str(src / "tokenizer.json")).save(str(dst / "tokenizer.json"), pretty=False)
    meta = json.loads((src / "meta.json").read_text())
    meta["format"] = "token_table_fp16"
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in dst.iterdir()) / 2**20
    print(f"exported {dst}: table {table.shape}, {size:.1f} MB")
    return dst


# ----------------------------------------------------------------------------- inference
def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class StaticPredictor:
    """Pure NumPy inference: tokenize -> gather rows -> mean -> (MLP) -> logits."""

    def __init__(self, path: str, threads: int = 1):
        path = Path(path)
        data = np.load(path / "model.npz")
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.unk = _unk_id(self.tokenizer)
        self.size_mb = sum(f.stat().st_size for f in path.iterdir()) / 2**20
        if "token_table" in data:  # folded deployment format (see export_table)
            self.table, self.scale = data["token_table"], None
            self.remap = np.arange(len(self.table), dtype=np.int32)
            self.head = [(None, data["bias"])]
            self.num_parameters = int(self.table.size + data["bias"].size)
            return
        self.remap = data["remap"]
        if "weight" in data:  # models saved before the MLP option existed
            self.layers = [(data["weight"], data["bias"])]
        else:
            n = sum(1 for k in data.files if k.startswith("W"))
            self.layers = [(data[f"W{i}"], data[f"b{i}"]) for i in range(n)]
        if len(self.layers) == 1:
            # Fold the linear head into the table: mean(E[ids]) @ W.T = mean((E @ W.T)[ids]),
            # giving a small float32 table of 17 values per token.
            emb = data["emb_q"].astype(np.float32) * data["emb_scale"][:, None]
            self.table, self.scale = (emb @ self.layers[0][0].T).astype(np.float32), None
            self.head = [(None, self.layers[0][1])]
        else:
            # Keep the table in int8 (4x less RAM); only the rows of a document are de-quantised.
            self.table, self.scale = data["emb_q"], data["emb_scale"]
            self.head = self.layers
        self.num_parameters = int(data["emb_q"].size + sum(w.size + b.size for w, b in self.layers))

    def logits(self, texts: list[str]) -> np.ndarray:
        rows = [self.remap[ids] for ids in encode(self.tokenizer, texts, self.unk)]
        if self.scale is None:
            x = np.stack([self.table[r].mean(axis=0, dtype=np.float32) for r in rows])
        else:
            x = np.stack([(self.table[r] * self.scale[r, None]).mean(axis=0) for r in rows])
        for i, (w, b) in enumerate(self.head):
            x = x + b if w is None else x @ w.T + b
            if i < len(self.head) - 1:
                x = _gelu(x)
        return x

    def predict(self, texts):
        return self.logits(texts).argmax(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--base", default=BASE_EMBEDDINGS)
    parser.add_argument("--sources", nargs="*", default=["emmediatopic", "ccnews"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-emb", type=float, default=1e-3)
    parser.add_argument("--lr-head", type=float, default=3e-3)
    parser.add_argument("--hidden", type=int, default=0, help="hidden units of an MLP head (0 = linear)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--token-dropout", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--gpt-weight", type=float, default=0.0)
    parser.add_argument("--freeze", action="store_true", help="train only the head")
    parser.add_argument("--val-fraction", type=float, default=0.03)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--keep-all-vocab", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
