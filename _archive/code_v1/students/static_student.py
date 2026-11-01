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
_HASH_MUL = np.uint64(0x9E3779B97F4A7C15)  # Fibonacci hashing constant


def load_base(name: str = BASE_EMBEDDINGS) -> tuple[Tokenizer, np.ndarray]:
    from huggingface_hub import snapshot_download
    from safetensors.numpy import load_file

    path = Path(name) if Path(name).exists() else Path(snapshot_download(name))
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    tensors = load_file(str(path / "model.safetensors"))
    embeddings = next(iter(tensors.values())).astype(np.float32)
    return tokenizer, embeddings


def encode(tokenizer: Tokenizer, texts: list[str], unk_id: int | None,
           max_tokens: int = MAX_TOKENS) -> list[np.ndarray]:
    encs = tokenizer.encode_batch(texts, add_special_tokens=False)
    out = []
    for e in encs:
        ids = np.asarray(e.ids[:max_tokens], dtype=np.int64)
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


def bigram_ids(ids: np.ndarray, bits: int) -> np.ndarray:
    """Hash each pair of consecutive token ids into one of 2**bits buckets (fastText-style).

    Pure uint64 arithmetic (wraps on overflow), so training and NumPy inference agree exactly.
    """
    if len(ids) < 2:
        return np.zeros(0, dtype=np.int64)
    a, b = ids[:-1].astype(np.uint64), ids[1:].astype(np.uint64)
    key = a * np.uint64(1_000_003) + b
    return ((key * _HASH_MUL) >> np.uint64(64 - bits)).astype(np.int64)


def token_idf(ids: list[np.ndarray], vocab_size: int) -> np.ndarray:
    """Smoothed inverse document frequency of every token over the training documents."""
    df = np.zeros(vocab_size, dtype=np.int64)
    for d in ids:
        df[np.unique(d)] += 1
    return (np.log((1 + len(ids)) / (1 + df)) + 1.0).astype(np.float32)


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
    if args.max_tokens < MAX_TOKENS:  # the cache holds 1,024 tokens; shorter inputs are prefixes
        ids = [i[:args.max_tokens] for i in ids]
    # Optional per-document weights: teacher confidence^gamma (down-weights documents the teacher
    # is unsure about), normalised to mean 1 so the learning rate keeps its meaning.
    sample_w = np.ones(len(ids), dtype=np.float32)
    if args.conf_weight > 0:
        sample_w = teacher_probs(df, 1.0).max(1).astype(np.float32) ** args.conf_weight
        sample_w /= sample_w.mean()
    bids = [bigram_ids(i, args.bigram_bits) for i in ids] if args.bigram_bits else None

    perm = rng.permutation(len(ids))
    n_val = int(len(ids) * args.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Token weighting: the document vector becomes sum(w_t e_t) / sum(w_t). With a linear head this
    # still folds into the table (one extra scalar per token): logits = sum(w_t T_t) / sum(w_t) + b.
    #   idf      w_t = idf_t ** p            (fixed)
    #   learned  w_t = exp(a_t), a_t trained  (initialised to p * log idf_t)
    weighted = args.token_weight != "none"
    emb = torch.nn.EmbeddingBag.from_pretrained(torch.from_numpy(base), freeze=args.freeze,
                                                mode="sum" if weighted else "mean", sparse=True).to(device)
    log_w = opt_weight = None
    if weighted:
        init = args.idf_power * np.log(token_idf(ids, len(base)))
        log_w = torch.nn.Embedding.from_pretrained(torch.from_numpy(init[:, None]),
                                                   freeze=args.token_weight == "idf", sparse=True).to(device)
        if args.token_weight == "learned":
            opt_weight = torch.optim.SparseAdam(list(log_w.parameters()), lr=args.lr_weight)
    head = build_head(torch, base.shape[1], args.hidden, args.dropout).to(device)
    opt_dense = torch.optim.AdamW(head.parameters(), lr=args.lr_head, weight_decay=1e-4)
    opt_sparse = None if args.freeze else torch.optim.SparseAdam(list(emb.parameters()), lr=args.lr_emb)
    # Optional hashed-bigram table that stores 17 logits per bucket directly (already "folded"),
    # zero-initialised so training starts from the unigram model. Its bag mean is added to the logits.
    bigram = opt_bigram = None
    if args.bigram_bits:
        bigram = torch.nn.EmbeddingBag(2 ** args.bigram_bits, NUM_LABELS, mode="mean", sparse=True).to(device)
        torch.nn.init.zeros_(bigram.weight)
        opt_bigram = torch.optim.SparseAdam(list(bigram.parameters()), lr=args.lr_bigram)

    def pack(seqs):
        offsets = np.cumsum([0] + [len(s) for s in seqs[:-1]])
        flat = torch.from_numpy(np.concatenate(seqs).astype(np.int64)).to(device)
        return flat, torch.from_numpy(offsets).to(device)

    def batch_tensors(idx, token_dropout=0.0):
        seqs = [ids[i] for i in idx]
        if token_dropout > 0:  # randomly drop tokens: cheap augmentation against over-fitting
            seqs = [s[rng.random(len(s)) >= token_dropout] if len(s) > 4 else s for s in seqs]
            seqs = [s if len(s) else ids[i][:1] for s, i in zip(seqs, idx)]
        return pack(seqs)

    def bigram_tensors(idx, token_dropout=0.0):
        seqs = [bids[i] for i in idx]
        if token_dropout > 0:  # bigrams are dropped independently of the unigrams
            seqs = [s[rng.random(len(s)) >= token_dropout] if len(s) > 4 else s for s in seqs]
        return pack(seqs)

    def pooled(flat, off):
        if log_w is None:
            return emb(flat, off)
        w = torch.exp(log_w(flat).squeeze(-1))
        lengths = torch.diff(off, append=torch.tensor([len(flat)], device=device))
        bag = torch.repeat_interleave(torch.arange(len(off), device=device), lengths)
        denom = torch.zeros(len(off), device=device, dtype=w.dtype).index_add_(0, bag, w)
        return emb(flat, off, per_sample_weights=w) / denom[:, None]

    def forward(idx, token_dropout=0.0):
        flat, off = batch_tensors(idx, token_dropout)
        logits = head(pooled(flat, off))
        if bigram is not None:
            bflat, boff = bigram_tensors(idx, token_dropout)
            logits = logits + bigram(bflat, boff)  # empty bags (1-token docs) give zeros
        return logits

    def evaluate(idx):
        emb.eval(), head.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(idx), 1024):
                preds.append(forward(idx[s:s + 1024]).argmax(-1).cpu().numpy())
        emb.train(), head.train()
        return float((np.concatenate(preds) == probs[idx].argmax(1)).mean())

    best_acc, best_state = -1.0, None
    steps_per_epoch = len(train_idx) // args.batch_size
    for epoch in range(args.epochs):
        rng.shuffle(train_idx)
        total = 0.0
        for step in range(steps_per_epoch):
            idx = train_idx[step * args.batch_size:(step + 1) * args.batch_size]
            logits = forward(idx, args.token_dropout)
            target = torch.from_numpy(probs[idx]).to(device)
            kl = F.kl_div(F.log_softmax(logits / args.temperature, -1), target, reduction="none").sum(-1)
            w = torch.from_numpy(sample_w[idx]).to(device)
            loss = (w * kl).mean() * args.temperature ** 2
            if args.gpt_weight > 0:  # optional: also fit GPT-4o labels where available
                y = torch.from_numpy(gpt[idx]).to(device)
                mask = y >= 0
                if mask.any():
                    loss = loss + args.gpt_weight * F.cross_entropy(logits[mask], y[mask],
                                                                    label_smoothing=args.label_smoothing)
            opt_dense.zero_grad()
            for opt in (opt_sparse, opt_bigram, opt_weight):
                if opt:
                    opt.zero_grad()
            loss.backward()
            opt_dense.step()
            for opt in (opt_sparse, opt_bigram, opt_weight):
                if opt:
                    opt.step()
            total += loss.item()
        val_acc = evaluate(val_idx)
        print(f"epoch {epoch + 1}: loss={total / steps_per_epoch:.4f} val_agreement={val_acc:.4f}", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            linears = [m for m in head if isinstance(m, torch.nn.Linear)]
            best_state = (emb.weight.detach().cpu().numpy().copy(),
                          [(m.weight.detach().cpu().numpy().copy(), m.bias.detach().cpu().numpy().copy())
                           for m in linears],
                          None if bigram is None else bigram.weight.detach().cpu().numpy().copy(),
                          None if log_w is None else np.exp(log_w.weight.detach().cpu().numpy().ravel()))

    return save(args.name, tokenizer, best_state[0], best_state[1],
                meta={"base": args.base, "val_agreement_with_teacher": best_acc, **vars(args)},
                keep_ids=None if args.keep_all_vocab else _used_ids(ids, args.min_count),
                bigram_table=best_state[2], token_weight=best_state[3])


def _used_ids(ids: list[np.ndarray], min_count: int) -> np.ndarray:
    counts = np.bincount(np.concatenate(ids))
    return np.flatnonzero(counts >= min_count)


def quantize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.abs(matrix).max(axis=1, keepdims=True) / 127.0
    scale[scale == 0] = 1.0
    return np.round(matrix / scale).astype(np.int8), scale.astype(np.float32).ravel()


def save(name, tokenizer, embeddings, layers, meta, keep_ids=None, bigram_table=None,
         token_weight=None) -> Path:
    """Save int8 embeddings + head layers. With `keep_ids`, tokens never seen in training are
    mapped to a shared zero vector (vocabulary pruning) to shrink the model."""
    out = MODELS_DIR / "static" / name
    out.mkdir(parents=True, exist_ok=True)
    vocab_size = embeddings.shape[0]
    if keep_ids is not None:
        remap = np.full(vocab_size, len(keep_ids), dtype=np.int32)  # last row = unknown / pruned
        remap[keep_ids] = np.arange(len(keep_ids), dtype=np.int32)
        embeddings = np.vstack([embeddings[keep_ids], np.zeros((1, embeddings.shape[1]), np.float32)])
        if token_weight is not None:
            token_weight = np.append(token_weight[keep_ids], np.float32(1e-6))
    else:
        remap = np.arange(vocab_size, dtype=np.int32)
    q, scale = quantize_rows(embeddings)
    head = {}
    for i, (w, b) in enumerate(layers):
        head[f"W{i}"], head[f"b{i}"] = w.astype(np.float32), b.astype(np.float32)
    if bigram_table is not None:
        head["bigram_table"] = bigram_table.astype(np.float16)
    if token_weight is not None:  # indexed like the embedding rows
        head["token_weight"] = token_weight.astype(np.float32)
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
    extra = {"bigram_table": data["bigram_table"]} if "bigram_table" in data else {}
    if "token_weight" in data:  # the table is indexed by token id, so apply the row remap
        extra["token_weight"] = data["token_weight"][data["remap"]].astype(np.float16)
    np.savez(dst / "model.npz", token_table=table, bias=bias.astype(np.float32), **extra)
    Tokenizer.from_file(str(src / "tokenizer.json")).save(str(dst / "tokenizer.json"), pretty=False)
    meta = json.loads((src / "meta.json").read_text())
    meta["format"] = "token_table_fp16"
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in dst.iterdir()) / 2**20
    print(f"exported {dst}: table {table.shape}, {size:.1f} MB")
    return dst


def ensemble_tables(table_dirs: list[str], name: str) -> Path:
    """Average several folded students trained with different seeds.

    Every student is linear in its token tables, so the mean of their logits equals the logits
    of one student whose tables and bias are the element-wise means: an ensemble at zero
    inference cost. All students must share the tokenizer (and bigram size, if any).
    """
    datas = [np.load(Path(d) / "model.npz") for d in table_dirs]
    mean = lambda key: np.mean([d[key].astype(np.float32) for d in datas], axis=0)  # noqa: E731
    arrays = {"token_table": mean("token_table").astype(np.float16), "bias": mean("bias")}
    if all("bigram_table" in d for d in datas):
        arrays["bigram_table"] = mean("bigram_table").astype(np.float16)
    if any("token_weight" in d for d in datas):
        raise ValueError("token-weighted students cannot be averaged exactly")
    dst = MODELS_DIR / "static" / name
    dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / "model.npz", **arrays)
    Tokenizer.from_file(str(Path(table_dirs[0]) / "tokenizer.json")).save(str(dst / "tokenizer.json"), pretty=False)
    meta = json.loads((Path(table_dirs[0]) / "meta.json").read_text())
    meta.update({"format": "token_table_fp16", "ensemble_of": [str(d) for d in table_dirs]})
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"ensembled {len(table_dirs)} students into {dst}")
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
        meta_file = path / "meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        self.max_tokens = int(meta.get("max_tokens", MAX_TOKENS))  # must match training
        self.size_mb = sum(f.stat().st_size for f in path.iterdir()) / 2**20
        self.bigram_table = data["bigram_table"] if "bigram_table" in data else None
        self.token_weight = data["token_weight"].astype(np.float32) if "token_weight" in data else None
        self.bigram_bits = int(np.log2(len(self.bigram_table))) if self.bigram_table is not None else 0
        if "token_table" in data:  # folded deployment format (see export_table)
            self.table, self.scale = data["token_table"], None
            self.remap = np.arange(len(self.table), dtype=np.int32)
            self.head = [(None, data["bias"])]
            self.num_parameters = int(self.table.size + data["bias"].size)
            if self.bigram_table is not None:
                self.num_parameters += int(self.bigram_table.size)
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
        token_ids = encode(self.tokenizer, texts, self.unk, self.max_tokens)
        rows = [self.remap[ids] for ids in token_ids]
        if self.token_weight is not None:  # weighted pooling: sum(w_t T_t) / sum(w_t)
            tw = self.token_weight
            if self.scale is None:
                x = np.stack([tw[r] @ self.table[r].astype(np.float32) / tw[r].sum() for r in rows])
            else:
                x = np.stack([tw[r] @ (self.table[r] * self.scale[r, None]) / tw[r].sum() for r in rows])
        elif self.scale is None:
            x = np.stack([self.table[r].mean(axis=0, dtype=np.float32) for r in rows])
        else:
            x = np.stack([(self.table[r] * self.scale[r, None]).mean(axis=0) for r in rows])
        for i, (w, b) in enumerate(self.head):
            x = x + b if w is None else x @ w.T + b
            if i < len(self.head) - 1:
                x = _gelu(x)
        if self.bigram_table is not None:
            for j, ids in enumerate(token_ids):
                bi = bigram_ids(ids, self.bigram_bits)
                if len(bi):
                    x[j] += self.bigram_table[bi].mean(axis=0, dtype=np.float32)
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
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="for the GPT-4o label loss")
    parser.add_argument("--conf-weight", type=float, default=0.0, help="gamma of teacher-confidence weights")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--token-weight", choices=["none", "idf", "learned"], default="none")
    parser.add_argument("--idf-power", type=float, default=0.5, help="p in idf**p (init for 'learned')")
    parser.add_argument("--lr-weight", type=float, default=1e-2)
    parser.add_argument("--freeze", action="store_true", help="train only the head")
    parser.add_argument("--val-fraction", type=float, default=0.03)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--keep-all-vocab", action="store_true")
    parser.add_argument("--bigram-bits", type=int, default=0, help="hashed bigram table of 2**bits rows (0 = off)")
    parser.add_argument("--lr-bigram", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
