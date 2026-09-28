"""StaticKD student: static multilingual token embeddings + mean pooling + linear head, trained by
knowledge distillation and folded into a table of 17 logits per token.

Model
    tokens t_1..t_n  ->  h = (1/n) sum_i E[t_i]  ->  z = W h + b              (training form)
    because W is linear:  z = (1/n) sum_i T[t_i] + b   with  T = E W^T       (deployment form)

    E is initialised from potion-multilingual-128M (Model2Vec static embeddings distilled from
    BGE-M3, 101 languages, aligned across languages), 500,353 tokens x 256 dims.

Loss (Hinton et al.):  L = tau^2 * KL( softmax(z_teacher / tau) || softmax(z_student / tau) )

Deployment needs only NumPy + `tokenizers`: tokenise, gather rows of T, average, add b.
Training needs PyTorch (a GPU makes it ~10x faster but is not required).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from statickd.config import BASE_EMBEDDINGS, LABELS, MODELS_DIR, NUM_LABELS, PROCESSED_DIR
from statickd.data import DISTILLATION_SOURCES, load_distillation_pool, softmax, teacher_probs

MAX_TOKENS = 1024          # a 512-word article rarely exceeds this; inference cost is linear anyway
TOKEN_CACHE_DIR = PROCESSED_DIR / "token_cache"
STUDENTS_DIR = MODELS_DIR / "students"
_HASH_MUL = np.uint64(0x9E3779B97F4A7C15)  # Fibonacci hashing constant (bigram option)


# ----------------------------------------------------------------------------- tokenisation
def load_base_embeddings(name: str = BASE_EMBEDDINGS) -> tuple[Tokenizer, np.ndarray]:
    """Tokenizer and float32 embedding matrix of a Model2Vec static model (local dir or HF id)."""
    from huggingface_hub import snapshot_download
    from safetensors.numpy import load_file

    path = Path(name) if Path(name).exists() else Path(snapshot_download(name))
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    embeddings = next(iter(load_file(str(path / "model.safetensors")).values())).astype(np.float32)
    return tokenizer, embeddings


def unk_id(tokenizer: Tokenizer) -> int | None:
    for token in ("[UNK]", "<unk>"):
        idx = tokenizer.token_to_id(token)
        if idx is not None:
            return idx
    return None


def encode(tokenizer: Tokenizer, texts: list[str], max_tokens: int = MAX_TOKENS) -> list[np.ndarray]:
    """Token ids of every text: no special tokens, unknown pieces removed, at most `max_tokens`.
    A text without any known token keeps id 0 so that its bag is never empty."""
    unk = unk_id(tokenizer)
    out = []
    for enc in tokenizer.encode_batch(texts, add_special_tokens=False):
        ids = np.asarray(enc.ids[:max_tokens], dtype=np.int64)
        if unk is not None:
            ids = ids[ids != unk]
        out.append(ids if len(ids) else np.zeros(1, dtype=np.int64))
    return out


def encode_cached(tokenizer: Tokenizer, texts: list[str], key: str) -> list[np.ndarray]:
    """Tokenising 286k articles takes a few minutes, so the ids are cached per tokenizer + corpus."""
    digest = hashlib.md5((key + str(len(texts)) + texts[0][:200] + texts[-1][:200]).encode()).hexdigest()[:12]
    cache = TOKEN_CACHE_DIR / f"{digest}.npz"
    if cache.exists():
        data = np.load(cache)
        return np.split(data["flat"], np.cumsum(data["lengths"])[:-1])
    ids = encode(tokenizer, texts)
    TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, flat=np.concatenate(ids), lengths=np.array([len(i) for i in ids]))
    return ids


def bigram_ids(ids: np.ndarray, bits: int) -> np.ndarray:
    """Hash each pair of consecutive tokens into one of 2**bits buckets (exploration option).
    Pure uint64 arithmetic, so training and NumPy inference agree exactly."""
    if len(ids) < 2:
        return np.zeros(0, dtype=np.int64)
    key = ids[:-1].astype(np.uint64) * np.uint64(1_000_003) + ids[1:].astype(np.uint64)
    return ((key * _HASH_MUL) >> np.uint64(64 - bits)).astype(np.int64)


def token_idf(ids: list[np.ndarray], vocab_size: int) -> np.ndarray:
    """Smoothed inverse document frequency of every token over the training documents."""
    df = np.zeros(vocab_size, dtype=np.int64)
    for d in ids:
        df[np.unique(d)] += 1
    return (np.log((1 + len(ids)) / (1 + df)) + 1.0).astype(np.float32)


# ----------------------------------------------------------------------------- configuration
@dataclass
class StaticKDConfig:
    """One training run. The defaults are the final StaticKD recipe (one seed); an earlier version used temperature=2."""

    name: str
    sources: tuple[str, ...] = DISTILLATION_SOURCES
    seed: int = 0
    # student
    init: str = "potion"            # 'potion' (aligned multilingual) | 'random' (ablation)
    freeze_embeddings: bool = False  # True: train the head only (ablation)
    hidden: int = 0                  # > 0: MLP head with this many units (cannot be folded)
    dropout: float = 0.1             # dropout on the pooled vector
    # distillation
    loss: str = "kd"                 # 'kd': soft teacher distribution | 'hard': teacher arg-max
    temperature: float = 1.0
    token_dropout: float = 0.2       # each training token is dropped with this probability
    # optimisation
    epochs: int = 6
    batch_size: int = 64
    lr_emb: float = 1e-3             # SparseAdam on the embedding rows
    lr_head: float = 3e-3            # AdamW on the head
    val_fraction: float = 0.03       # internal validation split (agreement with the teacher)
    max_docs: int = 0                # > 0: random subset of the pool (data-size ablation)
    # exploration options (negative results in the paper)
    bigram_bits: int = 0             # hashed bigram logit table of 2**bits rows
    lr_bigram: float = 1e-2
    idf_power: float = 0.0           # > 0: fixed idf**p weighted pooling
    conf_weight: float = 0.0         # gamma: weight documents by teacher confidence**gamma
    gpt_weight: float = 0.0          # + weight * CE(GPT-4o label) where available
    base: str = BASE_EMBEDDINGS
    extra: dict = field(default_factory=dict)

    @property
    def foldable(self) -> bool:
        return self.hidden == 0


# ----------------------------------------------------------------------------- training
def train(cfg: StaticKDConfig, out_dir: Path | None = None, verbose: bool = True) -> Path:
    """Train one student and save it (folded into a token table when the head is linear)."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    tokenizer, base = load_base_embeddings(cfg.base)
    if cfg.init == "random":
        base = np.random.default_rng(cfg.seed).normal(0.0, base.std(), base.shape).astype(np.float32)

    df = load_distillation_pool(tuple(cfg.sources))
    ids = encode_cached(tokenizer, df["text"].tolist(), key=f"{cfg.base}|{list(cfg.sources)}")
    if cfg.max_docs and cfg.max_docs < len(df):
        keep = np.sort(np.random.default_rng(1000 + cfg.seed).choice(len(df), cfg.max_docs, replace=False))
        df, ids = df.iloc[keep].reset_index(drop=True), [ids[i] for i in keep]
    probs = teacher_probs(df, cfg.temperature).astype(np.float32)
    hard = probs.argmax(1)
    gpt = df["gpt_label_id"].to_numpy()
    sample_w = np.ones(len(ids), dtype=np.float32)
    if cfg.conf_weight > 0:
        sample_w = teacher_probs(df, 1.0).max(1).astype(np.float32) ** cfg.conf_weight
        sample_w /= sample_w.mean()
    bids = [bigram_ids(i, cfg.bigram_bits) for i in ids] if cfg.bigram_bits else None

    perm = rng.permutation(len(ids))
    n_val = int(len(ids) * cfg.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weighted = cfg.idf_power > 0
    emb = torch.nn.EmbeddingBag.from_pretrained(torch.from_numpy(base), freeze=cfg.freeze_embeddings,
                                                mode="sum" if weighted else "mean", sparse=True).to(device)
    tok_w = None
    if weighted:  # fixed weights w_t = idf_t**p; the document vector is sum(w e) / sum(w)
        tok_w = torch.from_numpy(token_idf(ids, len(base)) ** cfg.idf_power).to(device)
    layers = [torch.nn.Dropout(cfg.dropout)]
    dim = base.shape[1]
    if cfg.hidden:
        layers += [torch.nn.Linear(dim, cfg.hidden), torch.nn.GELU(), torch.nn.Dropout(cfg.dropout)]
        dim = cfg.hidden
    layers.append(torch.nn.Linear(dim, NUM_LABELS))
    head = torch.nn.Sequential(*layers).to(device)
    optimisers = [torch.optim.AdamW(head.parameters(), lr=cfg.lr_head, weight_decay=1e-4)]
    if not cfg.freeze_embeddings:
        optimisers.append(torch.optim.SparseAdam(list(emb.parameters()), lr=cfg.lr_emb))
    bigram = None
    if cfg.bigram_bits:  # stores 17 logits per bucket directly, zero-initialised
        bigram = torch.nn.EmbeddingBag(2 ** cfg.bigram_bits, NUM_LABELS, mode="mean", sparse=True).to(device)
        torch.nn.init.zeros_(bigram.weight)
        optimisers.append(torch.optim.SparseAdam(list(bigram.parameters()), lr=cfg.lr_bigram))

    def pack(seqs):
        offsets = np.cumsum([0] + [len(s) for s in seqs[:-1]])
        flat = torch.from_numpy(np.concatenate(seqs).astype(np.int64)).to(device)
        return flat, torch.from_numpy(offsets).to(device)

    def drop_tokens(seqs, fallback):
        seqs = [s[rng.random(len(s)) >= cfg.token_dropout] if len(s) > 4 else s for s in seqs]
        return [s if len(s) else f[:1] for s, f in zip(seqs, fallback)]

    def forward(idx, augment=False):
        seqs = [ids[i] for i in idx]
        if augment and cfg.token_dropout > 0:
            seqs = drop_tokens(seqs, seqs)
        flat, off = pack(seqs)
        if tok_w is None:
            pooled = emb(flat, off)
        else:
            w = tok_w[flat]
            lengths = torch.diff(off, append=torch.tensor([len(flat)], device=device))
            bag = torch.repeat_interleave(torch.arange(len(off), device=device), lengths)
            denom = torch.zeros(len(off), device=device).index_add_(0, bag, w)
            pooled = emb(flat, off, per_sample_weights=w) / denom[:, None]
        logits = head(pooled)
        if bigram is not None:
            bseqs = [bids[i] for i in idx]
            if augment and cfg.token_dropout > 0:
                bseqs = [s[rng.random(len(s)) >= cfg.token_dropout] if len(s) > 4 else s for s in bseqs]
            logits = logits + bigram(*pack(bseqs))
        return logits

    def val_agreement():
        emb.eval(), head.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(val_idx), 1024):
                preds.append(forward(val_idx[s:s + 1024]).argmax(-1).cpu().numpy())
        emb.train(), head.train()
        return float((np.concatenate(preds) == hard[val_idx]).mean())

    history, best, best_state = [], -1.0, None
    steps = len(train_idx) // cfg.batch_size
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        rng.shuffle(train_idx)
        total = 0.0
        for step in range(steps):
            idx = train_idx[step * cfg.batch_size:(step + 1) * cfg.batch_size]
            logits = forward(idx, augment=True)
            w = torch.from_numpy(sample_w[idx]).to(device)
            if cfg.loss == "kd":
                target = torch.from_numpy(probs[idx]).to(device)
                per_doc = F.kl_div(F.log_softmax(logits / cfg.temperature, -1), target, reduction="none").sum(-1)
                loss = (w * per_doc).mean() * cfg.temperature ** 2
            else:
                loss = (w * F.cross_entropy(logits, torch.from_numpy(hard[idx]).to(device), reduction="none")).mean()
            if cfg.gpt_weight > 0:
                y = torch.from_numpy(gpt[idx]).to(device)
                if (y >= 0).any():
                    loss = loss + cfg.gpt_weight * F.cross_entropy(logits[y >= 0], y[y >= 0])
            for opt in optimisers:
                opt.zero_grad()
            loss.backward()
            for opt in optimisers:
                opt.step()
            total += loss.item()
        agreement = val_agreement()
        history.append({"epoch": epoch + 1, "loss": total / steps, "val_agreement": agreement,
                        "seconds": time.perf_counter() - t0})
        if verbose:
            print(f"[{cfg.name}] epoch {epoch + 1}: loss={total / steps:.4f} "
                  f"val_agreement={agreement:.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if agreement > best:
            best = agreement
            linears = [m for m in head if isinstance(m, torch.nn.Linear)]
            best_state = {
                "emb": emb.weight.detach().cpu().numpy().copy(),
                "layers": [(m.weight.detach().cpu().numpy().copy(), m.bias.detach().cpu().numpy().copy())
                           for m in linears],
                "bigram": None if bigram is None else bigram.weight.detach().cpu().numpy().copy(),
            }
    meta = {"config": asdict(cfg), "history": history, "best_val_agreement": best,
            "train_docs": int(len(train_idx)), "val_docs": int(n_val), "device": device,
            "train_seconds": time.perf_counter() - t0}
    idf = None if tok_w is None else tok_w.cpu().numpy()
    return save_student(out_dir or STUDENTS_DIR / cfg.name, tokenizer, best_state, meta, idf)


def save_student(out: Path, tokenizer: Tokenizer, state: dict, meta: dict,
                 token_weight: np.ndarray | None = None) -> Path:
    """Linear head: fold into T = E W^T (fp16). MLP head: keep fp16 embeddings + the layers."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    if len(state["layers"]) == 1:
        weight, bias = state["layers"][0]
        arrays["token_table"] = (state["emb"] @ weight.T).astype(np.float16)
        arrays["bias"] = bias.astype(np.float32)
        fmt = "token_table_fp16"
    else:
        arrays["embeddings"] = state["emb"].astype(np.float16)
        for i, (w, b) in enumerate(state["layers"]):
            arrays[f"W{i}"], arrays[f"b{i}"] = w.astype(np.float32), b.astype(np.float32)
        fmt = "embeddings_mlp"
    if state.get("bigram") is not None:
        arrays["bigram_table"] = state["bigram"].astype(np.float16)
    if token_weight is not None:
        arrays["token_weight"] = token_weight.astype(np.float32)
    np.savez(out / "model.npz", **arrays)
    tokenizer.save(str(out / "tokenizer.json"), pretty=False)
    meta = {**meta, "format": fmt, "labels": LABELS, "max_tokens": MAX_TOKENS}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf8")
    return out


def ensemble(model_dirs: list[Path], out: Path) -> Path:
    """Average several folded students (different seeds) into one.

    The student is linear in its table, so mean_k((1/n) sum_i T_k[t_i] + b_k) equals
    (1/n) sum_i mean_k(T_k)[t_i] + mean_k(b_k): the ensemble costs nothing at inference time.
    """
    datas = [np.load(Path(d) / "model.npz") for d in model_dirs]
    if any("token_table" not in d or "token_weight" in d for d in datas):
        raise ValueError("only plain folded students can be averaged exactly")
    mean = lambda key: np.mean([d[key].astype(np.float32) for d in datas], axis=0)  # noqa: E731
    arrays = {"token_table": mean("token_table").astype(np.float16), "bias": mean("bias")}
    if all("bigram_table" in d for d in datas):
        arrays["bigram_table"] = mean("bigram_table").astype(np.float16)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "model.npz", **arrays)
    Tokenizer.from_file(str(Path(model_dirs[0]) / "tokenizer.json")).save(str(out / "tokenizer.json"), pretty=False)
    metas = [json.loads((Path(d) / "meta.json").read_text(encoding="utf8")) for d in model_dirs]
    meta = {"format": "token_table_fp16", "labels": LABELS, "max_tokens": MAX_TOKENS,
            "ensemble_of": [Path(d).name for d in model_dirs], "config": metas[0]["config"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf8")
    return out


def token_counts(vocab_size: int, key: str = f"{BASE_EMBEDDINGS}|{list(DISTILLATION_SOURCES)}") -> np.ndarray:
    """How often every token occurs in the distillation corpus (from the token cache)."""
    tokenizer, _ = load_base_embeddings()
    df = load_distillation_pool()
    ids = encode_cached(tokenizer, df["text"].tolist(), key=key)
    return np.bincount(np.concatenate(ids), minlength=vocab_size)


def compact(model_dir: Path, out: Path, min_count: int = 10) -> Path:
    """Shrink RAM by pruning the tokenizer itself: pieces seen < min_count times in the distillation
    corpus are removed (single characters are always kept, so every text can still be segmented).

    The 500k-piece Unigram tokenizer needs ~490 MB of RAM, the 17-logit table only ~17 MB."""
    src = Path(model_dir)
    spec = json.loads((src / "tokenizer.json").read_text(encoding="utf8"))
    vocab = spec["model"]["vocab"]
    counts = token_counts(len(vocab))
    single = np.array([len(piece.lstrip("▁")) <= 1 for piece, _ in vocab])
    keep = np.union1d(np.flatnonzero((counts >= min_count) | single), [0, 1])  # + [PAD], [UNK]
    spec["model"]["vocab"] = [vocab[i] for i in keep]
    spec["model"]["unk_id"] = int(np.searchsorted(keep, spec["model"]["unk_id"]))
    spec["added_tokens"] = [t for t in spec.get("added_tokens", []) if t["id"] in (0, 1)]
    data = np.load(src / "model.npz")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "model.npz", token_table=data["token_table"][keep], bias=data["bias"])
    (out / "tokenizer.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf8")
    Tokenizer.from_file(str(out / "tokenizer.json"))  # validate
    meta = json.loads((src / "meta.json").read_text(encoding="utf8"))
    meta.update({"compact_min_count": min_count, "vocab_size": int(len(keep)), "compact_of": src.name})
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf8")
    return out


# ----------------------------------------------------------------------------- inference
def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class StaticKDModel:
    """Pure NumPy inference: tokenise -> gather table rows -> mean -> + bias -> 17 logits."""

    def __init__(self, path: str | Path):
        path = Path(path)
        data = np.load(path / "model.npz")
        self.path = path
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        meta_file = path / "meta.json"
        self.meta = json.loads(meta_file.read_text(encoding="utf8")) if meta_file.exists() else {}
        self.max_tokens = int(self.meta.get("max_tokens", MAX_TOKENS))
        self.size_mb = sum(f.stat().st_size for f in path.iterdir()) / 2**20
        self.bigram = data["bigram_table"] if "bigram_table" in data else None
        self.bigram_bits = int(np.log2(len(self.bigram))) if self.bigram is not None else 0
        self.token_weight = data["token_weight"] if "token_weight" in data else None
        if "token_table" in data:
            self.table, self.bias, self.layers = data["token_table"], data["bias"], None
            self.num_parameters = int(self.table.size + self.bias.size)
        else:
            self.table = data["embeddings"]
            n = sum(1 for k in data.files if k.startswith("W"))
            self.layers = [(data[f"W{i}"], data[f"b{i}"]) for i in range(n)]
            self.num_parameters = int(self.table.size + sum(w.size + b.size for w, b in self.layers))
        if self.bigram is not None:
            self.num_parameters += int(self.bigram.size)

    @classmethod
    def load(cls, path: str | Path) -> "StaticKDModel":
        return cls(path)

    def logits(self, texts: list[str]) -> np.ndarray:
        ids = encode(self.tokenizer, list(texts), self.max_tokens)
        if self.token_weight is None:
            x = np.stack([self.table[i].mean(axis=0, dtype=np.float32) for i in ids])
        else:
            tw = self.token_weight
            x = np.stack([tw[i] @ self.table[i].astype(np.float32) / tw[i].sum() for i in ids])
        if self.layers is None:
            x = x + self.bias
        else:
            for k, (w, b) in enumerate(self.layers):
                x = x @ w.T + b
                if k < len(self.layers) - 1:
                    x = _gelu(x)
        if self.bigram is not None:
            for j, doc in enumerate(ids):
                bi = bigram_ids(doc, self.bigram_bits)
                if len(bi):
                    x[j] += self.bigram[bi].mean(axis=0, dtype=np.float32)
        return x

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.logits(texts).argmax(axis=1)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return softmax(self.logits(texts))

    def predict_labels(self, texts: list[str]) -> list[str]:
        return [LABELS[i] for i in self.predict(texts)]
