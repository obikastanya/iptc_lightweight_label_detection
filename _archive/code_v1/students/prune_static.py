"""Post-hoc vocabulary pruning of a trained static student.

Tokens that occur fewer than `--min-count` times in the distillation corpus are mapped to a
shared zero row. This shrinks the int8 table (and RAM) with little effect on accuracy, because
rare tokens contribute little to a mean-pooled document vector.

Usage:
    python students/prune_static.py --model models/static/static_potion_kd_tdrop --min-count 3
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS_DIR  # noqa: E402

import numpy as np  # noqa: E402

from students.static_student import TOKEN_CACHE_DIR  # noqa: E402


def token_counts(vocab_size: int) -> np.ndarray:
    """Token frequencies from the cached tokenisation that matches this vocabulary."""
    # Caches are keyed by a hash; the matching one is the cache whose largest token id fits
    # this vocabulary most tightly (potion: 500k tokens, XLM-R: 250k tokens).
    best, best_gap = None, None
    for cache in TOKEN_CACHE_DIR.glob("*.npz"):
        flat = np.load(cache)["flat"]
        gap = vocab_size - int(flat.max())
        if gap > 0 and (best_gap is None or gap < best_gap):
            best, best_gap = flat, gap
    if best is None:
        raise FileNotFoundError("no token cache matching this vocabulary")
    return np.bincount(best, minlength=vocab_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--min-count", type=int, default=3)
    args = parser.parse_args()

    src = Path(args.model)
    data = dict(np.load(src / "model.npz"))
    remap_old = data["remap"]
    vocab_size = len(remap_old)
    counts = token_counts(vocab_size)
    keep = np.flatnonzero(counts >= args.min_count)

    rows = remap_old[keep]
    emb_q = np.vstack([data["emb_q"][rows], np.zeros((1, data["emb_q"].shape[1]), np.int8)])
    scale = np.concatenate([data["emb_scale"][rows], [1.0]]).astype(np.float32)
    remap = np.full(vocab_size, len(keep), dtype=np.int32)
    remap[keep] = np.arange(len(keep), dtype=np.int32)
    data.update(emb_q=emb_q, emb_scale=scale, remap=remap)

    dst = MODELS_DIR / "static" / f"{src.name}_pruned{args.min_count}"
    dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / "model.npz", **data)
    shutil.copy(src / "tokenizer.json", dst / "tokenizer.json")
    meta = json.loads((src / "meta.json").read_text())
    meta.update({"pruned_min_count": args.min_count, "rows": int(emb_q.shape[0])})
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in dst.iterdir()) / 2**20
    print(f"kept {len(keep)}/{vocab_size} tokens -> {dst} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
