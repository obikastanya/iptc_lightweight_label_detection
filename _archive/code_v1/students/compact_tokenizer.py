"""Shrink the RAM footprint of a token-table student by pruning the tokenizer vocabulary.

The 500k-piece Unigram tokenizer of potion-multilingual needs ~490 MB of RAM (its trie), while
the 17-logit table needs only ~17 MB. Pieces that are rare in the distillation corpus are
removed from the tokenizer itself (single characters are always kept, so every text can still
be segmented); Viterbi segmentation then falls back to shorter pieces for rare words.

Usage:
    python students/compact_tokenizer.py --model models/static/static_final_table --min-count 20
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402,F401

import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from students.static_student import TOKEN_CACHE_DIR  # noqa: E402


def corpus_counts(vocab_size: int) -> np.ndarray:
    """Token counts from the largest token cache that matches this vocabulary."""
    best = None
    for cache in TOKEN_CACHE_DIR.glob("*.npz"):
        flat = np.load(cache)["flat"]
        if flat.max() < vocab_size and (best is None or len(flat) > len(best)):
            best = flat
    return np.bincount(best, minlength=vocab_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="a *_table student directory")
    parser.add_argument("--min-count", type=int, default=20)
    args = parser.parse_args()

    src = Path(args.model)
    spec = json.loads((src / "tokenizer.json").read_text(encoding="utf8"))
    vocab = spec["model"]["vocab"]
    counts = corpus_counts(len(vocab))
    short = np.array([len(piece.lstrip("▁")) <= 1 for piece, _ in vocab])
    keep = np.flatnonzero((counts >= args.min_count) | short)
    keep = np.union1d(keep, [0, 1])  # [PAD], [UNK]

    spec["model"]["vocab"] = [vocab[i] for i in keep]
    spec["model"]["unk_id"] = int(np.searchsorted(keep, spec["model"]["unk_id"]))
    spec["added_tokens"] = [t for t in spec.get("added_tokens", []) if t["id"] in (0, 1)]

    data = np.load(src / "model.npz")
    dst = src.with_name(f"{src.name}_compact{args.min_count}")
    dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / "model.npz", token_table=data["token_table"][keep], bias=data["bias"])
    (dst / "tokenizer.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf8")
    Tokenizer.from_file(str(dst / "tokenizer.json"))  # validate
    meta = json.loads((src / "meta.json").read_text())
    meta.update({"tokenizer_min_count": args.min_count, "vocab_size": int(len(keep))})
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in dst.iterdir()) / 2**20
    print(f"kept {len(keep)}/{len(vocab)} pieces -> {dst} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
