"""Distil static token embeddings from the teacher's own fine-tuned XLM-R-large encoder.

Model2Vec-style (Tulkens & van Dongen, 2024): every vocabulary token is passed through the
encoder on its own (`<s> token </s>`); its contextual output vector becomes the token's static
embedding. PCA reduces 1024 -> 256 dimensions and Zipf weighting down-weights frequent tokens
(XLM-R's SentencePiece ids are roughly frequency ordered). The result has the same layout as
a model2vec model (tokenizer.json + model.safetensors) and can be used as `--base` for
students/static_student.py. Because the encoder was fine-tuned for IPTC topics, these
embeddings start out in a topic-aware space, unlike generic sentence-embedding vectors.

Usage:
    python students/distill_static_embeddings.py --out models/static_base/teacher_distilled_256
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS_DIR, TEACHER_MODEL_ID  # noqa: E402  (sets HF_HOME)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402


@torch.inference_mode()
def token_vectors(model, tokenizer, batch_size: int = 2048) -> np.ndarray:
    vocab_size = len(tokenizer)
    bos, eos = tokenizer.cls_token_id, tokenizer.sep_token_id
    out = np.zeros((vocab_size, model.config.hidden_size), dtype=np.float32)
    for start in range(0, vocab_size, batch_size):
        ids = torch.arange(start, min(start + batch_size, vocab_size))
        batch = torch.stack([torch.full_like(ids, bos), ids, torch.full_like(ids, eos)], dim=1).cuda()
        hidden = model(input_ids=batch, attention_mask=torch.ones_like(batch)).last_hidden_state
        out[start:start + len(ids)] = hidden[:, 1].float().cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(MODELS_DIR / "static_base" / "teacher_distilled_256"))
    parser.add_argument("--dims", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
    model = AutoModel.from_pretrained(TEACHER_MODEL_ID, dtype=torch.float16).cuda().eval()
    vectors = token_vectors(model, tokenizer)
    print(f"encoded {vectors.shape[0]} tokens", flush=True)

    vectors -= vectors.mean(axis=0)
    reduced = PCA(n_components=args.dims, random_state=0).fit_transform(vectors).astype(np.float32)
    zipf = np.log(1 + np.arange(len(reduced), dtype=np.float32))[:, None]
    zipf[:5] = 0.0  # <s>, <pad>, </s>, <unk> carry no topic information
    embeddings = reduced * zipf / max(np.linalg.norm(reduced, axis=1).mean(), 1e-6)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({"embeddings": embeddings.astype(np.float32)}, str(out / "model.safetensors"))
    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))
    print(f"saved {out}: {embeddings.shape}")


if __name__ == "__main__":
    main()
