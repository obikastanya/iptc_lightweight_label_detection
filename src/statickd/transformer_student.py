"""Small transformer students distilled from the same teacher on the same data (baselines).

    mMiniLMv2-L6-H384 (nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large, 107M params incl. embeddings)
    DistilmBERT        (distilbert/distilbert-base-multilingual-cased, 135M params)

Fine-tuned with the same KD loss as StaticKD, then exported to ONNX fp32 for the CPU benchmark.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import statickd.config as config  # noqa: F401  (sets HF_HOME)
from statickd.config import LABELS, MODELS_DIR, NUM_LABELS
from statickd.data import DISTILLATION_SOURCES, load_distillation_pool, teacher_probs

BASES = {
    "minilm": "nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large",
    "distilmbert": "distilbert/distilbert-base-multilingual-cased",
}


@dataclass
class TransformerKDConfig:
    name: str
    base: str = BASES["minilm"]
    sources: tuple[str, ...] = DISTILLATION_SOURCES
    epochs: int = 2
    batch_size: int = 32
    lr: float = 5e-5
    max_length: int = 256
    temperature: float = 2.0
    val_fraction: float = 0.03
    seed: int = 0


def train(cfg: TransformerKDConfig, out: Path | None = None) -> Path:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base, num_labels=NUM_LABELS, id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)}, dtype=torch.float32).cuda()  # fp32 master weights (AMP)
    df = load_distillation_pool(tuple(cfg.sources))
    texts = df["text"].tolist()
    probs = teacher_probs(df, cfg.temperature).astype(np.float32)
    perm = rng.permutation(len(texts))
    n_val = int(len(texts) * cfg.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    steps = cfg.epochs * (len(train_idx) // cfg.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler()

    def encode(idx):
        return tokenizer([texts[i] for i in idx], truncation=True, max_length=cfg.max_length,
                         padding=True, return_tensors="pt").to("cuda")

    def val_agreement():
        model.eval()
        preds = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for s in range(0, len(val_idx), 128):
                preds.append(model(**encode(val_idx[s:s + 128])).logits.argmax(-1).cpu().numpy())
        model.train()
        return float((np.concatenate(preds) == probs[val_idx].argmax(1)).mean())

    out = Path(out or MODELS_DIR / "baselines" / cfg.name)
    best, history, t0 = -1.0, [], time.perf_counter()
    for epoch in range(cfg.epochs):
        rng.shuffle(train_idx)
        for step in range(len(train_idx) // cfg.batch_size):
            idx = train_idx[step * cfg.batch_size:(step + 1) * cfg.batch_size]
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**encode(idx)).logits.float()
            target = torch.from_numpy(probs[idx]).cuda()
            loss = F.kl_div(F.log_softmax(logits / cfg.temperature, -1), target,
                            reduction="batchmean") * cfg.temperature ** 2
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 1000 == 0:
                print(f"[{cfg.name}] epoch {epoch + 1} step {step}: loss={loss.item():.4f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
        agreement = val_agreement()
        history.append({"epoch": epoch + 1, "val_agreement": agreement, "seconds": time.perf_counter() - t0})
        print(f"[{cfg.name}] epoch {epoch + 1}: val_agreement={agreement:.4f}", flush=True)
        if agreement > best:
            best = agreement
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
    meta = {**asdict(cfg), "history": history, "best_val_agreement": best,
            "train_seconds": time.perf_counter() - t0}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    export_onnx(out)
    return out


def export_onnx(model_dir: Path) -> Path:
    """ONNX fp32 export (dynamic batch and sequence length) for CPU inference."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval().float().cpu()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    dummy = tokenizer(["export"] * 2, return_tensors="pt", padding="max_length", max_length=16)
    path = model_dir / "model.onnx"
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]), str(path),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    return path
