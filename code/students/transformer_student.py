"""Small transformer student: mMiniLMv2 (6 layers, 384 hidden, distilled from XLM-R-large)
fine-tuned with knowledge distillation from the IPTC teacher, then exported to ONNX.

This is the "middle ground" reference: contextual, but far lighter than XLM-R-large.

Usage:
    python students/transformer_student.py --name minilm_l6_kd --max-length 256
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABELS, MODELS_DIR, NUM_LABELS  # noqa: E402  (sets HF_HOME)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from students.training_data import load_pool, teacher_probs  # noqa: E402

BASE_MODEL = "nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large"


def train(args) -> Path:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=NUM_LABELS, id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)}, dtype=torch.float32).cuda()

    df = load_pool(tuple(args.sources))
    texts = df["text"].tolist()
    probs = teacher_probs(df, args.temperature).astype(np.float32)
    perm = rng.permutation(len(texts))
    n_val = int(len(texts) * args.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    steps = args.epochs * (len(train_idx) // args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler()

    def encode(idx):
        return tokenizer([texts[i] for i in idx], truncation=True, max_length=args.max_length,
                         padding=True, return_tensors="pt").to("cuda")

    def val_agreement():
        model.eval()
        preds = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for s in range(0, len(val_idx), 128):
                preds.append(model(**encode(val_idx[s:s + 128])).logits.argmax(-1).cpu().numpy())
        model.train()
        return float((np.concatenate(preds) == probs[val_idx].argmax(1)).mean())

    out = MODELS_DIR / "transformer" / args.name
    best = -1.0
    for epoch in range(args.epochs):
        rng.shuffle(train_idx)
        for step in range(len(train_idx) // args.batch_size):
            idx = train_idx[step * args.batch_size:(step + 1) * args.batch_size]
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**encode(idx)).logits.float()
            target = torch.from_numpy(probs[idx]).cuda()
            loss = F.kl_div(F.log_softmax(logits / args.temperature, -1), target,
                            reduction="batchmean") * args.temperature ** 2
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 500 == 0:
                print(f"epoch {epoch + 1} step {step}: loss={loss.item():.4f}", flush=True)
        acc = val_agreement()
        print(f"epoch {epoch + 1}: val_agreement={acc:.4f}", flush=True)
        if acc > best:
            best = acc
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
    (out / "train_meta.json").write_text(json.dumps({**vars(args), "val_agreement_with_teacher": best}, indent=1))
    export_onnx(out, args.max_length)
    return out


def export_onnx(model_dir: Path, max_length: int) -> None:
    """Export to ONNX (model.onnx, fp32) plus a dynamic-int8 variant (model_int8.onnx).

    The int8 variant is kept for reference only: even per-channel MatMul-only quantisation
    cost ~14 macro-F1 points on dev for this 384-dim student, so fp32 is the deployed model.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval().float().cpu()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    dummy = tokenizer(["export"] * 2, return_tensors="pt", padding="max_length", max_length=16)
    fp32_path = model_dir / "model.onnx"
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]), str(fp32_path),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17,
    )
    quantize_dynamic(str(fp32_path), str(model_dir / "model_int8.onnx"), weight_type=QuantType.QInt8,
                     per_channel=True, op_types_to_quantize=["MatMul"])
    print(f"exported {fp32_path} (+ model_int8.onnx)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--sources", nargs="*", default=["emmediatopic", "ccnews"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--val-fraction", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.export_only:
        export_onnx(MODELS_DIR / "transformer" / args.name, args.max_length)
    else:
        train(args)


if __name__ == "__main__":
    main()
