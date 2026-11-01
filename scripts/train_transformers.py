"""Train the transformer KD students (GPU), export them to ONNX and store their predictions.

Both students use the same distillation pool and KD loss as StaticKD. The temperature of MiniLM is
chosen on the development set (EMMediaTopic dev) from {1, 2}; DistilmBERT uses the chosen value.
Predictions are computed with PyTorch on the GPU (numerically equivalent to the ONNX fp32 export,
which is checked on the dev set); CPU speed is measured separately with the ONNX files.

Usage (from the project root):
    python scripts/train_transformers.py [--only minilm_kd_t1 minilm_kd_t2 distilmbert_kd]
    python scripts/train_transformers.py --max-length 512 --only minilm_kd_t1_len512 distilmbert_kd_len512 [--grad-accum 2]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statickd.analysis import BASELINES_DIR  # noqa: E402
from statickd.baselines import OnnxClassifier  # noqa: E402
from statickd.evaluation import eval_set, evaluate_model, has_predictions, load_pred  # noqa: E402
from statickd.transformer_student import BASES, TransformerKDConfig, export_onnx, train  # noqa: E402


class GpuClassifier:
    def __init__(self, model_dir, max_length):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.max_length = torch, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()

    def logits(self, texts):
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.float16):
            enc = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding=True,
                                 return_tensors="pt").to("cuda")
            return self.model(**enc).logits.float().cpu().numpy()


def run(name: str, cfg: TransformerKDConfig) -> None:
    out = BASELINES_DIR / name
    if not (out / "model.onnx").exists():
        if (out / "train_meta.json").exists():
            export_onnx(out)
        else:
            train(cfg, out)
    if not all(has_predictions(name, s) for s in ("dev", "heldout", "test")):
        summary = evaluate_model(name, GpuClassifier(out, cfg.max_length), batch_size=128 if cfg.max_length <= 256 else 32)
        print(name, json.dumps(summary), flush=True)
    # ONNX fp32 must give the same predictions as PyTorch (checked on 200 dev documents)
    texts = eval_set("dev")["text"].tolist()[:200]
    onnx_pred = OnnxClassifier(out, "model.onnx", threads=8, max_length=cfg.max_length).logits(texts).argmax(1)
    agree = float((onnx_pred == load_pred(name, "dev")[:200]).mean())
    meta = json.loads((out / "train_meta.json").read_text())
    meta["onnx_vs_pytorch_agreement_dev200"] = agree
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{name}: ONNX fp32 vs PyTorch agreement on 200 dev docs = {agree:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=["minilm_kd_t1", "minilm_kd_t2", "distilmbert_kd"])
    parser.add_argument("--max-length", type=int, default=256,
                        help="input token limit; 512 trains *_len512 students with tau = 1, the value chosen on dev at 256")
    parser.add_argument("--grad-accum", type=int, default=1, help="micro-batches per batch of 32 (GPU memory)")
    args = parser.parse_args()
    if args.max_length != 256:
        sfx = f"_len{args.max_length}"
        for base in ("minilm", "distilmbert"):
            name = ("minilm_kd_t1" if base == "minilm" else "distilmbert_kd") + sfx
            if name in args.only:
                run(name, TransformerKDConfig(name=name, base=BASES[base], temperature=1.0,
                                              max_length=args.max_length, grad_accum=args.grad_accum))
        return
    for tau in (1.0, 2.0):
        name = f"minilm_kd_t{int(tau)}"
        if name in args.only:
            run(name, TransformerKDConfig(name=name, base=BASES["minilm"], temperature=tau))
    if "distilmbert_kd" in args.only:
        # temperature chosen on dev (never on the test set)
        dev = eval_set("dev")["y"].to_numpy()
        from statickd.evaluation import macro_f1
        scores = {t: macro_f1(dev, load_pred(f"minilm_kd_t{t}", "dev")) for t in (1, 2)
                  if has_predictions(f"minilm_kd_t{t}", "dev")}
        tau = float(max(scores, key=scores.get)) if scores else 2.0
        print(f"DistilmBERT temperature (best MiniLM on dev): {tau} {scores}", flush=True)
        run("distilmbert_kd", TransformerKDConfig(name="distilmbert_kd", base=BASES["distilmbert"], temperature=tau))


if __name__ == "__main__":
    main()
