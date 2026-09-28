"""Store the logits of one model on the evaluation sets (results/final/predictions/).

Usage (from the project root):
    python scripts/evaluate_model.py --name teacher_onnx_int8 --spec teacher-onnx-int8 --sets dev test --threads 8
    python scripts/evaluate_model.py --name teacher --spec teacher-gpu --sets dev heldout test
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statickd.baselines import load_predictor  # noqa: E402
from statickd.evaluation import evaluate_model  # noqa: E402


class _GpuTeacher:
    """The teacher in fp16 on the GPU: the same logits that labelled the distillation corpus."""

    def __init__(self):
        from statickd.teacher import Teacher
        self.teacher = Teacher(device="cuda", fp16=True)

    def logits(self, texts):
        return self.teacher.logits(texts, batch_size=16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--spec", required=True, help="baselines.load_predictor spec, or 'teacher-gpu'")
    parser.add_argument("--sets", nargs="*", default=["dev", "heldout", "test"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()
    predictor = _GpuTeacher() if args.spec == "teacher-gpu" else load_predictor(args.spec, args.threads)
    batch = 512 if args.spec == "teacher-gpu" else args.batch_size  # the GPU teacher batches internally
    summary = evaluate_model(args.name, predictor, sets=args.sets, overwrite=args.overwrite, batch_size=batch)
    for s, r in summary.items():
        print(f"{args.name} [{s}] macro-F1={r['macro_f1']:.4f} accuracy={r['accuracy']:.4f} n={r['n']}")
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
