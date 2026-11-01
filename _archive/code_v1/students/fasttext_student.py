"""fastText student (Joulin et al., 2017): the "fastText language-ID"-style baseline.

Label sources
  gpt           GPT-4o labels of EMMediaTopic only (what Kuzman's student was trained on)
  teacher       teacher arg-max labels on the full multilingual pool (hard distillation)
  teacher_soft  like `teacher`, plus extra copies of a document for every label to which the
                teacher gives >= 20% probability (soft targets approximated by resampling)

Usage:
    python students/fasttext_student.py --labels teacher --tokens subword --name ft_teacher_subword
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABELS, MODELS_DIR  # noqa: E402

import fasttext  # noqa: E402
import numpy as np  # noqa: E402

from students.text_preprocessing import subword_tokens, word_tokens  # noqa: E402
from students.training_data import load_pool, teacher_probs  # noqa: E402

SOFT_MIN_PROB = 0.2
SOFT_COPIES = 5


def tokenize(texts: list[str], mode: str) -> list[str]:
    if mode == "subword":
        return subword_tokens(texts)
    return [word_tokens(t) for t in texts]


def training_lines(labels: str, tokens: str) -> list[str]:
    sources = ("emmediatopic",) if labels == "gpt" else ("emmediatopic", "ccnews")
    df = load_pool(sources)
    docs = tokenize(df["text"].tolist(), tokens)
    lines = []
    if labels == "gpt":
        for doc, y in zip(docs, df["gpt_label_id"]):
            lines.append(f"__label__{y} {doc}")
        return lines
    probs = teacher_probs(df)
    for doc, p in zip(docs, probs):
        lines.append(f"__label__{int(p.argmax())} {doc}")
        if labels == "teacher_soft":
            for label_id in np.flatnonzero(p >= SOFT_MIN_PROB):
                lines.extend([f"__label__{label_id} {doc}"] * int(round(p[label_id] * SOFT_COPIES)))
    return lines


def train(labels: str, tokens: str, name: str, dim: int, epoch: int, lr: float,
          word_ngrams: int, quantize: bool, threads: int, seed: int = 0) -> Path:
    lines = training_lines(labels, tokens)
    rng = np.random.default_rng(seed)
    rng.shuffle(lines)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf8") as f:
        f.write("\n".join(lines) + "\n")
        train_path = f.name
    print(f"training fastText on {len(lines)} lines", flush=True)
    model = fasttext.train_supervised(
        input=train_path, dim=dim, epoch=epoch, lr=lr, wordNgrams=word_ngrams,
        minn=2 if tokens == "word" else 0, maxn=5 if tokens == "word" else 0,
        bucket=2_000_000, loss="softmax", thread=threads, seed=seed, verbose=1,
    )
    if quantize:
        model.quantize(input=train_path, qnorm=True, retrain=True, cutoff=300_000, thread=threads)
    Path(train_path).unlink()

    out = MODELS_DIR / "fasttext" / f"{name}.{'ftz' if quantize else 'bin'}"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({"tokens": tokens, "labels": labels}, f)
    return out


class FastTextPredictor:
    def __init__(self, path: str):
        path = Path(path)
        fasttext.FastText.eprint = lambda *args, **kwargs: None
        self.model = fasttext.load_model(str(path))
        self.tokens = json.loads(path.with_suffix(".json").read_text())["tokens"]
        tokenize(["warm-up"], self.tokens)  # load the subword tokenizer now, so RAM is measured
        self.size_mb = path.stat().st_size / 2**20
        self.num_parameters = None

    def predict(self, texts):
        docs = tokenize(texts, self.tokens)
        labels, _ = self.model.predict(docs, k=1)
        return np.array([int(lab[0].removeprefix("__label__")) for lab in labels])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", choices=["gpt", "teacher", "teacher_soft"], default="teacher")
    parser.add_argument("--tokens", choices=["word", "subword"], default="subword")
    parser.add_argument("--name", required=True)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--word-ngrams", type=int, default=2)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    out = train(args.labels, args.tokens, args.name, args.dim, args.epoch, args.lr,
                args.word_ngrams, args.quantize, args.threads)
    print(f"saved {out} ({out.stat().st_size / 2**20:.1f} MB); labels: {len(LABELS)}")


if __name__ == "__main__":
    main()
