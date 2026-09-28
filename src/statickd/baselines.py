"""Baselines: fastText, TF-IDF + linear SVM, small transformer students, and the teacher on CPU.

Every predictor exposes `logits(texts) -> (n, 17)` in config.LABELS order, `size_mb` and
`num_parameters`, so that accuracy and CPU benchmarks treat all models identically.

| spec                    | model                                                               |
|-------------------------|---------------------------------------------------------------------|
| teacher-pytorch         | XLM-R-large teacher, PyTorch fp32 (as in its model card)            |
| teacher-onnx-int8       | the same model, ONNX Runtime int8 export by onnx-community          |
| onnx:<dir>              | a transformer student exported to ONNX (fp32)                       |
| fasttext:<file>         | a fastText student                                                  |
| tfidf:<file>            | TF-IDF (XLM-R subword tokens, uni+bigrams) + linear SVM             |
| static:<dir>            | a StaticKD student                                                  |
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

import statickd.config as config  # noqa: F401  (sets HF_HOME)
from statickd.config import LABEL2ID, LABELS, MODELS_DIR, NUM_LABELS, TEACHER_MODEL_ID, TEACHER_ONNX_ID

WORD_RE = re.compile(r"\w+", re.UNICODE)


def dir_size_mb(path) -> float:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 2**20
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20


# ----------------------------------------------------------------------------- tokenisation
def word_tokens(text: str) -> str:
    """Lower-cased word tokens (the classic fastText setting; weak for scripts without spaces)."""
    return " ".join(WORD_RE.findall(text.lower()))


@lru_cache(maxsize=None)
def subword_tokenizer(name: str = "xlm-roberta-base"):
    """XLM-R SentencePiece tokenizer: language-agnostic, also segments Chinese, Japanese, Thai."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    return Tokenizer.from_file(hf_hub_download(name, "tokenizer.json"))


def subword_tokens(texts: list[str]) -> list[str]:
    encs = subword_tokenizer().encode_batch([t.lower() for t in texts], add_special_tokens=False)
    return [" ".join(e.tokens) for e in encs]


def tokenize(texts: list[str], mode: str) -> list[str]:
    return subword_tokens(texts) if mode == "subword" else [word_tokens(t) for t in texts]


# ----------------------------------------------------------------------------- fastText
def train_fasttext(texts: list[str], labels: np.ndarray, out: Path, tokens: str = "subword",
                   dim: int = 100, epoch: int = 15, lr: float = 0.5, word_ngrams: int = 2,
                   quantize: bool = False, threads: int = 8, seed: int = 0) -> Path:
    """Supervised fastText (Joulin et al., 2017) on hard labels (GPT-4o or teacher arg-max)."""
    import fasttext

    docs = tokenize(texts, tokens)
    lines = [f"__label__{int(y)} {d}" for d, y in zip(docs, labels)]
    np.random.default_rng(seed).shuffle(lines)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf8") as f:
        f.write("\n".join(lines) + "\n")
        train_path = f.name
    model = fasttext.train_supervised(
        input=train_path, dim=dim, epoch=epoch, lr=lr, wordNgrams=word_ngrams,
        minn=2 if tokens == "word" else 0, maxn=5 if tokens == "word" else 0,
        bucket=2_000_000, loss="softmax", thread=threads, seed=seed, verbose=0)
    if quantize:
        model.quantize(input=train_path, qnorm=True, retrain=True, cutoff=300_000, thread=threads)
    os.unlink(train_path)
    out = Path(out).with_suffix(".ftz" if quantize else ".bin")
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))
    out.with_suffix(".json").write_text(json.dumps({"tokens": tokens, "quantized": quantize}))
    return out


class FastTextPredictor:
    def __init__(self, path):
        import fasttext

        path = Path(path)
        fasttext.FastText.eprint = lambda *a, **k: None
        self.model = fasttext.load_model(str(path))
        self.tokens = json.loads(path.with_suffix(".json").read_text())["tokens"]
        tokenize(["warm-up"], self.tokens)  # load the subword tokenizer now (RAM is measured)
        self.size_mb = dir_size_mb(path)
        self.num_parameters = None

    def logits(self, texts):
        docs = tokenize(texts, self.tokens)
        out = np.full((len(docs), NUM_LABELS), -30.0, dtype=np.float32)
        for i, d in enumerate(docs):  # f.predict: NumPy-2-safe path of the fasttext bindings
            for p, lab in self.model.f.predict(d.replace("\n", " "), -1, 0.0, "strict"):
                out[i, int(lab.removeprefix("__label__"))] = np.log(max(p, 1e-12))
        return out

    def predict(self, texts):
        docs = [d.replace("\n", " ") for d in tokenize(texts, self.tokens)]
        return np.array([int(self.model.f.predict(d, 1, 0.0, "strict")[0][1].removeprefix("__label__"))
                         for d in docs])


# ----------------------------------------------------------------------------- TF-IDF + linear SVM
def train_tfidf(texts: list[str], labels: np.ndarray, out: Path, max_features: int = 1_000_000,
                c: float = 0.5, seed: int = 0) -> Path:
    """TF-IDF over lower-cased XLM-R subword uni+bigrams and a linear SVM, the non-neural baseline of
    Kuzman & Ljubesic (TF-IDF + SVC), here trained on the same teacher labels as the students."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, ngram_range=(1, 2),
                          min_df=3, max_features=max_features, sublinear_tf=True, dtype=np.float32)
    x = vec.fit_transform(subword_tokens(texts))
    clf = LinearSVC(C=c, random_state=seed).fit(x, labels)
    out = Path(out).with_suffix(".joblib")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vec, "classifier": clf}, out)
    return out


class TfidfPredictor:
    def __init__(self, path):
        import joblib

        obj = joblib.load(path)
        self.vec, self.clf = obj["vectorizer"], obj["classifier"]
        subword_tokenizer()
        self.size_mb = dir_size_mb(path)
        self.num_parameters = int(self.clf.coef_.size + self.clf.intercept_.size)

    def logits(self, texts):
        scores = self.clf.decision_function(self.vec.transform(subword_tokens(texts)))
        out = np.full((len(texts), NUM_LABELS), -1e4, dtype=np.float32)
        out[:, self.clf.classes_] = scores
        return out

    def predict(self, texts):
        return self.logits(texts).argmax(1)


# ----------------------------------------------------------------------------- transformers on CPU
class TeacherPyTorch:
    def __init__(self, threads: int = 1):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        path = snapshot_download(TEACHER_MODEL_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        self.order = np.array([LABEL2ID[self.model.config.id2label[i]] for i in range(NUM_LABELS)])
        self.size_mb = dir_size_mb(Path(path) / "model.safetensors")
        self.num_parameters = sum(p.numel() for p in self.model.parameters())

    def logits(self, texts):
        with self.torch.inference_mode():
            enc = self.tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
            z = self.model(**enc).logits.numpy()
        out = np.empty_like(z)
        out[:, self.order] = z
        return out


class OnnxClassifier:
    """ONNX Runtime sequence classifier on CPU (int8 teacher, transformer students)."""

    def __init__(self, model_dir, onnx_file: str, threads: int = 1, max_length: int = 512):
        import onnxruntime as ort
        from transformers import AutoConfig, AutoTokenizer

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # busy-waiting worker threads made multi-threaded runs slower on the hybrid benchmark CPU
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        onnx_path = Path(model_dir) / onnx_file
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        cfg = AutoConfig.from_pretrained(model_dir)
        self.order = np.array([LABEL2ID[cfg.id2label[i]] for i in range(NUM_LABELS)])
        self.max_length = max_length
        self.size_mb = dir_size_mb(onnx_path) + (dir_size_mb(str(onnx_path) + "_data")
                                                 if Path(str(onnx_path) + "_data").exists() else 0.0)
        self.num_parameters = None

    def logits(self, texts):
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="np")
        feeds = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
        z = self.session.run(None, feeds)[0]
        out = np.empty_like(z)
        out[:, self.order] = z
        return out


def teacher_onnx_int8(threads: int = 1) -> OnnxClassifier:
    from huggingface_hub import snapshot_download

    path = snapshot_download(TEACHER_ONNX_ID, allow_patterns=["*.json", "onnx/model_quantized.onnx"])
    return OnnxClassifier(path, "onnx/model_quantized.onnx", threads)


def load_predictor(spec: str, threads: int = 1):
    """Build a CPU predictor from a spec string (see the module docstring)."""
    kind, _, path = spec.partition(":")
    if kind == "teacher-pytorch":
        return TeacherPyTorch(threads)
    if kind == "teacher-onnx-int8":
        return teacher_onnx_int8(threads)
    if kind == "onnx":
        meta = json.loads((Path(path) / "train_meta.json").read_text()) if (Path(path) / "train_meta.json").exists() else {}
        return OnnxClassifier(path, "model.onnx", threads, max_length=int(meta.get("max_length", 256)))
    if kind == "fasttext":
        return FastTextPredictor(path)
    if kind == "tfidf":
        return TfidfPredictor(path)
    if kind == "static":
        from statickd.student import StaticKDModel
        return StaticKDModel(path)
    raise ValueError(f"unknown model spec: {spec}")


BASELINES_DIR = MODELS_DIR / "baselines"
