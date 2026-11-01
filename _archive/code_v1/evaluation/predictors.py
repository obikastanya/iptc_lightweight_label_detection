"""Uniform CPU predictors for every model compared in this study.

A model is referenced by a spec string `kind[:path]`, e.g.
    teacher-pytorch            Kuzman's XLM-R-large in PyTorch fp32
    teacher-onnx-int8          the same model exported to ONNX and int8-quantised (onnx-community)
    fasttext:models/x.bin      a fastText student
    static:models/static_x     a static-embedding student (see students/static_student.py)
    onnx:models/minilm         a small transformer student exported to ONNX (fp32, 256 tokens)
    onnx-int8:models/minilm    the same student with dynamic int8 quantisation
Every predictor exposes `predict(texts) -> np.ndarray[label_id]` in config.LABELS order,
`size_mb` and `num_parameters`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABEL2ID, LABELS, TEACHER_MODEL_ID  # noqa: E402  (sets HF_HOME)

import numpy as np  # noqa: E402

from evaluation.speed import dir_size_mb  # noqa: E402

TEACHER_ONNX_ID = "onnx-community/multilingual-IPTC-news-topic-classifier-ONNX"


class TeacherPyTorch:
    def __init__(self, threads: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from huggingface_hub import snapshot_download

        torch.set_num_threads(threads)
        self.torch = torch
        path = snapshot_download(TEACHER_MODEL_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        self.order = np.array([LABEL2ID[self.model.config.id2label[i]] for i in range(len(LABELS))])
        self.size_mb = dir_size_mb(Path(path) / "model.safetensors")
        self.num_parameters = sum(p.numel() for p in self.model.parameters())

    def predict(self, texts):
        with self.torch.inference_mode():
            enc = self.tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
            return self.order[self.model(**enc).logits.argmax(-1).numpy()]


class OnnxClassifier:
    """ONNX Runtime sequence classifier (used for the int8 teacher and transformer students)."""

    def __init__(self, model_dir: str, onnx_file: str, threads: int, max_length: int = 512):
        import onnxruntime as ort
        from transformers import AutoConfig, AutoTokenizer

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Busy-waiting worker threads made multi-threaded runs slower than single-threaded ones
        # on the hybrid benchmark CPU; disabling spinning gives stable scaling.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        onnx_path = str(Path(model_dir) / onnx_file)
        self.session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        config = AutoConfig.from_pretrained(model_dir)
        self.order = np.array([LABEL2ID[config.id2label[i]] for i in range(len(LABELS))])
        self.max_length = max_length
        self.size_mb = dir_size_mb(onnx_path)
        data_file = Path(onnx_path + "_data")
        if data_file.exists():
            self.size_mb += dir_size_mb(data_file)
        self.num_parameters = None

    def predict(self, texts):
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="np")
        feeds = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
        logits = self.session.run(None, feeds)[0]
        return self.order[logits.argmax(-1)]


def teacher_onnx_int8(threads: int) -> OnnxClassifier:
    from huggingface_hub import snapshot_download

    path = snapshot_download(TEACHER_ONNX_ID, allow_patterns=["*.json", "onnx/model_quantized.onnx"])
    return OnnxClassifier(path, "onnx/model_quantized.onnx", threads)


def load_predictor(spec: str, threads: int):
    kind, _, path = spec.partition(":")
    if kind == "teacher-pytorch":
        return TeacherPyTorch(threads)
    if kind == "teacher-onnx-int8":
        return teacher_onnx_int8(threads)
    if kind == "onnx":
        return OnnxClassifier(path, "model.onnx", threads, max_length=256)
    if kind == "onnx-int8":
        return OnnxClassifier(path, "model_int8.onnx", threads, max_length=256)
    if kind == "fasttext":
        from students.fasttext_student import FastTextPredictor
        return FastTextPredictor(path)
    if kind == "static":
        from students.static_student import StaticPredictor
        return StaticPredictor(path, threads)
    raise ValueError(f"Unknown model spec: {spec}")
