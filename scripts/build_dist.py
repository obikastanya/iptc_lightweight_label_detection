"""Assemble the release folders:

    dist/github/        clean repository (package, scripts, notebook, docs, test-set manifest)
    dist/huggingface/   model repositories ready for `huggingface-cli upload`
        statickd-iptc-multilingual/          StaticKD (3-seed table ensemble)
        statickd-iptc-multilingual-compact/  StaticKD-compact (pruned tokenizer, lower RAM)

Usage (from the project root, after the notebook has been executed):
    python scripts/build_dist.py
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

from statickd.config import LABELS  # noqa: E402

DIST = ROOT / "dist"
GH = DIST / "github"
HF = DIST / "huggingface"
TEMPLATES = ROOT / "scripts" / "dist_templates"
NUMBERS = json.loads((ROOT / "results/final/paper_numbers.json").read_text(encoding="utf8"))


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints"))


def normalise_newlines(folder: Path) -> None:
    for f in folder.rglob("*"):
        if f.suffix in {".py", ".md", ".txt", ".toml", ".cfg", ".json", ".csv", ".cff"} and f.is_file():
            data = f.read_bytes()
            if b"\r\n" in data:
                f.write_bytes(data.replace(b"\r\n", b"\n"))


def text_hash(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).encode("utf8")).hexdigest()


def build_github() -> None:
    if GH.exists():
        shutil.rmtree(GH)
    GH.mkdir(parents=True)
    copy_tree(ROOT / "src", GH / "src")
    (GH / "scripts").mkdir()
    for f in ["train_statickd.py", "train_baselines.py", "train_transformers.py", "evaluate_model.py",
              "label_with_teacher.py", "build_dist.py", "verify_testset.py", "validity_silver.py", "run_benchmarks.py", "check_paper.py"]:
        if (ROOT / "scripts" / f).exists():
            shutil.copy2(ROOT / "scripts" / f, GH / "scripts" / f)
    (GH / "notebooks").mkdir()
    shutil.copy2(ROOT / "notebooks" / "StaticKD_IPTC_final.ipynb", GH / "notebooks" / "StaticKD_IPTC_final.ipynb")
    (GH / "docs").mkdir()
    shutil.copy2(ROOT / "docs" / "RANCANGAN_PENELITIAN.md", GH / "docs" / "RANCANGAN_PENELITIAN.md")
    for f in TEMPLATES.glob("github_*"):
        shutil.copy2(f, GH / f.name.removeprefix("github_"))
    # test-set manifest: identifiers, labels and a hash of the text -- no copyrighted article text
    test = pd.read_json(ROOT / "data/processed/testset_jsonld/unified_test.jsonl", lines=True)
    manifest = pd.DataFrame({"doc_id": test["doc_id"], "source": test["source"], "label_origin": test["label_origin"],
                             "lang": test["lang"], "label": test["label"], "label_id": test["label_id"],
                             "source_label": test["source_label"], "url": test["url"], "domain": test["domain"],
                             "text_sha1": test["text"].map(text_hash), "n_words": test["text"].str.split().str.len()})
    out = GH / "data" / "testset"
    out.mkdir(parents=True)
    manifest.to_csv(out / "unified_test_manifest.csv", index=False, encoding="utf8")
    shutil.copy2(ROOT / "src/statickd/testset/iptc_reference/mapping_table.csv", out / "iptc_mapping_table.csv")
    for f in ["unified_funnel.json", "silver_funnel.json"]:
        shutil.copy2(ROOT / "data/processed/testset_jsonld" / f, out / f)
    results = GH / "results"
    results.mkdir()
    for f in ["paper_numbers.json", "cpu_benchmark.jsonl", "environment.json"]:
        if (ROOT / "results/final" / f).exists():
            shutil.copy2(ROOT / "results/final" / f, results / f)
    normalise_newlines(GH)
    print(f"github -> {GH} ({sum(1 for _ in GH.rglob('*') if _.is_file())} files)")


def export_model(model_dir: Path, out: Path, card: str, variant: str) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    data = np.load(model_dir / "model.npz")
    save_file({"token_table": np.ascontiguousarray(data["token_table"]), "bias": data["bias"].astype(np.float32)},
              str(out / "model.safetensors"), metadata={"format": "statickd-token-table", "labels": json.dumps(LABELS)})
    shutil.copy2(model_dir / "tokenizer.json", out / "tokenizer.json")
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf8"))
    config = {"model_type": "statickd", "variant": variant, "format": "token_table_fp16", "num_labels": len(LABELS),
              "id2label": dict(enumerate(LABELS)), "label2id": {l: i for i, l in enumerate(LABELS)},
              "vocab_size": int(data["token_table"].shape[0]), "max_tokens": int(meta.get("max_tokens", 1024)),
              "teacher": "classla/multilingual-IPTC-news-topic-classifier",
              "base_embeddings": "minishlab/potion-multilingual-128M",
              "ensemble_of": meta.get("ensemble_of", meta.get("config", {}).get("name")),
              "training": {k: v for k, v in meta.get("config", {}).items() if k not in ("extra",)}}
    (out / "config.json").write_text(json.dumps(config, indent=1, default=str), encoding="utf8")
    shutil.copy2(TEMPLATES / "hf_statickd_inference.py", out / "statickd_inference.py")
    (out / "README.md").write_text(card, encoding="utf8")
    normalise_newlines(out)
    print(f"huggingface -> {out}")


def model_card(variant: str) -> str:
    tpl = (TEMPLATES / "hf_README.md").read_text(encoding="utf8")
    n = NUMBERS

    def g(key, digits=3):
        v = n.get(key)
        return "–" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))

    m = "statickd" if variant == "full" else "statickd_compact"
    values = {
        "VARIANT": "StaticKD" if variant == "full" else "StaticKD-compact",
        "REPO": "statickd-iptc-multilingual" + ("" if variant == "full" else "-compact"),
        "DEV_MACRO": g(f"{m}.dev.macro"), "TEST_MACRO": g(f"{m}.test.macro"), "TEST_ACC": g(f"{m}.test.acc"),
        "TEST_LANGMACRO": g(f"{m}.test.langmacro"), "AGREE": g(f"{m}.heldout.acc"),
        "T_DEV": g("teacher.dev.macro"), "T_TEST": g("teacher.test.macro"), "T_ACC": g("teacher.test.acc"),
        "DPS1": g(f"eff.{m}.docs_per_sec_1c", 0), "DPS4": g(f"eff.{m}.docs_per_sec_4c", 0),
        "LAT": g(f"eff.{m}.latency_ms_p50_1c", 2), "RAM": g(f"eff.{m}.ram", 0), "SIZE": g(f"eff.{m}.size", 0),
        "T_DPS1": g("eff.teacher_onnx_int8.docs_per_sec_1c", 2), "T_LAT": g("eff.teacher_onnx_int8.latency_ms_p50_1c", 0),
        "TS_DOCS": g("ts.docs", 0), "TS_LANGS": g("ts.langs", 0), "POOL_DOCS": g("data.pool.docs", 0),
        "POOL_LANGS": g("data.pool.langs", 0),
        "VARIANT_NOTE": "" if variant == "full" else (
            "- **Compact variant.** The tokenizer keeps only pieces seen at least 10 times in the distillation data. "
            "This lowers RAM, but languages that are (almost) absent from the distillation data (e.g. Bengali, "
            "Punjabi, Odia, Malayalam) collapse to near-zero accuracy. Use the full model for those languages."),
    }
    for k, v in values.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    return tpl


def main() -> None:
    build_github()
    final = ROOT / "models" / "final"
    export_model(final / "statickd", HF / "statickd-iptc-multilingual", model_card("full"), "full")
    export_model(final / "statickd_compact", HF / "statickd-iptc-multilingual-compact", model_card("compact"), "compact")


if __name__ == "__main__":
    main()
