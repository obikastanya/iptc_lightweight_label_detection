# StaticKD: distilling a multilingual IPTC news topic classifier into a static token-logit table for lightweight CPU inference

StaticKD distils the multilingual XLM-RoBERTa-large IPTC Media Topic classifier of Kuzman & Ljubešić
(IEEE Access 2025) into a student that, after training, is **a single table of 17 logits per token**.
Classifying an article means tokenising it, averaging the table rows of its tokens and adding a bias.
This needs only NumPy and `tokenizers`, runs at hundreds of articles per second on one CPU core, and
works across dozens of languages.

This repository contains:
- the research code (`src/statickd`);
- the scripts that run each heavy step (`scripts/`);
- the notebook that reproduces every number, table and figure of the paper
  (`notebooks/StaticKD_IPTC_final.ipynb`, explanations in Indonesian);
- the manifest of the 47-language unified IPTC test set (`data/testset/`).

## Quick start (inference)

```bash
pip install numpy tokenizers safetensors
```

```python
import sys; sys.path.insert(0, "src")
from statickd import StaticKDModel

model = StaticKDModel.load("models/final/statickd")      # or the Hugging Face folder, see below
model.predict_labels(["Bank Indonesia menaikkan suku bunga acuan untuk menahan inflasi."])
# ['economy, business and finance']
```

The released models are on Hugging Face as `statickd-iptc-multilingual` and
`statickd-iptc-multilingual-compact`. Each ships with a standalone `statickd_inference.py`.

## Repository layout

| path | content |
|---|---|
| `src/statickd/config.py` | paths, the 17 IPTC labels, preprocessing (title + body, first 512 words) |
| `src/statickd/data.py` | loaders: EMMediaTopic, distillation pool, CC-News held-out, unified test set |
| `src/statickd/corpus/` | multilingual CC-News 2019–2021 distillation corpus (domain-disjoint held-out) |
| `src/statickd/teacher.py` | the XLM-R-large teacher and soft-label annotation |
| `src/statickd/student.py` | **StaticKD**: training, folding into a token table, ensembling, compact tokenizer, NumPy inference |
| `src/statickd/experiments.py` | registry of all 72 StaticKD runs (ablations, temperatures, seeds) |
| `src/statickd/baselines.py`, `transformer_student.py` | fastText, TF-IDF + SVM, MiniLM-L6 / DistilmBERT KD, teacher on CPU |
| `src/statickd/testset/` | unified test set: CC-NEWS WARC download, JSON-LD extraction, IPTC-grounded mapping, public datasets, filters |
| `src/statickd/evaluation.py` | metrics, site-cluster bootstrap, paired tests, McNemar, Holm, calibration |
| `src/statickd/benchmark.py` | CPU-only latency / throughput / RAM benchmark (P-core pinning) |
| `src/statickd/report.py` | writes numbers and tables into the LaTeX paper (`\val{key}` macros) |
| `scripts/` | command-line entry points for every heavy step |
| `notebooks/StaticKD_IPTC_final.ipynb` | the full study, step by step |
| `data/testset/` | test-set manifest (URL/ID, label, label origin, text hash), IPTC mapping table, construction funnel |
| `results/` | the paper's numbers (`paper_numbers.json`), CPU benchmark, software environment |

## Reproducing the study

```bash
python -m venv env && env/Scripts/activate        # Linux/macOS: source env/bin/activate
pip install -r requirements-research.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124   # training / teacher labelling on GPU
```

Large inputs are downloaded by the scripts:
- EMMediaTopic 1.0 from CLARIN.SI `11356/1991`; place `EMMediaTopic-1.0.jsonl` in `data/raw/`;
- CC-News 2019–2021 from Hugging Face `CloverSearch/cc-news-mutlilingual`;
- CC-NEWS 2024–2026 WARC files from Common Crawl;
- the public test datasets.

The notebook's `RUN_*` switches recompute each stage from scratch. With all switches off, it rebuilds
every table and figure from the stored predictions.

| step | command | time (RTX 4050 + Core Ultra 7) |
|---|---|---|
| distillation corpus | `python -m statickd.corpus.ccnews && python -m statickd.corpus.expand` | ~1 h |
| teacher soft labels | `python scripts/label_with_teacher.py` | ~1.5 h (GPU) |
| unified test set | `python -m statickd.testset.ccnews_warc`, `…jsonld`, `…silver`, `…external`, `…unified` | ~6 h, 34 GB download |
| StaticKD (72 runs) | `python scripts/train_statickd.py` | ~3 h (GPU) |
| baselines | `python scripts/train_baselines.py`, `python scripts/train_transformers.py` | ~4 h |
| teacher predictions | `python scripts/evaluate_model.py --name teacher --spec teacher-gpu` | ~15 min (GPU) |
| CPU benchmark | `python -m statickd.benchmark --model static:models/final/statickd --name statickd --threads 1` | minutes per model |

Set `PYTHONPATH=src` or run from the repository root (the scripts add `src/` themselves).

## Unified test set

The test set has 12,744 articles in 47 languages covering the 17 IPTC labels. Its labels are
independent of the teacher, and each article comes from one of these sources:

- **CC-NEWS 2024–2026.** schema.org `articleSection` / breadcrumb names are mapped to the IPTC Media
  Topic tree. A name is mapped only when everything published under it belongs to one top-level topic.
- **Public datasets.** MasakhaNEWS, MN-DS, L3Cube-IndicNews, Bangla news (PLOS ONE) and IndoNews.

All candidates go through the same filters:
- language check with fastText lid.176;
- minimum length of 75 words;
- MinHash near-duplicate removal across all sources;
- removal of articles with ≥ 30% word 8-gram overlap with any training document.

Because of copyright, `data/testset/unified_test_manifest.csv` contains identifiers, labels and a SHA-1
of the normalised text, but not the text itself. To rebuild the test set, run the construction
pipeline and check the result with `python scripts/verify_testset.py`.

## Licence

- **Code:** MIT.
- **Models:** CC BY-SA 4.0, inherited from the teacher model.
- **Test-set manifest:** CC BY-SA 4.0. The source datasets keep their own licences; MasakhaNEWS, for
  example, is CC BY-NC 4.0.

## Citation

See `CITATION.cff`.
