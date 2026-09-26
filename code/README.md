# Lightweight multilingual IPTC news topic classification — research code

Distils the XLM-R-large IPTC classifier of Kuzman & Ljubešić (2025)
(`classla/multilingual-IPTC-news-topic-classifier`) into CPU-friendly students.
All commands are run from this `code/` directory with the project venv (`../env/Scripts/python.exe`).

## Layout

| Path | Purpose |
|---|---|
| `config.py` | paths, the 17 IPTC labels, preprocessing constants (first 512 words, as in the paper) |
| `data/emmediatopic.py` | loader for EMMediaTopic 1.0 (GPT-4o labels; `dev` split = clean held-out set) |
| `data/download_ccnews.py` | multilingual unlabeled news corpus (CC-News, ~70 languages, domain-disjoint held-out) |
| `data/expand_ccnews_train.py` | extra CC-News training docs for the data-scaling experiment |
| `data/build_llm_test.py` | 320-doc, 16-language test set annotated by an LLM with Kuzman's GPT-4o prompt |
| `teacher/teacher.py`, `teacher/pseudo_label.py` | teacher wrapper; stores the teacher's 17 logits for every corpus |
| `students/fasttext_student.py` | fastText baselines (GPT labels / teacher labels, word / subword tokens) |
| `students/static_student.py` | **proposed method**: static multilingual embeddings + mean pooling + linear head, trained by KD; NumPy inference; `export_table` folds it into a 17-logit-per-token table |
| `students/distill_static_embeddings.py` | static embeddings distilled from the teacher encoder (ablation) |
| `students/prune_static.py` | vocabulary pruning of the embedding table (ablation) |
| `students/compact_tokenizer.py` | prunes the tokenizer vocabulary itself (RAM 505 → 281 MB) |
| `students/transformer_student.py` | mMiniLMv2-L6 KD student + ONNX int8 export (middle-ground reference) |
| `evaluation/evaluate_accuracy.py` | macro/micro-F1 on dev, LLM test, CC-News held-out (teacher agreement) |
| `evaluation/benchmark_cpu.py`, `run_cpu_benchmarks.py` | CPU-only latency / throughput / RAM at 1 and 4 threads |
| `evaluation/confidence_analysis.py` | accuracy vs coverage and student→teacher cascade |
| `evaluation/summarize.py` | joins everything into `../results/summary.md` |

## Reproduce

```bash
python data/download_ccnews.py --per-lang 3000 --heldout 300
python teacher/pseudo_label.py --corpora emmediatopic ccnews_heldout ccnews_train
python data/expand_ccnews_train.py --extra 5000
python teacher/pseudo_label.py --corpora ccnews_train_extra
python data/build_llm_test.py sample        # then annotate batches, then:
python data/build_llm_test.py merge

python students/static_student.py --name static_final --sources emmediatopic ccnews ccnews_extra \
       --token-dropout 0.2 --epochs 6 --keep-all-vocab
python -c "from students.static_student import export_table; export_table('../models/static/static_final')"
python students/compact_tokenizer.py --model ../models/static/static_final_table --min-count 10   # low-RAM variant

python evaluation/evaluate_accuracy.py --model static:../models/static/static_final_table --name static_final_table
python evaluation/run_cpu_benchmarks.py static_final_table=static:../models/static/static_final_table
python evaluation/summarize.py
```

## Using the final model

```python
from students.static_student import StaticPredictor
from config import LABELS

model = StaticPredictor("../models/static/static_final_table")
print([LABELS[i] for i in model.predict(["Pemerintah menaikkan harga BBM mulai pekan depan ..."])])
```
Only `numpy` and `tokenizers` are needed at inference time.
