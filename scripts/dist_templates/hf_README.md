---
license: cc-by-sa-4.0
library_name: numpy
pipeline_tag: text-classification
language:
- multilingual
tags:
- news
- topic-classification
- iptc
- knowledge-distillation
- static-embeddings
- model2vec
base_model:
- minishlab/potion-multilingual-128M
- classla/multilingual-IPTC-news-topic-classifier
---

# {{VARIANT}}: lightweight multilingual IPTC news topic classification on CPU

{{VARIANT}} classifies a news article (title + body, first 512 words) into one of the 17 top-level
[IPTC Media Topics](https://www.iptc.org/std/NewsCodes/treeview/mediatopic/mediatopic-en-GB.html).
The whole model is **one table of 17 logits per token**. Inference means tokenising, averaging table
rows and adding a bias. It needs only `numpy` + `tokenizers` + `safetensors` and runs at
**{{DPS1}} articles/s on one CPU core** (p50 latency {{LAT}} ms, {{SIZE}} MB on disk, {{RAM}} MB RAM).

It was trained by knowledge distillation from the XLM-RoBERTa-large IPTC classifier of Kuzman &
Ljubešić (2025, `classla/multilingual-IPTC-news-topic-classifier`) on {{POOL_DOCS}} unlabelled news
articles in {{POOL_LANGS}} languages. The student's token embeddings were initialised from
`minishlab/potion-multilingual-128M`, and its linear head was folded into the token table.

## Usage

```python
from statickd_inference import StaticKD   # file in this repository

model = StaticKD(".")
model.predict(["The central bank raised its benchmark interest rate by 50 basis points on Tuesday ..."])
# ['economy, business and finance']
model.predict_proba(["..."])               # (n, 17) probabilities, label order = config.json id2label
```

## Results

All numbers come from the paper's notebook.
- **Unified test set**: {{TS_DOCS}} news articles in {{TS_LANGS}} languages, published 2024–2026 or
  drawn from public datasets. Labels are independent of the teacher: human annotations or publisher
  categories mapped to IPTC.
- **dev**: EMMediaTopic, 4 languages, GPT-4o labels.

| model | dev macro-F1 | test macro-F1 | test accuracy | articles/s, 1 CPU core |
|---|---|---|---|---|
| teacher XLM-R-large | {{T_DEV}} | {{T_TEST}} | {{T_ACC}} | {{T_DPS1}} (ONNX int8) |
| **{{VARIANT}}** | **{{DEV_MACRO}}** | **{{TEST_MACRO}}** | **{{TEST_ACC}}** | **{{DPS1}}** |

Further figures: mean per-language macro-F1 on the test set is {{TEST_LANGMACRO}}, and agreement with
the teacher on 68-language held-out CC-News is {{AGREE}}. Throughput on 4 cores is {{DPS4}} articles/s.

## Intended use and limitations

- **Intended use.** A fast first-stage topic filter for multilingual news pipelines on CPU-only
  servers. The softmax confidence can route uncertain articles to the teacher (cascade).
- **Mistakes follow the teacher.** The model is a distilled student, so it reproduces the teacher's
  errors and biases.
- **Weakest labels.** Accuracy is lowest on context-dependent labels (*lifestyle and leisure*,
  *society*, *human interest*).
- **Weakest languages.** Accuracy is also lowest for languages with little distillation data.
- **Scope.** Single-label only, top level of IPTC only.
{{VARIANT_NOTE}}

## Licence

CC BY-SA 4.0, inherited from the teacher model. The base static embeddings are MIT-licensed.

## Citation

Kastanya, O., & Alwy, A. D. P. (2026). *StaticKD: Knowledge Distillation Pengklasifikasi Topik Berita IPTC Multibahasa ke Tabel Logit Token Statis
untuk Inferensi Ringan di CPU*. Draft.
