# Silver IPTC test set from CC-NEWS 2024–2026 (JSON-LD)

Builds a multilingual news test set whose IPTC labels come from the **publisher's own section
names** in schema.org JSON-LD (`articleSection`, `BreadcrumbList`). These are **silver labels**:
precise for unambiguous sections, but not a human annotation. Check `audit_sample.csv` before
treating them as gold.

## Steps

Run from `code/`. The raw WARC files come from `python data/download_ccnews_warc.py`.

```bash
python testset_jsonld/extract_jsonld.py --workers 10          # 1. WARC -> JSON-LD article records
python testset_jsonld/build_testset.py --per-cell 50 --per-domain 5 --min-lang-docs 100   # 2. label + filter + sample
```

| File | Role |
|---|---|
| `extract_jsonld.py` | Parses every `<script type="application/ld+json">`. Keeps NewsArticle/Article pages from sites **not** in the training/held-out pools. Stores section, breadcrumb, keywords, dates, author, publisher, paywall flag and text (JSON-LD `articleBody`, else trafilatura). Language is `inLanguage`, or fastText `lid.176` when that is missing. |
| `iptc_section_map.py` | Hand-curated, high-precision section-name → IPTC table (17 labels, multilingual). Mixed sections (cronaca, sucesos, güncel, общество, entertainment, lifestyle, …) are deliberately not mapped. Conflicting names reject the article. |
| `build_testset.py` | Labels articles, then applies the filters: 2024+ publication date, length, paywall, inLanguage/fastText agreement, URL/text/MinHash dedup, and < 30% 8-gram overlap with any training or evaluation corpus. Finally samples with per-(language, label) and per-site caps. |

## Outputs (`data/processed/testset_jsonld/`)

- `extracted/*.parquet`: one shard per WARC file, with all JSON-LD news articles from new sites.
- `pool.parquet`: every confidently labelled article that passed all filters.
- `silver_test.jsonl`: the sampled test set. Fields: `doc_id`, `lang`, `label`, `label_id`, `text` (title + body, first 512 words), `url`, `domain`, `date`, `section_names`, and more.
- `audit_sample.csv`: random articles per label, with empty `human_label` / `correct` columns for a manual check.
- `report.md`: filtering funnel, language × label table, and the section names behind each label.

fastText note: `fasttext-wheel` 0.9.2 `predict()` fails with NumPy ≥ 2, so language ID uses the low-level `model.f.predict(text, k, threshold, "strict")`.
