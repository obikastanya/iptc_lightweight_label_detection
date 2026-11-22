# Review round 3: fact sheet

Script: `scripts/review_experiments/r3_reviewer_checks.py` (no training; stored data and predictions only).
Machine-readable numbers: `results/review_experiments/r3_reviewer_checks.json`.
Bootstrap settings are the notebook's: CIs use 1,000 resamples (`N_BOOT`, part1.py:75), and paired tests use 2 x 1,000 = 2,000 resamples with `paired_bootstrap_seeds` and site clusters, seed 0. Model keys: MiniLM-L6 = `minilm_kd_t1` (part3.py:186 `key()`), StaticKD = `statickd`. Sanity check: the full-test values reproduce the paper (`sig.test.*` and `eqv.model2vec_clf.ci90` [-0.008, +0.008]).

## 1. Site overlap with the distillation data (M9)

How the domain is known per source (`statickd/testset/unified.py` `_row()`: `domain = urlparse(url).netloc`):
- ccnews_jsonld: the URL of the WARC record. Filter 1 already removed every host of ccnews_train, ccnews_train_extra and ccnews_heldout (exact host with and without `www.`, `testset/jsonld.py:47-52`).
- masakhanews: the dataset's URL field. 295 rows have no host: xh 200 (`not available`) and en 95 (relative BBC-style paths such as `/sport/canoeing/...`).
- mnds: the dataset's URL field. 45 rows point to `feedproxy.google.com`, so their real site is unknown.
- l3cube_indic, bangla_plos, indonews: no URL or site information, so overlap cannot be checked.
- EMMediaTopic train (20,000 docs): no URL or site (CLASSLA-web `document_id` only), so overlap with it cannot be checked. The test set shares two languages with it: ca 45 and el 308 articles, all from CC-NEWS.

Overlap is measured by exact host match after removing a leading `www.` against the CC-News train + extra pool (2,960 sites) and the held-out set:

| Source | n | Host known | In distillation | In held-out | Either | Same site and same language | Looser registrable-domain match |
|---|---|---|---|---|---|---|---|
| ccnews_jsonld | 6,050 | 6,050 | 0 | 0 | 0 | 0 | 605 (subdomains, e.g. m.jpnn.com, sports.ettoday.net, rbc.ua) |
| masakhanews | 3,339 | 3,044 | 2,394 (bbc.com) | 2,394 | 2,394 | 694 (sw 250, so 244, am 200) | 2,394 |
| mnds | 755 | 755 | 1 (realclearpolitics.com) | 29 (bbc.co.uk) | 30 | 1 | 61 |
| l3cube_indic | 2,361 | 0 | n/a | n/a | n/a | n/a | n/a |
| bangla_plos | 139 | 0 | n/a | n/a | n/a | n/a | n/a |
| indonews | 100 | 0 | n/a | n/a | n/a | n/a | n/a |
| **Total** | 12,744 | 9,849 | 2,395 | 2,423 | **2,424 (19.0%)** | 695 | 3,060 (24.0%) |

- No test URL occurs exactly in the distillation or held-out data (0 matches). The 8-gram contamination filter (`unified.py`, < 30%) also passed every bbc.com test article: their maximum overlap is 0.247 and the mean 0.0005.
- bbc.com appears in the distillation data in 14 languages (fa, sr, sw 400 each, cy 289, th 145, so 117, vi 84, my 83, gu 43, am 20, ...). Its MasakhaNEWS test articles share the language only for sw (400 distillation docs), so (117 + 33 held-out) and am (20 + 4 held-out).
- Scores of the MasakhaNEWS overlapping articles (2,394), as macro-F1/accuracy: teacher 0.706/0.607, StaticKD 0.611/0.507, Model2Vec 0.616/0.530, MiniLM 0.587/0.528. On the other MasakhaNEWS rows (945): teacher 0.656, StaticKD 0.391, Model2Vec 0.409, MiniLM 0.587. On the same-site, same-language rows (694): teacher 0.636, StaticKD 0.651, Model2Vec 0.659, MiniLM 0.554. The overlap is confounded with language, because the non-BBC rows are mostly languages without distillation data.

Test macro-F1 [95% site-cluster CI] and the paired difference StaticKD minus model [95% CI], Holm p over the paper's 14-model family:

| Subset | n (clusters, eff.) | Teacher | StaticKD | Model2Vec | MiniLM-L6 | Δ vs teacher | Δ vs Model2Vec | Δ vs MiniLM |
|---|---|---|---|---|---|---|---|---|
| Full test (paper) | 12,744 (1,227; 13.8) | 0.598 [0.546, 0.643] | 0.545 [0.490, 0.605] | 0.547 [0.495, 0.599] | 0.566 [0.511, 0.624] | -0.053 [-0.065, -0.033], p<0.001 | -0.002 [-0.009, +0.010], p=1.000 | -0.021 [-0.038, -0.008], p=0.005 |
| Without exact overlap | 10,320 (1,224; 17.7) | 0.626 [0.598, 0.650] | 0.575 [0.540, 0.616] | 0.572 [0.540, 0.610] | 0.604 [0.576, 0.631] | -0.050 [-0.065, -0.027], p<0.001 | +0.003 [-0.004, +0.012], p=0.696 | -0.028 [-0.041, -0.010], p<0.001 |
| Also without the 95 relative-URL MasakhaNEWS en rows | 10,225 | 0.627 | 0.576 [0.541, 0.616] | 0.573 | 0.605 | -0.051 [-0.066, -0.028], p<0.001 | +0.003 [-0.005, +0.012], p=0.786 | -0.028 [-0.042, -0.010], p<0.001 |
| Without registrable-domain overlap | 9,684 (1,058; 15.6) | 0.618 | 0.565 [0.528, 0.610] | 0.562 | 0.596 | -0.053 [-0.068, -0.029], p<0.001 | +0.003 [-0.005, +0.012], p=0.828 | -0.031 [-0.044, -0.011], p=0.005 |
| Verified-unseen only (host known and no overlap) | 6,789 (1,054; 152.2) | 0.632 [0.606, 0.654] | 0.589 [0.550, 0.626] | 0.587 [0.552, 0.621] | 0.612 [0.585, 0.637] | -0.044 [-0.063, -0.021], p<0.001 | +0.002 [-0.007, +0.012], p=0.630 | -0.024 [-0.042, -0.004], p=0.085 |

- Ranking: the teacher stays first and MiniLM-L6 second in every subset. StaticKD and the Model2Vec classifier swap places: Model2Vec is 0.002 ahead on the full test set and StaticKD is 0.002 to 0.003 ahead without the overlap. Neither difference is significant. In the verified-unseen subset, DistilmBERT (0.606) moves above StaticKD.
- **The StaticKD and Model2Vec equivalence at ±0.01 no longer holds without the overlap.** The 90% interval becomes [-0.003, +0.011] (TOST p=0.071), or [-0.003, +0.010] (p=0.059) when the relative-URL rows are also removed. Equivalence holds only at ±0.02.
- MiniLM's lead over StaticKD grows to 0.028 to 0.031 and stays significant, except in the verified-unseen subset (Holm p=0.085; raw p=0.017).
- The paper's test-set filter 1 ("site absent from both the distillation data and the held-out set") is applied to CC-NEWS only (main.tex:272). 2,394 MasakhaNEWS articles come from bbc.com, a distillation site.

## 2. Per-source macro-F1 with 95% CIs (M8b)

Site-cluster bootstrap with 1,000 resamples (`bootstrap_ci`, clusters = `site_clusters`). The document bootstrap is shown in brackets after "doc".

| Source | n | Clusters (eff.) | Teacher | StaticKD | Model2Vec | MiniLM-L6 |
|---|---|---|---|---|---|---|
| CC-NEWS | 6,050 | 1,092 (221.3) | 0.651 [0.624, 0.671] | 0.624 [0.596, 0.644] | 0.618 [0.590, 0.639] | 0.634 [0.607, 0.651] |
| MasakhaNEWS | 3,339 | 6 (1.9) | 0.695 [0.624, 0.711]; doc [0.679, 0.709] | 0.556 [0.315, 0.601]; doc [0.538, 0.574] | 0.562 [0.323, 0.606]; doc [0.544, 0.579] | 0.582 [0.567, 0.636]; doc [0.564, 0.600] |
| L3Cube-IndicNews | 2,361 | 1 | 0.709; doc [0.683, 0.733] | 0.639; doc [0.613, 0.663] | 0.644; doc [0.620, 0.668] | 0.680; doc [0.653, 0.704] |
| MN-DS | 755 | 127 (32.7) | 0.549 [0.496, 0.591] | 0.535 [0.479, 0.576] | 0.536 [0.478, 0.575] | 0.535 [0.482, 0.571] |
| Bangla news | 139 | 1 | 0.835; doc [0.771, 0.890] | 0.748; doc [0.678, 0.809] | 0.716; doc [0.641, 0.783] | 0.817; doc [0.739, 0.874] |
| IndoNews | 100 | 1 | 0.885; doc [0.833, 0.933] | 0.862; doc [0.803, 0.912] | 0.849; doc [0.784, 0.903] | 0.856; doc [0.794, 0.907] |

- The point estimates match Table S3.
- L3Cube, Bangla and IndoNews each form one cluster, so the site-cluster CI has zero width there. Only the document CI is informative, and it ignores site dependence.
- MasakhaNEWS has 6 clusters (5 hosts plus the no-URL cluster, effective 1.9), so its site-cluster CI is unstable. The StaticKD lower bound of 0.315 comes from resamples dominated by small clusters.

## 3. TOST margin sensitivity (M8c)

The 90% bootstrap interval of each difference is fixed. A claim is equivalent at a margin when the interval lies inside it (`evaluation.tost`). Each cell gives the verdict with TOST p in brackets. "Paper" marks the claims that `paper_numbers.json` reports as equivalent at ±0.01.

| Comparison | Δ | 90% CI | ±0.005 | ±0.01 | ±0.02 | Paper |
|---|---|---|---|---|---|---|
| StaticKD − Model2Vec clf. (`eqv.`) | -0.002 | [-0.008, +0.008] | no (0.207) | yes (0.025) | yes (<0.001) | yes |
| StaticKD − DistilmBERT (`eqv.`) | +0.010 | [-0.023, +0.036] | no | no (0.428) | no (0.244) | no |
| StaticKD − SD-style 0.13B / 0.27B (`e1.`) | +0.114 / +0.115 | [+0.064, +0.141] / [+0.063, +0.143] | no | no | no | no |
| StaticKD − MiniLM 512 tok. (`len512.`) | -0.035 | [-0.047, -0.023] | no | no | no | no |
| StaticKD − DistilmBERT 512 tok. (`len512.`) | -0.002 | [-0.026, +0.018] | no | no (0.382) | no (0.170) | no |
| Cascade θ=0.8 − teacher ONNX int8 (`casc.eq`) | +0.002 | [-0.002, +0.006] | no (0.104) | yes (<0.001) | yes | yes |
| Hard labels − final (`abltost.hard`) | +0.005 | [-0.001, +0.009] | no (0.440) | yes (0.017) | yes | yes |
| 130k data − final | -0.002 | [-0.007, +0.002] | no (0.144) | yes (0.004) | yes | yes |
| No token dropout − final | +0.001 | [-0.000, +0.003] | yes (<0.001) | yes | yes | yes |
| τ=0.5 − final | +0.004 | [-0.001, +0.007] | no (0.292) | yes (<0.001) | yes | yes |
| τ=2 − final | -0.005 | [-0.009, +0.006] | no (0.419) | yes (0.032) | yes | yes |
| + bigram − final | +0.002 | [-0.000, +0.003] | yes (0.001) | yes | yes | yes |
| + idf pooling − final | -0.001 | [-0.003, +0.001] | yes (0.004) | yes | yes | yes |
| + confidence weights − final | +0.000 | [-0.001, +0.001] | yes (<0.001) | yes | yes | yes |
| + GPT-4o CE − final | -0.004 | [-0.007, +0.002] | no (0.246) | yes (<0.001) | yes | yes |
| MLP head − final | +0.009 | [+0.001, +0.012] | no | no (0.199) | yes (<0.001) | no |
| τ=4 − final | -0.014 | [-0.019, -0.001] | no | no (0.636) | yes (0.031) | no |
| Frozen emb. − final | +0.014 | [-0.001, +0.022] | no | no (0.642) | no (0.122) | no |
| EMM-only − final; random init − final | -0.051; -0.120 | [-0.062, -0.038]; [-0.160, -0.051] | no | no | no | no |

- At ±0.005, only 4 of the 11 paper equivalence claims survive: no token dropout, bigram, idf and confidence weights. Model2Vec, the cascade, hard labels, 130k data, τ=0.5, τ=2 and GPT-4o CE lose equivalence.
- At ±0.02, the MLP head and τ=4 are added. τ=4 is then equivalent although its Welch test is significant.
- The cascade θ=0.8 was re-derived with the notebook's dev rule (part5.py:452-457). The paper's `casc.eq.ci` [-0.003, +0.007] is the 95% interval.

## 4. MLP head (512 units) on held-out agreement

The values come from `paper_numbers.json`, computed at part4.py:587-612.
- Paired site-cluster bootstrap: Δ held-out agreement +0.005 (0.0050) [+0.002, +0.008], Holm p=0.006 (`ablboot.mlp.heldout_acc.*`). The 5 MLP seeds reach 0.813±0.002 (`abl.mlp.heldout_acc`).
- Welch test: Holm p=0.027 (`abl.mlp.heldout_acc.p`).
- Test macro-F1: +0.009 [-0.000, +0.012], Holm p=0.780, TOST ±0.01 p=0.199, not equivalent. Dev macro-F1: +0.008 [-0.007, +0.025], p=1.000.

## 5. "Teacher confidence" in the code

The teacher confidence is the **maximum softmax probability at τ=1 over the teacher's 17 logits**, `softmax(teacher_logits).max()`. The logits come from `classla/multilingual-IPTC-news-topic-classifier` (XLM-R-large) in PyTorch, computed once with fp16 on GPU, max 512 tokens, and reordered to `config.LABELS` (`statickd/teacher.py`, `scripts/label_with_teacher.py`). They are stored in `data/processed/teacher_labels/*.parquet`.
- Factor analysis (Table S4, ρ=0.71): per language, the mean of this value over the site-unseen held-out documents, for the 53 languages with ≥20 documents (part5.py:232-234 `hconf`).
- "+ document weight = teacher confidence" variant: per distillation document, the weight is max-prob^γ with γ=1 (`student.py:158-159`, `teacher_probs(df, 1.0).max(1) ** conf_weight`).
- Figure: `fig_teacher_labels` plots the per-language mean of the same value (part2.py:59-73).
- The StaticKD confidence used by the cascade and the calibration is the max softmax of the StaticKD logits (part5.py:418, `evaluation.ece`/`reliability`).

## 6. Composition of the 285,999 distillation documents

`statickd/data.py` `load_distillation_pool`:
- **20,000** EMMediaTopic train documents (ca, el, hr, sl; no sites)
- **109,842** CC-News 2019-2021 train documents (69 languages)
- **156,157** CC-News extra documents (50 languages, ≤5,000 per language, all from training sites)
- The three sum to 285,999.

The "base version of about 130 thousand" is the `data130k` ablation, sources `("emmediatopic", "ccnews")` (`experiments.py`). Its size is 20,000 + 109,842 = **129,842**, so **the 20,000 EMMediaTopic articles are inside the base version**, not in addition.

Sites:
- 2,960 CC-News sites in total (the pool also holds one pseudo-domain, "emmediatopic"). The base uses 2,832 sites, and the extra data add 128 sites not in the base.
- The per-site caps hold per **(site, language)**: at most 150 in the base and at most 400 in total. Per site over all languages, the maximum is 1,107 (bbc.com) in the base and 2,237 in total. The paper text "at most 150/400 documents per site" (main.tex:258) omits "per language".
- Median length: 269 words.

## 7. Category mappings of the public datasets

The mapping is done by `unified.iptc_label()`. Every source category except MN-DS goes through the same section-name table `testset/iptc_map.py` (`MAPPING`, `map_section`). That table was drafted by an LLM, checked automatically against the IPTC tree, and reviewed by both authors (`mapping_review_record.json`: 2026-10-17, 814 names, "all rules accepted, no changes"; the paper says the authors "corrected it where needed", main.tex:268). Names listed in `AMBIGUOUS` are never mapped, and their articles are dropped. Each MasakhaNEWS category is mapped as follows (n = test articles):

| Category | IPTC label | Test n |
|---|---|---|
| business | economy, business and finance | 675 |
| health | health | 774 |
| politics | politics | 768 |
| religion | religion | 344 |
| sports | sport | 778 |
| entertainment | **dropped** (ambiguous: arts > arts and entertainment, or human interest > celebrity) | 0 |
| technology | **dropped** (ambiguous: science and technology, or economy > computing and IT) | 0 |

MN-DS:
- The label is the dataset's own `category_level_1` field (17 IPTC level-1 names), with `religion and belief` mapped to `religion` (`unified.py` `MNDS_LABEL`). `category_level_2` (109 values) is ignored, and nothing is dropped.
- The CSV has 10,917 rows but only 10,491 unique ids. 137 ids appear in several rows with different level-1 categories, which makes them multi-label in effect. The pipeline treats each row separately, and exact or near-duplicate removal then keeps one row, chosen by the seeded random shuffle. 13 of the 755 test MN-DS articles have a URL that carries several level-1 rows.

Other sources (mapped categories and their test counts):
- **L3Cube-IndicNews** (LDC test + valid):
  - Business/business → economy (300)
  - Crime/crime/crime-news-hindi → crime (450)
  - Politics/politics → politics (332)
  - Sports/sports/khel → sport (479)
  - Health/health/health-news-hindi/Covid → health (250)
  - Education/education → education (200)
  - Cinema/Fashion/Literature → arts (150)
  - Cuisine/Travel → lifestyle (100)
  - Religious → religion (50)
  - career → labour (50)
  - Dropped: entertainment, lifestyle, technology/tech, national/state/world/international, editorial, opinion, auto, astro, bhakti, spirituals, jobs, city names
- **Bangla news:** economy → economy (50), education → education (50), politics → politics (18), sports → sport (21). Dropped: entertainment, international, state, technology.
- **IndoNews:** bisnis → economy (50), bola → sport (50). Dropped: news, otomotif, tekno.
- Amharic news and Swahili news were loaded but contribute **0** test articles. MasakhaNEWS human labels filled the am and sw cells first.

## 8. StaticKD search budget

v1 exploration, before the recipe was fixed (`models/_archive_v1/static/*/meta.json`; `results/_archive_v1/exploration/`):
- **22 trained StaticKD runs in 17 configurations**, plus 2 post-hoc vocabulary-pruned copies and compact tables.
  - Phase 1 (base data): 6 configurations in 7 runs: teacher-distilled init, EMM-only, KD, KD+MLP, KD+MLP+tdrop, KD+tdrop (2 seeds).
  - Phase 2 (full data): 11 configurations in 15 runs: τ=2 final (3 seeds), bigram 2^21, bigram 2^20 with lr 2e-3, τ=0.5, τ=1 (3 seeds), τ=4, GPT-4o CE, confidence weights, 512 tokens, idf pooling, learned token weights.
- Every v1 variant was scored on all three sets: the dev set (1,000), the 320-article LLM check set (16 languages) and CC-News agreement (`all_variants.csv`, `phase2_comparison.csv`).

v2 registry (`experiments.py` `ABLATIONS`): **15 codes, 72 runs** (`models/students` has 72 directories). The final recipe has 10 seeds, the 10 ablations 5 seeds each, and the 4 additions 3 seeds each. These are the reported ablations of the fixed recipe.

Choices by set:
- On the dev set: the 4 additions (bigram, idf, confidence weights, GPT-4o CE) were rejected because none changed dev significantly. The v1 dev macro-F1 deltas against τ=2 were: confidence weights +0.004 (p=0.21), idf pooling -0.004 (p=0.55), GPT-4o CE -0.015 (p=0.16), bigram 2^21 -0.013 (no test recorded).
- Per the paper (supplementary.tex, "LLM check set"), the 320-article LLM set decided τ=1 and K=3.
- The v1 numbers for τ=1 against τ=2 are:
  - dev: +0.002 for seed 0 (p=0.88) and +0.010 for the 3-seed ensemble (p=0.43)
  - LLM set: -0.009 for seed 0, +0.016 for seeds 1 and 2, and +0.005 for the ensemble
  - CC-News agreement: +0.008 (p=0.004)
- So τ and K did not rest on the LLM set alone, but the paper attributes them to it.
- Epochs (6), token dropout (0.2), the linear head and the data came from phase-1 runs scored on the same three sets.

## 9. Test-set languages (47) and regions (`analysis.REGION`, English via `analysis.EN`)

| Region | Languages |
|---|---|
| Europe (16) | bg Bulgarian, ca Catalan, cs Czech, de German, el Greek, en English, es Spanish, fr French, it Italian, nl Dutch, pl Polish, pt Portuguese, ro Romanian, ru Russian, tr Turkish, uk Ukrainian |
| South Asia (10) | bn Bengali, gu Gujarati, hi Hindi, kn Kannada, ml Malayalam, mr Marathi, or Odia, pa Punjabi, ta Tamil, te Telugu |
| Africa (14) | am Amharic, ha Hausa, ig Igbo, lg Luganda, ln Lingala, om Oromo, pcm Nigerian Pidgin, rn Kirundi, sn Shona, so Somali, sw Swahili, ti Tigrinya, xh isiXhosa, yo Yoruba |
| East/Southeast Asia & Middle East (7) | ar Arabic, fa Persian, id Indonesian, ko Korean, th Thai, vi Vietnamese, zh Chinese |

- Per-language n and sources are in the JSON (`item9_languages`).
- `pcm` is an ISO 639-3 code (no ISO 639-1 code exists), although the release `data/README.md` says "ISO 639-1".
- en (MN-DS 755, MasakhaNEWS 95) and fr (CC-NEWS 208, MasakhaNEWS 200) count as Europe although part of their articles is African news.

## 10. Licenses and the released manifest

Licenses (supplementary Table S5 = `dist/.../THIRD_PARTY_LICENSES.md`):

| Source | License |
|---|---|
| MasakhaNEWS | CC BY-NC 4.0 |
| MN-DS, L3Cube-IndicNews, IndoNews | CC BY 4.0 |
| Bangla news | not stated on the dataset card |
| CC-News 2019-2021, CC-NEWS 2024-2026 | Common Crawl terms of use |
| Released manifests | CC BY-SA 4.0 |

The released manifest (`dist/static_kd_iptc_classifier/data/testset/unified_test_manifest.csv`) contains no article text. Its columns are doc_id, source, label_origin, lang, label, label_id, source_label, url, domain, warc_file, source_row_id, text_sha1 and n_words. Every row has a label, a source label and a text SHA-1. Per source:

| Source | Identifiers released |
|---|---|
| CC-NEWS (6,050) | url, domain and warc_file for all rows |
| MasakhaNEWS (3,339) | url column filled for all rows, but 200 are "not available" (xh) and 95 are relative paths (en). Domain for 3,044 rows. |
| MN-DS (755) | url and domain for all rows |
| L3Cube (2,361), Bangla (139), IndoNews (100) | `source_row_id` only (dataset file#row, e.g. `L3Cube-IndicNews/Odia/valid.csv#row=1756`, `jakartaresearch/indonews/validation#row=1129`) |

Possible reviewer point: the manifest is released under CC BY-SA 4.0 and carries MasakhaNEWS URLs and derived labels, while MasakhaNEWS itself is CC BY-NC 4.0.

## 11. Figure labels

`fig_calibration` is produced in `src/notebook_build/part5.py:400-415`, with the axis labels at line 413:
- `ax.set_xlabel("keyakinan StaticKD")` should read **"StaticKD confidence"** (the max softmax probability).
- `ax.set_ylabel("akurasi")` should read **"accuracy"**.

The title ("StaticKD reliability diagram") and the legend ("dev (ECE 0.041)", "test (ECE 0.139)") are already English.

I extracted the text of every figure PDF used in `supplementary.tex` and `main.tex` with pdftotext. **No other figure contains non-English text**: fig_testset_heatmap, fig_latency_ram_size, fig_temperature, fig_ensemble_size, fig_groups, fig_per_label, and all 7 main-text figures. One cosmetic point: fig_testset_heatmap truncates the label origin to "(JSON-LD s)" and "(human ann)".
