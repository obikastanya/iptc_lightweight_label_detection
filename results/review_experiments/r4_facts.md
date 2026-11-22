# Review round 4: fact sheet

Script: `scripts/review_experiments/r4_reviewer_checks.py` (no training; stored data, predictions and trained students).
Machine-readable numbers: `results/review_experiments/r4_checks.json`.
Bootstrap settings are the notebook's: CIs use 1,000 resamples and paired tests 2,000 resamples, seed 0, site clusters on test and held-out, documents on dev. Δ = StaticKD − model unless stated otherwise.

Sanity check against paper_numbers.json: unrestricted Δ vs Model2Vec -0.002 [-0.009, +0.010] (paper −0.009, +0.010), TOST 90% CI [-0.008, +0.008] (paper −0.008, +0.008); teacher test CI [0.546, 0.643] (paper 0.546--0.643); Δ vs MiniLM Holm p 0.005 (paper 0.005).

## 1. Pooled macro-F1 artifact (M2)

### (a) Label inventory per source (gold labels in the test set)

| Source | n | # labels | Labels (count) |
|---|---|---|---|
| ccnews_jsonld | 6,050 | 17 | arts, culture, entertainment and media (567), conflict, war and peace (110), crime, law and justice (414), disaster, accident and emergency incident (28), economy, business and finance (809), education (317), environment (228), health (540), human interest (251), labour (53), lifestyle and leisure (297), politics (872), religion (72), science and technology (333), society (82), sport (895), weather (182) |
| masakhanews | 3,339 | 5 | economy, business and finance (675), health (774), politics (768), religion (344), sport (778) |
| l3cube_indic | 2,361 | 10 | arts, culture, entertainment and media (150), crime, law and justice (450), economy, business and finance (300), education (200), health (250), labour (50), lifestyle and leisure (100), politics (332), religion (50), sport (479) |
| mnds | 755 | 17 | arts, culture, entertainment and media (50), conflict, war and peace (50), crime, law and justice (50), disaster, accident and emergency incident (50), economy, business and finance (25), education (50), environment (50), health (26), human interest (50), labour (50), lifestyle and leisure (50), politics (32), religion (50), science and technology (50), society (50), sport (22), weather (50) |
| bangla_plos | 139 | 4 | economy, business and finance (50), education (50), politics (18), sport (21) |
| indonews | 100 | 2 | economy, business and finance (50), sport (50) |

### (b) Restricted-inventory evaluation (argmax over the labels of the article's source)

All models have stored 17-way logits for every test article (`results/final/predictions/<model>__test.npz`). Teacher, transformers, StaticKD and SD-style store real logits; Model2Vec stores log-probabilities (clipped at 1e-12), fastText log-probabilities (floored at 1e-5 = −11.513, so labels can tie at the floor), TF-IDF+SVM the SVM decision scores. Ties at the restricted maximum are broken by the lowest label index (argmax).

| Model | Logits | Macro-F1 | 95% CI | Acc. | Macro-F1 restricted | 95% CI | Acc. restricted | Pred. outside inventory | Ties at restricted max |
|---|---|---|---|---|---|---|---|---|---|
| Teacher XLM-R-large | real | 0.598 | 0.546–0.643 | 0.700 | 0.658 | 0.635–0.672 | 0.754 | 1,309 (10.3%) | 1 |
| StaticKD (proposed) | real | 0.545 | 0.490–0.605 | 0.641 | 0.619 | 0.596–0.635 | 0.707 | 1,825 (14.3%) | 0 |
| Model2Vec classifier + KD | pseudo | 0.547 | 0.495–0.599 | 0.642 | 0.611 | 0.587–0.627 | 0.696 | 1,578 (12.4%) | 0 |
| MiniLM-L6 + KD | real | 0.566 | 0.511–0.624 | 0.667 | 0.625 | 0.605–0.644 | 0.720 | 1,382 (10.8%) | 4 |
| DistilmBERT + KD | real | 0.534 | 0.481–0.618 | 0.623 | 0.597 | 0.568–0.634 | 0.675 | 1,352 (10.6%) | 1 |
| fastText + KD, subword | pseudo | 0.447 | 0.385–0.566 | 0.527 | 0.510 | 0.464–0.579 | 0.574 | 1,871 (14.7%) | 4 |
| TF-IDF + SVM + KD | pseudo | 0.486 | 0.431–0.585 | 0.567 | 0.534 | 0.492–0.597 | 0.607 | 1,659 (13.0%) | 0 |
| SD-style + KD, 0.13B | real | 0.431 | 0.374–0.538 | 0.502 | 0.491 | 0.445–0.552 | 0.562 | 2,159 (16.9%) | 0 |

Other models of the main table (restricted macro-F1 / acc.): Teacher XLM-R-large, ONNX int8 0.660 / 0.754 (unrestricted 0.600); StaticKD-compact 0.557 / 0.608 (unrestricted 0.484); fastText + KD, subword, quantized 0.505 / 0.565 (unrestricted 0.439); fastText + KD, word 0.512 / 0.571 (unrestricted 0.445); SD-style + KD, 0.27B 0.493 / 0.562 (unrestricted 0.430); fastText, GPT-4o labels, subword 0.116 / 0.212 (unrestricted 0.086); fastText, GPT-4o labels, word 0.133 / 0.251 (unrestricted 0.106).

### (c) Paired site-cluster bootstrap, StaticKD − model (Holm over the 14 comparisons of the significance table, per condition)

| Model | Δ unrestricted [95% CI] | Holm p | Δ restricted [95% CI] | Holm p |
|---|---|---|---|---|
| Teacher XLM-R-large | -0.053 [-0.065, -0.033] | <0.001 | -0.039 [-0.050, -0.025] | <0.001 |
| Teacher XLM-R-large, ONNX int8 | -0.055 [-0.067, -0.035] | <0.001 | -0.041 [-0.053, -0.027] | <0.001 |
| DistilmBERT + KD | +0.010 [-0.026, +0.041] | 1.000 | +0.023 [-0.019, +0.044] | 0.606 |
| MiniLM-L6 + KD | -0.021 [-0.038, -0.008] | 0.005 | -0.006 [-0.027, +0.009] | 0.606 |
| StaticKD-compact | +0.060 [+0.003, +0.100] | 0.042 | +0.062 [+0.004, +0.103] | 0.048 |
| fastText + KD, subword, quantized | +0.106 [+0.037, +0.145] | <0.001 | +0.114 [+0.042, +0.155] | <0.001 |
| fastText + KD, subword | +0.098 [+0.036, +0.134] | <0.001 | +0.109 [+0.040, +0.148] | <0.001 |
| fastText + KD, word | +0.099 [+0.037, +0.141] | <0.001 | +0.107 [+0.039, +0.150] | <0.001 |
| TF-IDF + SVM + KD | +0.059 [+0.009, +0.093] | 0.040 | +0.085 [+0.019, +0.121] | <0.001 |
| Model2Vec classifier + KD | -0.002 [-0.009, +0.010] | 1.000 | +0.008 [+0.001, +0.015] | 0.069 |
| SD-style + KD, 0.13B | +0.114 [+0.060, +0.145] | <0.001 | +0.128 [+0.063, +0.166] | <0.001 |
| SD-style + KD, 0.27B | +0.115 [+0.058, +0.148] | <0.001 | +0.126 [+0.060, +0.167] | <0.001 |
| fastText, GPT-4o labels, subword | +0.458 [+0.415, +0.496] | <0.001 | +0.503 [+0.474, +0.524] | <0.001 |
| fastText, GPT-4o labels, word | +0.439 [+0.393, +0.475] | <0.001 | +0.487 [+0.450, +0.507] | <0.001 |

TOST ±0.01 vs Model2Vec classifier + KD: unrestricted 90% CI [-0.008, +0.008], p 0.025, equivalent True; restricted 90% CI [+0.002, +0.014], p 0.295, equivalent False.

TOST ±0.01 vs DistilmBERT + KD: unrestricted 90% CI [-0.023, +0.036], p 0.428, equivalent False; restricted 90% CI [-0.013, +0.041], p 0.670, equivalent False.

Conclusion changes (significance at Holm 0.05 or sign of Δ): MiniLM-L6 + KD: unrestricted Δ -0.021 (p 0.005) -> restricted Δ -0.006 (p 0.606); Model2Vec classifier + KD: unrestricted Δ -0.002 (p 1.000) -> restricted Δ +0.008 (p 0.069)

### (d) Why the pooled human-annotation subgroup is below both of its sources

The subgroup pools MasakhaNEWS (3,339 articles, 5 labels: economy, business and finance, health, politics, religion, sport) and MN-DS (755 articles, 17 labels). Pooled macro-F1 averages over all 17 labels. The 12 labels that occur only in MN-DS (arts, culture, entertainment and media, conflict, war and peace, crime, law and justice, disaster, accident and emergency incident, education, environment, human interest, labour, lifestyle and leisure, science and technology, society, weather) have few gold articles, but every MasakhaNEWS article that a model assigns to one of them is a false positive for that label, so their precision collapses in the pool.

Teacher: pooled 0.463 = mean of 17 label F1s. Mean F1 of the 5 MasakhaNEWS labels in the pool 0.679; mean F1 of the 12 MN-DS-only labels in the pool 0.372 against 0.587 within MN-DS alone. 1,041 MasakhaNEWS articles (31.2%) are predicted as an MN-DS-only label. MasakhaNEWS alone 0.695, MN-DS alone 0.549, pooled with restricted argmax 0.630.

Teacher, per label in the pooled subgroup:

| Label | Gold Masakha | Gold MN-DS | TP | FP from Masakha | FP from MN-DS | FN | F1 pooled | Contribution (F1/17) | F1 within Masakha | F1 within MN-DS |
|---|---|---|---|---|---|---|---|---|---|---|
| arts, culture, entertainment and media | 0 | 50 | 24 | 110 | 27 | 26 | 0.227 | 0.013 | -- | 0.475 |
| conflict, war and peace | 0 | 50 | 41 | 187 | 22 | 9 | 0.273 | 0.016 | -- | 0.726 |
| crime, law and justice | 0 | 50 | 25 | 145 | 38 | 25 | 0.194 | 0.011 | -- | 0.442 |
| disaster, accident and emergency incident | 0 | 50 | 32 | 71 | 21 | 18 | 0.368 | 0.022 | -- | 0.621 |
| economy, business and finance | 675 | 25 | 319 | 27 | 18 | 381 | 0.600 | 0.035 | 0.600 | 0.590 |
| education | 0 | 50 | 36 | 29 | 13 | 14 | 0.562 | 0.033 | -- | 0.727 |
| environment | 0 | 50 | 35 | 52 | 15 | 15 | 0.461 | 0.027 | -- | 0.700 |
| health | 774 | 26 | 443 | 27 | 22 | 357 | 0.686 | 0.040 | 0.703 | 0.316 |
| human interest | 0 | 50 | 20 | 253 | 24 | 30 | 0.115 | 0.007 | -- | 0.426 |
| labour | 0 | 50 | 26 | 38 | 2 | 24 | 0.448 | 0.026 | -- | 0.667 |
| lifestyle and leisure | 0 | 50 | 24 | 17 | 5 | 26 | 0.500 | 0.029 | -- | 0.608 |
| politics | 768 | 32 | 489 | 210 | 97 | 311 | 0.613 | 0.036 | 0.647 | 0.280 |
| religion | 344 | 50 | 190 | 28 | 3 | 204 | 0.618 | 0.036 | 0.632 | 0.507 |
| science and technology | 0 | 50 | 18 | 19 | 5 | 32 | 0.391 | 0.023 | -- | 0.493 |
| society | 0 | 50 | 13 | 118 | 12 | 37 | 0.135 | 0.008 | -- | 0.347 |
| sport | 778 | 22 | 644 | 3 | 16 | 156 | 0.880 | 0.052 | 0.891 | 0.593 |
| weather | 0 | 50 | 36 | 2 | 3 | 14 | 0.791 | 0.047 | -- | 0.809 |

All models (pooled human subgroup):

| Model | Pooled | Masakha alone | MN-DS alone | Pool: mean F1 5 Masakha labels | Pool: mean F1 12 MN-DS-only labels | Same 12 within MN-DS | Masakha arts. -> MN-DS-only label | Pooled restricted |
|---|---|---|---|---|---|---|---|---|
| Teacher XLM-R-large | 0.463 | 0.695 | 0.549 | 0.679 | 0.372 | 0.587 | 1,041 (31.2%) | 0.630 |
| StaticKD (proposed) | 0.414 | 0.556 | 0.535 | 0.548 | 0.359 | 0.590 | 1,489 (44.6%) | 0.600 |
| Model2Vec classifier + KD | 0.414 | 0.562 | 0.536 | 0.553 | 0.356 | 0.589 | 1,170 (35.0%) | 0.593 |
| MiniLM-L6 + KD | 0.436 | 0.582 | 0.535 | 0.577 | 0.377 | 0.571 | 1,084 (32.5%) | 0.591 |
| DistilmBERT + KD | 0.436 | 0.545 | 0.546 | 0.542 | 0.392 | 0.587 | 902 (27.0%) | 0.593 |
| fastText + KD, subword | 0.345 | 0.362 | 0.489 | 0.368 | 0.336 | 0.534 | 1,517 (45.4%) | 0.504 |
| TF-IDF + SVM + KD | 0.393 | 0.434 | 0.461 | 0.431 | 0.378 | 0.512 | 1,181 (35.4%) | 0.500 |
| SD-style + KD, 0.13B | 0.314 | 0.350 | 0.425 | 0.353 | 0.298 | 0.458 | 1,798 (53.8%) | 0.455 |

### (e) Mean of per-source macro-F1 (6 sources)

| Model | Pooled test macro-F1 | Mean unweighted | Mean article-weighted | Restricted: unweighted | Restricted: weighted |
|---|---|---|---|---|---|
| Teacher XLM-R-large | 0.598 | 0.721 | 0.671 | 0.757 | 0.689 |
| Teacher XLM-R-large, ONNX int8 | 0.600 | 0.714 | 0.671 | 0.757 | 0.691 |
| DistilmBERT + KD | 0.534 | 0.633 | 0.581 | 0.683 | 0.607 |
| MiniLM-L6 + KD | 0.566 | 0.684 | 0.627 | 0.728 | 0.648 |
| StaticKD (proposed) | 0.545 | 0.661 | 0.607 | 0.699 | 0.632 |
| StaticKD-compact | 0.484 | 0.469 | 0.514 | 0.516 | 0.541 |
| fastText + KD, subword, quantized | 0.439 | 0.461 | 0.464 | 0.507 | 0.488 |
| fastText + KD, subword | 0.447 | 0.461 | 0.471 | 0.518 | 0.495 |
| fastText + KD, word | 0.445 | 0.441 | 0.472 | 0.507 | 0.498 |
| TF-IDF + SVM + KD | 0.486 | 0.455 | 0.501 | 0.494 | 0.518 |
| Model2Vec classifier + KD | 0.547 | 0.654 | 0.606 | 0.705 | 0.625 |
| SD-style + KD, 0.13B | 0.431 | 0.442 | 0.450 | 0.490 | 0.481 |
| SD-style + KD, 0.27B | 0.430 | 0.432 | 0.447 | 0.476 | 0.482 |
| fastText, GPT-4o labels, subword | 0.086 | 0.117 | 0.102 | 0.198 | 0.139 |
| fastText, GPT-4o labels, word | 0.106 | 0.131 | 0.119 | 0.167 | 0.155 |

## 2. Frozen-embedding variant, folded three-seed ensemble (M3)

Trained frozen students are stored (`models/students/frozen_s0..s4`, config: {'freeze_embeddings': True, 'init': 'potion', 'hidden': 0, 'loss': 'kd', 'temperature': 1.0, 'token_dropout': 0.2, 'epochs': 6, 'sources': ['emmediatopic', 'ccnews', 'ccnews_extra']}). The ensemble averages the token tables and biases of seeds 0, 1, 2 in memory (same arithmetic as `student.ensemble`; nothing written to `models/`). Its logits are saved in `results/review_experiments/r4_frozen_ensemble/`. Procedure check: the same in-memory fold of full_s0..2 reproduces the stored StaticKD dev logits (max |Δlogit| 0.00e+00, argmax agreement 1.0000). Agreement with the mean of the stored per-seed logits: dev 1.0000, heldout 1.0000, test 1.0000.

Main-table format:

| Model | Dev macro-F1 [95% CI] | Dev acc. | Held-out agreement | Test macro-F1 [95% CI] | Test acc. | Macro-F1/lang. | % teacher (test) |
|---|---|---|---|---|---|---|---|
| Frozen embedding, 3-seed ensemble | 0.719 [0.673–0.751] | 0.757 | 0.788 | 0.555 [0.506–0.602] | 0.659 | 0.644 | 92.7% |
| StaticKD (paper) | 0.761 [0.717--0.792] | 0.800 | 0.811 | 0.545 [0.490--0.605] | 0.641 | 0.622 | 91.1% |

Dev % teacher: frozen ensemble 87.5%. Held-out macro-F1 0.759 [0.740–0.770], agreement CI [0.773–0.802].

Seed view (macro-F1): dev: frozen seeds mean 0.720 (s0–s2 0.720) -> ensemble 0.719; final-recipe seeds mean 0.755 (s0–s2 0.756) -> StaticKD 0.761; test: frozen seeds mean 0.557 (s0–s2 0.555) -> ensemble 0.555; final-recipe seeds mean 0.542 (s0–s2 0.542) -> StaticKD 0.545.

Paired, StaticKD − frozen ensemble (single comparison, no Holm):

| Set / metric | Δ | 95% CI | p | TOST ±0.01 90% CI | TOST p | Equivalent |
|---|---|---|---|---|---|---|
| test macro_f1 | -0.010 | [-0.021, +0.009] | 0.344 | [-0.019, +0.007] | 0.406 | False |
| test accuracy | -0.018 | [-0.028, +0.002] | 0.078 | [-0.027, -0.002] | 0.795 | False |
| heldout accuracy | +0.023 | [+0.017, +0.030] | <0.001 | [+0.018, +0.029] | 1.000 | False |
| dev macro_f1 | +0.042 | [+0.013, +0.070] | 0.004 | [+0.018, +0.065] | 0.986 | False |

Test macro-F1 by distillation data of the language:

| Tier | n | Frozen ens. | StaticKD | Teacher | Model2Vec | Frozen seed mean | Final-recipe seed mean | Frozen ens. % teacher | StaticKD % teacher |
|---|---|---|---|---|---|---|---|---|---|
| sufficient (≥500) | 7,644 | 0.606 | 0.618 | 0.633 | 0.611 | 0.607 | 0.615 | 95.7% | 97.5% |
| few (<500) | 2,243 | 0.542 | 0.478 | 0.537 | 0.475 | 0.541 | 0.482 | 101.0% | 89.1% |
| none (0) | 2,857 | 0.430 | 0.402 | 0.557 | 0.429 | 0.440 | 0.403 | 77.3% | 72.3% |

## 3. Source-level sensitivity (M4)

Per source, Δ test macro-F1 = StaticKD − model, paired bootstrap 2,000 (site-cluster where the source has site information, documents otherwise; no multiplicity correction):

| Source | n | Eff. clusters | StaticKD | Δ vs Model2Vec [95% CI] | Δ vs teacher [95% CI] | Δ vs MiniLM [95% CI] |
|---|---|---|---|---|---|---|
| ccnews_jsonld | 6,050 | 221.3 | 0.624 | +0.006 [-0.003, +0.014] | -0.028 [-0.041, -0.015] | -0.010 [-0.022, +0.002] |
| masakhanews | 3,339 | 1.9 | 0.556 | -0.006 [-0.037, +0.001] | -0.139 [-0.322, -0.105] | -0.026 [-0.270, +0.016] |
| l3cube_indic | 2,361 | 1.0 | 0.639 | -0.005 [-0.021, +0.010] | -0.069 [-0.091, -0.048] | -0.041 [-0.063, -0.018] |
| mnds | 755 | 32.7 | 0.535 | -0.001 [-0.016, +0.014] | -0.014 [-0.037, +0.007] | -0.000 [-0.025, +0.028] |
| bangla_plos | 139 | 1.0 | 0.748 | +0.031 [-0.005, +0.070] | -0.087 [-0.163, -0.017] | -0.070 [-0.137, -0.004] |
| indonews | 100 | 1.0 | 0.862 | +0.013 [+0.000, +0.035] | -0.023 [-0.068, +0.017] | +0.005 [-0.035, +0.048] |

Sign consistency over the 6 sources (macro-F1 Δ > 0 / < 0; accuracy Δ > 0): Model2Vec classifier + KD 3/3 (acc. 4/6); Teacher XLM-R-large 0/6 (acc. 0/6); MiniLM-L6 + KD 1/5 (acc. 2/6).

Discordant articles per source (only StaticKD correct / only model correct): bangla_plos: model2vec_clf 6/1, teacher 9/28, minilm_kd 8/26; ccnews_jsonld: model2vec_clf 166/127, teacher 291/387, minilm_kd 356/356; indonews: model2vec_clf 2/0, teacher 3/7, minilm_kd 5/4; l3cube_indic: model2vec_clf 92/69, teacher 85/218, minilm_kd 108/172; masakhanews: model2vec_clf 150/228, teacher 211/699, minilm_kd 331/583; mnds: model2vec_clf 22/22, teacher 55/61, minilm_kd 58/56.

Leave-one-source-out range of Δ (StaticKD − model) over the 6 left-out sources:

| Model | Full test Δ | Min Δ (source left out) | Max Δ (source left out) |
|---|---|---|---|
| Teacher XLM-R-large | -0.053 | -0.065 (ccnews_jsonld) | -0.033 (masakhanews) |
| MiniLM-L6 + KD | -0.021 | -0.030 (ccnews_jsonld) | -0.012 (masakhanews) |
| DistilmBERT + KD | +0.010 | -0.013 (l3cube_indic) | +0.027 (masakhanews) |
| Model2Vec classifier + KD | -0.002 | -0.003 (l3cube_indic) | +0.006 (masakhanews) |
| TF-IDF + SVM + KD | +0.059 | +0.028 (l3cube_indic) | +0.087 (ccnews_jsonld) |
| fastText + KD, subword, quantized | +0.106 | +0.070 (l3cube_indic) | +0.154 (ccnews_jsonld) |
| fastText + KD, subword | +0.098 | +0.064 (l3cube_indic) | +0.141 (ccnews_jsonld) |
| fastText + KD, word | +0.099 | +0.064 (l3cube_indic) | +0.146 (ccnews_jsonld) |
| SD-style + KD, 0.13B | +0.114 | +0.085 (l3cube_indic) | +0.147 (ccnews_jsonld) |
| SD-style + KD, 0.27B | +0.115 | +0.083 (l3cube_indic) | +0.150 (ccnews_jsonld) |

Leave-one-source-out paired tests (Holm over the three focus comparisons; TOST ±0.01 for Model2Vec):

| Left out | n | Eff. clusters | Δ vs Model2Vec [95% CI], TOST p | Δ vs teacher, p_Holm | Δ vs MiniLM, p_Holm |
|---|---|---|---|---|---|
| bangla_plos | 12,605 | 13.6 | -0.002 [-0.009, +0.010], 0.022 (equiv.) | -0.052, <0.001 | -0.019, 0.022 |
| ccnews_jsonld | 6,694 | 3.9 | -0.003 [-0.011, +0.010], 0.037 (equiv.) | -0.065, 0.003 | -0.030, 0.008 |
| indonews | 12,644 | 13.6 | -0.002 [-0.009, +0.010], 0.024 (equiv.) | -0.053, <0.001 | -0.021, 0.010 |
| l3cube_indic | 10,383 | 17.5 | -0.003 [-0.010, +0.011], 0.042 (equiv.) | -0.048, <0.001 | -0.018, 0.014 |
| masakhanews | 9,405 | 15.3 | +0.006 [-0.000, +0.012], 0.080 (not equiv.) | -0.033, <0.001 | -0.012, 0.132 |
| mnds | 11,989 | 12.3 | -0.003 [-0.011, +0.012], 0.046 (equiv.) | -0.060, <0.001 | -0.026, <0.001 |

Mixed-effects logistic regression: skipped: statsmodels is not installed in env/ (no install was made). Within-article pairing makes the site and language intercepts cancel for the model contrast; the discordant counts per source are given instead.

## 4. Holm adjustment of the TOST p-values (margin ±0.01)

bootstrap TOST p = larger one-sided tail share of 2,000 paired site-cluster resamples; a tail count of 0 gives p = 0, so a conservative (k+1)/(B+1) version is Holm-adjusted as well.

**Family A: the equivalence claims of the paper** (11 tests)

| Test | Δ | 90% CI | TOST p | Holm p | Survives | Holm p (conservative) | Survives |
|---|---|---|---|---|---|---|---|
| cascade_theta0.8_vs_teacher_onnx_int8 | +0.002 | [-0.002, +0.006] | <0.001 | <0.001 | yes | 0.005 | yes |
| ablation_no_tdrop | +0.001 | [-0.000, +0.003] | <0.001 | <0.001 | yes | 0.005 | yes |
| ablation_tau0.5 | +0.004 | [-0.001, +0.007] | <0.001 | <0.001 | yes | 0.005 | yes |
| ablation_bigram | +0.002 | [-0.000, +0.003] | <0.001 | <0.001 | yes | 0.005 | yes |
| ablation_idf | -0.001 | [-0.003, +0.001] | <0.001 | <0.001 | yes | 0.005 | yes |
| ablation_conf | +0.000 | [-0.001, +0.001] | <0.001 | <0.001 | yes | 0.005 | yes |
| ablation_gpt_ce | -0.004 | [-0.007, +0.002] | <0.001 | <0.001 | yes | 0.005 | yes |
| ablation_data130k | -0.002 | [-0.007, +0.002] | 0.004 | 0.018 | yes | 0.020 | yes |
| ablation_hard | +0.005 | [-0.001, +0.009] | 0.017 | 0.051 | no | 0.052 | no |
| model2vec_clf_vs_statickd | -0.002 | [-0.008, +0.008] | 0.025 | 0.051 | no | 0.052 | no |
| ablation_tau2 | -0.005 | [-0.009, +0.006] | 0.032 | 0.051 | no | 0.052 | no |

**Family B: every TOST reported in the paper** (19 tests)

| Test | Δ | 90% CI | TOST p | Holm p | Survives | Holm p (conservative) | Survives |
|---|---|---|---|---|---|---|---|
| cascade_theta0.8_vs_teacher_onnx_int8 | +0.002 | [-0.002, +0.006] | <0.001 | <0.001 | yes | 0.009 | yes |
| ablation_no_tdrop | +0.001 | [-0.000, +0.003] | <0.001 | <0.001 | yes | 0.009 | yes |
| ablation_tau0.5 | +0.004 | [-0.001, +0.007] | <0.001 | <0.001 | yes | 0.009 | yes |
| ablation_bigram | +0.002 | [-0.000, +0.003] | <0.001 | <0.001 | yes | 0.009 | yes |
| ablation_idf | -0.001 | [-0.003, +0.001] | <0.001 | <0.001 | yes | 0.009 | yes |
| ablation_conf | +0.000 | [-0.001, +0.001] | <0.001 | <0.001 | yes | 0.009 | yes |
| ablation_gpt_ce | -0.004 | [-0.007, +0.002] | <0.001 | <0.001 | yes | 0.009 | yes |
| ablation_data130k | -0.002 | [-0.007, +0.002] | 0.004 | 0.054 | no | 0.060 | no |
| ablation_hard | +0.005 | [-0.001, +0.009] | 0.017 | 0.187 | no | 0.192 | no |
| model2vec_clf_vs_statickd | -0.002 | [-0.008, +0.008] | 0.025 | 0.245 | no | 0.250 | no |
| ablation_tau2 | -0.005 | [-0.009, +0.006] | 0.032 | 0.288 | no | 0.292 | no |
| ablation_mlp | +0.009 | [+0.001, +0.012] | 0.199 | 1.000 | no | 1.000 | no |
| distilmbert_kd_len512_vs_statickd | -0.002 | [-0.026, +0.018] | 0.382 | 1.000 | no | 1.000 | no |
| distilmbert_kd_vs_statickd | +0.010 | [-0.023, +0.036] | 0.428 | 1.000 | no | 1.000 | no |
| ablation_tau4 | -0.014 | [-0.019, -0.001] | 0.636 | 1.000 | no | 1.000 | no |
| ablation_frozen | +0.014 | [-0.001, +0.022] | 0.642 | 1.000 | no | 1.000 | no |
| minilm_kd_t1_len512_vs_statickd | -0.035 | [-0.047, -0.023] | 1.000 | 1.000 | no | 1.000 | no |
| ablation_emm_only | -0.051 | [-0.062, -0.038] | 1.000 | 1.000 | no | 1.000 | no |
| ablation_random_init | -0.120 | [-0.160, -0.051] | 1.000 | 1.000 | no | 1.000 | no |

## 5. Label-audit ranking (300 audited articles)

| Model | Editorial: acc. (k/300) | Wilson 95% | Audited: acc. (k/300) | Wilson 95% |
|---|---|---|---|---|
| teacher | 78.0% (234) | 73.0%–82.3% | 90.7% (272) | 86.8%–93.5% |
| MiniLM-L6 + KD | 74.0% (222) | 68.8%–78.6% | 86.0% (258) | 81.6%–89.5% |
| Model2Vec classifier + KD | 74.0% (222) | 68.8%–78.6% | 82.3% (247) | 77.6%–86.2% |
| StaticKD (final) | 72.3% (217) | 67.0%–77.1% | 81.7% (245) | 76.9%–85.6% |
| DistilmBERT + KD | 69.3% (208) | 63.9%–74.3% | 74.7% (224) | 69.5%–79.3% |
| fastText + KD | 61.3% (184) | 55.7%–66.7% | 67.3% (202) | 61.8%–72.4% |
| TF-IDF + SVM + KD | 60.7% (182) | 55.0%–66.0% | 66.7% (200) | 61.2%–71.8% |

Ranking, editorial labels: teacher > Model2Vec classifier + KD > MiniLM-L6 + KD > StaticKD (final) > DistilmBERT + KD > fastText + KD > TF-IDF + SVM + KD (ties: [['Model2Vec classifier + KD', 'MiniLM-L6 + KD']]).
Ranking, audited labels: teacher > MiniLM-L6 + KD > Model2Vec classifier + KD > StaticKD (final) > DistilmBERT + KD > fastText + KD > TF-IDF + SVM + KD (ties: []).
fastText + KD (ft_kd_subword (scripts/label_audit/analyze.py MODELS)) minus TF-IDF + SVM + KD, correct articles: editorial +2, audited +2; swap: False.

## 6. 512-token transformers in main-table format

| Model | Dev macro-F1 [95% CI] | Dev acc. | Held-out agreement | Test macro-F1 [95% CI] | Test acc. | Macro-F1/lang. | % teacher (test) |
|---|---|---|---|---|---|---|---|
| MiniLM-L6 + KD, 512 tokens | 0.808 [0.769–0.836] | 0.835 | 0.848 | 0.580 [0.526–0.633] | 0.681 | 0.672 | 97.0% |
| DistilmBERT + KD, 512 tokens | 0.772 [0.729–0.803] | 0.813 | 0.827 | 0.547 [0.497–0.621] | 0.637 | 0.614 | 91.4% |
| MiniLM-L6 + KD, 256 tokens (paper row) | 0.785 [0.737–0.815] | 0.820 | 0.825 | 0.566 [0.511–0.624] | 0.667 | 0.661 | 94.6% |
| DistilmBERT + KD, 256 tokens (paper row) | 0.753 [0.710–0.785] | 0.801 | 0.799 | 0.534 [0.481–0.618] | 0.623 | 0.602 | 89.4% |
| StaticKD (paper row) | 0.761 [0.717–0.792] | 0.800 | 0.811 | 0.545 [0.490–0.605] | 0.641 | 0.622 | 91.1% |

Dev % teacher: MiniLM-L6 + KD, 512 tokens 98.4%, DistilmBERT + KD, 512 tokens 94.0%, MiniLM-L6 + KD, 256 tokens (paper row) 95.5%, DistilmBERT + KD, 256 tokens (paper row) 91.7%, StaticKD (paper row) 92.6%.
Paired tests vs StaticKD for the 512-token rows are in paper_numbers.json (`len512.*`): MiniLM-512 Δ -0.035 [−0.049, −0.021], p <0.001; DistilmBERT-512 Δ -0.002 [−0.029, +0.022], p 0.779.
