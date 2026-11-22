# Review round 5: fact sheet

Scripts: `scripts/review_experiments/r5_a_analyses.py` (part A), `r5_b1_model2vec_linear.py`, `r5_b2_ccnews20k.py` (part B, GPU), `r5_c_prepare_eval.py`, `r5_c_benchmark.py` (part C, CPU), `r5_report.py` (statistics of B/C and this sheet). Numbers: `results/review_experiments/r5_checks.json`. Bootstrap conventions as the notebook and r4: CIs from 1,000 resamples, paired tests from 2,000, seed 0; site clusters on test and held-out, documents on dev. Δ = StaticKD − model unless stated otherwise.

Sanity: the unstratified pass reproduces r4 and the paper (Δ vs Model2Vec -0.002 [-0.009, +0.010], MiniLM Holm p 0.005 (paper 0.005), teacher CI [0.546, 0.643] (paper 0.546--0.643), TOST p vs Model2Vec 0.0245 (r4 0.0245)).

## A1. Source-stratified site-cluster bootstrap

Stratified scheme: the six test sources are fixed; within each source its sites are resampled with replacement. Sources without site information form a single cluster in the paper's bootstrap; within the stratified scheme they are resampled by documents. Units per source: bangla_plos 139 documents (n=139), ccnews_jsonld 1092 clusters (n=6,050), indonews 100 documents (n=100), l3cube_indic 2361 documents (n=2,361), masakhanews 6 clusters (n=3,339), mnds 127 clusters (n=755).

### (a) Test macro-F1, 95% CI

| Model | Macro-F1 | CI unstratified (paper) | CI stratified |
|---|---|---|---|
| Teacher XLM-R-large | 0.598 | 0.546–0.643 | 0.558–0.635 |
| Teacher XLM-R-large, ONNX int8 | 0.600 | 0.547–0.646 | 0.560–0.638 |
| DistilmBERT + KD | 0.534 | 0.481–0.618 | 0.493–0.576 |
| MiniLM-L6 + KD | 0.566 | 0.511–0.624 | 0.520–0.613 |
| StaticKD (proposed) | 0.545 | 0.490–0.605 | 0.501–0.585 |
| StaticKD-compact | 0.484 | 0.421–0.597 | 0.438–0.530 |
| fastText + KD, subword, quantized | 0.439 | 0.370–0.561 | 0.383–0.495 |
| fastText + KD, subword | 0.447 | 0.385–0.566 | 0.399–0.499 |
| fastText + KD, word | 0.445 | 0.380–0.563 | 0.400–0.495 |
| TF-IDF + SVM + KD | 0.486 | 0.431–0.585 | 0.445–0.533 |
| Model2Vec classifier + KD | 0.547 | 0.495–0.599 | 0.509–0.580 |
| SD-style + KD, 0.13B | 0.431 | 0.374–0.538 | 0.385–0.479 |
| SD-style + KD, 0.27B | 0.430 | 0.372–0.538 | 0.386–0.479 |
| fastText, GPT-4o labels, subword | 0.086 | 0.069–0.119 | 0.073–0.105 |
| fastText, GPT-4o labels, word | 0.106 | 0.085–0.150 | 0.088–0.126 |
| MiniLM-L6 + KD, 512 tokens | 0.580 | 0.526–0.633 | 0.537–0.625 |
| DistilmBERT + KD, 512 tokens | 0.547 | 0.497–0.621 | 0.506–0.586 |
| StaticKD, frozen embeddings (3-seed ens.) | 0.555 | 0.506–0.602 | 0.519–0.589 |

### (b) Paired Δ macro-F1 = StaticKD − model (Holm over the 14 comparisons of the significance table)

| Model | Δ | 95% CI unstratified | Holm p | 95% CI stratified | Holm p |
|---|---|---|---|---|---|
| Teacher XLM-R-large | -0.053 | [-0.065, -0.033] | <0.001 | [-0.063, -0.043] | <0.001 |
| Teacher XLM-R-large, ONNX int8 | -0.055 | [-0.067, -0.035] | <0.001 | [-0.064, -0.045] | <0.001 |
| DistilmBERT + KD | +0.010 | [-0.026, +0.041] | 1.000 | [-0.000, +0.019] | 0.136 |
| MiniLM-L6 + KD | -0.021 | [-0.038, -0.008] | 0.005 | [-0.038, -0.012] | <0.001 |
| StaticKD-compact | +0.060 | [+0.003, +0.100] | 0.042 | [+0.049, +0.068] | <0.001 |
| fastText + KD, subword, quantized | +0.106 | [+0.037, +0.145] | <0.001 | [+0.082, +0.121] | <0.001 |
| fastText + KD, subword | +0.098 | [+0.036, +0.134] | <0.001 | [+0.076, +0.108] | <0.001 |
| fastText + KD, word | +0.099 | [+0.037, +0.141] | <0.001 | [+0.082, +0.108] | <0.001 |
| TF-IDF + SVM + KD | +0.059 | [+0.009, +0.093] | 0.040 | [+0.044, +0.067] | <0.001 |
| Model2Vec classifier + KD | -0.002 | [-0.009, +0.010] | 1.000 | [-0.009, +0.009] | 0.724 |
| SD-style + KD, 0.13B | +0.114 | [+0.060, +0.145] | <0.001 | [+0.094, +0.125] | <0.001 |
| SD-style + KD, 0.27B | +0.115 | [+0.058, +0.148] | <0.001 | [+0.095, +0.125] | <0.001 |
| fastText, GPT-4o labels, subword | +0.458 | [+0.415, +0.496] | <0.001 | [+0.425, +0.489] | <0.001 |
| fastText, GPT-4o labels, word | +0.439 | [+0.393, +0.475] | <0.001 | [+0.408, +0.469] | <0.001 |
| MiniLM-L6 + KD, 512 tokens (Holm over 2 (paper)) | -0.035 | [-0.049, -0.021] | <0.001 | [-0.048, -0.028] | <0.001 |
| DistilmBERT + KD, 512 tokens (Holm over 2 (paper)) | -0.002 | [-0.029, +0.022] | 0.779 | [-0.011, +0.006] | 0.581 |
| StaticKD, frozen embeddings (3-seed ens.) (unadjusted (paper)) | -0.010 | [-0.021, +0.009] | 0.344 | [-0.018, +0.000] | 0.059 |

### (c) TOST ±0.01, the eleven equivalence claims (Holm over 11)

| Test | Δ | 90% CI unstrat. | TOST p | Holm p | 90% CI strat. | TOST p | Holm p | Holm p (k+1)/(B+1) strat. |
|---|---|---|---|---|---|---|---|---|
| cascade_theta0.8_vs_teacher_onnx_int8 | +0.002 | [-0.002, +0.006] | <0.001 | <0.001 | [-0.002, +0.005] | <0.001 | <0.001 | 0.005 |
| ablation_no_tdrop | +0.001 | [-0.000, +0.003] | <0.001 | <0.001 | [-0.000, +0.003] | <0.001 | <0.001 | 0.005 |
| ablation_tau0.5 | +0.004 | [-0.001, +0.007] | <0.001 | <0.001 | [-0.001, +0.006] | <0.001 | <0.001 | 0.005 |
| ablation_bigram | +0.002 | [-0.000, +0.003] | <0.001 | <0.001 | [-0.000, +0.003] | <0.001 | <0.001 | 0.005 |
| ablation_idf | -0.001 | [-0.003, +0.001] | <0.001 | <0.001 | [-0.004, +0.000] | <0.001 | <0.001 | 0.005 |
| ablation_conf | +0.000 | [-0.001, +0.001] | <0.001 | <0.001 | [-0.001, +0.002] | <0.001 | <0.001 | 0.005 |
| ablation_gpt_ce | -0.004 | [-0.007, +0.002] | <0.001 | <0.001 | [-0.007, -0.000] | <0.001 | <0.001 | 0.005 |
| ablation_data130k | -0.002 | [-0.007, +0.002] | 0.004 | 0.018 | [-0.006, +0.001] | <0.001 | 0.002 | 0.005 |
| ablation_hard | +0.005 | [-0.001, +0.009] | 0.017 | 0.051 | [-0.001, +0.008] | 0.002 | 0.006 | 0.007 |
| model2vec_clf_vs_statickd | -0.002 | [-0.008, +0.008] | 0.025 | 0.051 | [-0.008, +0.008] | 0.019 | 0.038 | 0.039 |
| ablation_tau2 | -0.005 | [-0.009, +0.006] | 0.032 | 0.051 | [-0.010, +0.007] | 0.035 | 0.038 | 0.039 |

Conclusions that change from the unstratified to the stratified bootstrap (Holm 0.05 significance, TOST equivalence, Holm-surviving equivalence): TOST model2vec_clf_vs_statickd (survives_holm): unstratified [0.0245, 0.051000000000000004, False], stratified [0.019, 0.038, True]; TOST ablation_hard (survives_holm): unstratified [0.017, 0.051000000000000004, False], stratified [0.002, 0.006, True]; TOST ablation_tau2 (survives_holm): unstratified [0.032, 0.051000000000000004, False], stratified [0.0355, 0.038, True] (lists: TOST p, Holm p, flag).

Restricted-inventory paired tests, stratified (Holm 14): Teacher XLM-R-large -0.039 [-0.050, -0.030] p <0.001 (unstrat. p <0.001); Teacher XLM-R-large, ONNX int8 -0.041 [-0.053, -0.032] p <0.001 (unstrat. p <0.001); DistilmBERT + KD +0.023 [+0.002, +0.034] p 0.064 (unstrat. p 0.606); MiniLM-L6 + KD -0.006 [-0.028, +0.005] p 0.281 (unstrat. p 0.606); StaticKD-compact +0.062 [+0.052, +0.071] p <0.001 (unstrat. p 0.048); fastText + KD, subword, quantized +0.114 [+0.089, +0.131] p <0.001 (unstrat. p <0.001); fastText + KD, subword +0.109 [+0.084, +0.126] p <0.001 (unstrat. p <0.001); fastText + KD, word +0.107 [+0.087, +0.121] p <0.001 (unstrat. p <0.001); TF-IDF + SVM + KD +0.085 [+0.058, +0.103] p <0.001 (unstrat. p <0.001); Model2Vec classifier + KD +0.008 [+0.001, +0.014] p 0.057 (unstrat. p 0.069); SD-style + KD, 0.13B +0.128 [+0.105, +0.143] p <0.001 (unstrat. p <0.001); SD-style + KD, 0.27B +0.126 [+0.105, +0.141] p <0.001 (unstrat. p <0.001); fastText, GPT-4o labels, subword +0.503 [+0.480, +0.520] p <0.001 (unstrat. p <0.001); fastText, GPT-4o labels, word +0.487 [+0.463, +0.503] p <0.001 (unstrat. p <0.001).

**What changes for the paper:** The stratified bootstrap narrows every interval (StaticKD 0.501–0.585 instead of 0.490–0.605) because it treats the six sources as fixed and resamples the three single-cluster sources by documents; its inference is conditional on these sources and is a sensitivity analysis, not a replacement. No significance conclusion of the 14-comparison table changes at Holm 0.05: DistilmBERT stays non-significant (Holm p 0.136), MiniLM-L6 stays ahead (p<0.001), StaticKD-compact and TF-IDF + SVM + KD, which were borderline (p 0.042 and 0.040), become clear (p<0.001). The 512-token rows and the frozen ensemble keep their conclusions (frozen ensemble p 0.344 -> 0.059, still n.s.). The only changes are in the equivalence family: under the stratified bootstrap all eleven TOST claims survive Holm over 11 (Model2Vec 0.038, hard labels 0.006, tau=2 0.038), whereas under the paper's bootstrap these three do not (0.051). The paper's sentence that Holm raises the Model2Vec TOST p to 0.051 stays correct; it can add that the stratified bootstrap gives 0.038, but the equivalence should still be called borderline because it depends on how the sources are treated.

## A2. Mean of per-source macro-F1 and restricted-inventory macro-F1 (stratified bootstrap)

| Model | Mean per-source macro-F1 [95% CI] | Δ StaticKD − model [95% CI] | Holm p (14) | Restricted pooled macro-F1 [95% CI] |
|---|---|---|---|---|
| Teacher XLM-R-large | 0.721 [0.699, 0.735] | -0.060 [-0.095, -0.045] | <0.001 | 0.658 [0.636, 0.673] |
| Teacher XLM-R-large, ONNX int8 | 0.714 [0.689, 0.729] | -0.054 [-0.088, -0.038] | <0.001 | 0.660 [0.640, 0.675] |
| DistilmBERT + KD | 0.633 [0.607, 0.656] | +0.028 [-0.013, +0.046] | 0.364 | 0.597 [0.577, 0.614] |
| MiniLM-L6 + KD | 0.684 [0.663, 0.702] | -0.024 [-0.068, -0.007] | 0.012 | 0.625 [0.608, 0.644] |
| StaticKD (proposed) | 0.661 [0.614, 0.678] | -- | -- | 0.619 [0.595, 0.633] |
| StaticKD-compact | 0.469 [0.442, 0.480] | +0.192 [+0.166, +0.204] | <0.001 | 0.557 [0.535, 0.571] |
| fastText + KD, subword, quantized | 0.461 [0.438, 0.480] | +0.200 [+0.155, +0.220] | <0.001 | 0.505 [0.483, 0.526] |
| fastText + KD, subword | 0.461 [0.437, 0.481] | +0.199 [+0.159, +0.219] | <0.001 | 0.510 [0.488, 0.532] |
| fastText + KD, word | 0.441 [0.416, 0.452] | +0.219 [+0.187, +0.234] | <0.001 | 0.512 [0.492, 0.528] |
| TF-IDF + SVM + KD | 0.455 [0.438, 0.476] | +0.206 [+0.161, +0.222] | <0.001 | 0.534 [0.513, 0.557] |
| Model2Vec classifier + KD | 0.654 [0.609, 0.672] | +0.006 [-0.002, +0.015] | 0.364 | 0.611 [0.589, 0.626] |
| SD-style + KD, 0.13B | 0.442 [0.417, 0.466] | +0.219 [+0.176, +0.235] | <0.001 | 0.491 [0.472, 0.508] |
| SD-style + KD, 0.27B | 0.432 [0.406, 0.453] | +0.229 [+0.190, +0.245] | <0.001 | 0.493 [0.473, 0.510] |
| fastText, GPT-4o labels, subword | 0.117 [0.102, 0.129] | +0.543 [+0.498, +0.562] | <0.001 | 0.116 [0.104, 0.126] |
| fastText, GPT-4o labels, word | 0.131 [0.113, 0.148] | +0.529 [+0.482, +0.549] | <0.001 | 0.133 [0.120, 0.145] |
| MiniLM-L6 + KD, 512 tokens | 0.698 [0.678, 0.715] | -0.038 [-0.079, -0.023] | <0.001 (unadj.) | 0.641 [0.622, 0.657] |
| DistilmBERT + KD, 512 tokens | 0.643 [0.619, 0.664] | +0.017 [-0.022, +0.035] | 0.571 (unadj.) | 0.609 [0.589, 0.626] |
| StaticKD, frozen embeddings (3-seed ens.) | 0.670 [0.629, 0.687] | -0.009 [-0.026, -0.002] | 0.016 (unadj.) | 0.614 [0.593, 0.630] |

**What changes for the paper:** The mean of per-source macro-F1 keeps the ranking teacher (0.721) > MiniLM-L6 (0.684) > frozen ensemble (0.670) > StaticKD (0.661) > Model2Vec (0.654) > DistilmBERT (0.633), and every fastText, TF-IDF, SD-style and StaticKD-compact model is about 0.19–0.23 below StaticKD (all Holm p<0.001). StaticKD vs Model2Vec +0.006 [−0.002, +0.015], Holm p 0.364; vs MiniLM-L6 −0.024, Holm p 0.012; vs DistilmBERT +0.028, p 0.364. One claim weakens: on this metric the frozen-embedding ensemble scores significantly higher than StaticKD (−0.009 [−0.026, −0.002], unadjusted p 0.016), whereas the pooled test shows no significant difference. With the stratified bootstrap the restricted-inventory comparisons stay as in r4 (MiniLM lead not significant, Holm p 0.281; Model2Vec +0.008, p 0.057; DistilmBERT +0.023, p 0.064).

## A3. Expected calibration error

Definition: ev.ece: top-1 confidence, 15 equal-width bins (Guo et al., 2017), as notebook part5 (ece.* numbers).

| Set (bootstrap) | StaticKD ECE [95% CI] | Teacher ECE [95% CI] | Δ StaticKD − teacher [95% CI] | p |
|---|---|---|---|---|
| dev_document | 0.041 [0.036, 0.070] | 0.090 [0.072, 0.110] | -0.049 [-0.060, -0.016] | 0.001 |
| test_site-cluster | 0.139 [0.122, 0.165] | 0.245 [0.212, 0.278] | -0.106 [-0.139, -0.073] | <0.001 |
| test_stratified | 0.139 [0.129, 0.151] | 0.245 [0.217, 0.267] | -0.106 [-0.132, -0.079] | <0.001 |

**What changes for the paper:** The calibration claim now has intervals: StaticKD ECE 0.041 [0.036, 0.070] on dev and 0.139 [0.122, 0.165] on test, the teacher 0.090 [0.072, 0.110] and 0.245 [0.212, 0.278]; the difference is −0.049 [−0.060, −0.016] (p 0.001) on dev and −0.106 [−0.139, −0.073] (p<0.001) on test (stratified: [−0.132, −0.079]). The claim that StaticKD is better calibrated than the teacher holds on both sets. Note that bootstrap ECE is biased upward on small resamples, which is why the dev point estimate lies near the lower end of its interval.

## A4. Audited and human labels

### 300 audited articles (copy of r4 item 5, re-derived per article)

| Model | Editorial acc. (k) | Wilson 95% | Audited acc. (k) | Wilson 95% | Gain [95% CI] |
|---|---|---|---|---|---|
| teacher | 78.0% (234) | 73.0%–82.3% | 90.7% (272) | 86.8%–93.5% | +12.7 pp [+8.7, +17.0] |
| MiniLM-L6 + KD | 74.0% (222) | 68.8%–78.6% | 86.0% (258) | 81.6%–89.5% | +12.0 pp [+8.3, +16.0] |
| Model2Vec classifier + KD | 74.0% (222) | 68.8%–78.6% | 82.3% (247) | 77.6%–86.2% | +8.3 pp [+4.7, +12.0] |
| StaticKD (final) | 72.3% (217) | 67.0%–77.1% | 81.7% (245) | 76.9%–85.6% | +9.3 pp [+6.0, +13.0] |
| DistilmBERT + KD | 69.3% (208) | 63.9%–74.3% | 74.7% (224) | 69.5%–79.3% | +5.3 pp [+2.0, +9.0] |
| fastText + KD | 61.3% (184) | 55.7%–66.7% | 67.3% (202) | 61.8%–72.4% | +6.0 pp [+2.7, +9.3] |
| TF-IDF + SVM + KD | 60.7% (182) | 55.0%–66.0% | 66.7% (200) | 61.2%–71.8% | +6.0 pp [+3.3, +9.0] |

17 of the 300 audited items accept two labels (a prediction matching either counts as correct, as in final_model_check.json).

Paired Δ accuracy StaticKD − model (article bootstrap, 2,000):

| vs | Editorial labels Δ [95% CI], p | Audited labels Δ [95% CI], p | Change (audited − editorial) [95% CI], p |
|---|---|---|---|
| teacher | -5.7 pp [-9.7, -2.0], 0.003 | -9.0 pp [-13.3, -5.0], <0.001 | -3.3 pp [-6.7, +0.0], 0.064 |
| MiniLM-L6 + KD | -1.7 pp [-5.3, +1.7], 0.417 | -4.3 pp [-8.7, -0.3], 0.039 | -2.7 pp [-5.7, +0.3], 0.094 |
| Model2Vec classifier + KD | -1.7 pp [-4.7, +1.3], 0.315 | -0.7 pp [-4.0, +2.7], 0.753 | +1.0 pp [-1.0, +3.3], 0.435 |

Gain from editorial to audited labels, teacher-like models (teacher, MiniLM-L6 + KD, DistilmBERT + KD, StaticKD (final), Model2Vec classifier + KD) +9.5 pp on average vs the others (fastText + KD, TF-IDF + SVM + KD) +6.0 pp; difference +3.5 pp [+0.8, +6.5], p 0.012.

### 100 blind double-annotated articles (adjudicated human main topic, no LLM in labelling)

Sources: {'ccnews_jsonld': 67, 'l3cube_indic': 23, 'bangla_plos': 5, 'indonews': 5}; overlap with the 300 audited articles: 0; editorial label = human main topic for 77% of the articles.

| Model | Editorial acc. | Wilson 95% | Human acc. | Wilson 95% | Gain [95% CI] |
|---|---|---|---|---|---|
| teacher | 73% (73) | 63.6%–80.7% | 87% (87) | 79.0%–92.2% | +14 pp [+6, +22] |
| StaticKD (final) | 69% (69) | 59.4%–77.2% | 84% (84) | 75.6%–89.9% | +15 pp [+8, +23] |
| Model2Vec classifier + KD | 69% (69) | 59.4%–77.2% | 83% (83) | 74.5%–89.1% | +14 pp [+7, +21] |
| MiniLM-L6 + KD | 66% (66) | 56.3%–74.5% | 82% (82) | 73.3%–88.3% | +16 pp [+9, +23] |
| DistilmBERT + KD | 69% (69) | 59.4%–77.2% | 78% (78) | 68.9%–85.0% | +9 pp [+0, +18] |
| TF-IDF + SVM + KD | 67% (67) | 57.3%–75.4% | 74% (74) | 64.6%–81.6% | +7 pp [+0, +15] |
| fastText + KD | 63% (63) | 53.2%–71.8% | 73% (73) | 63.6%–80.7% | +10 pp [+3, +17] |

Paired Δ accuracy StaticKD − model on the 100 articles: teacher: editorial -4 pp [-10, +2] p 0.230, human -3 pp [-9, +3] p 0.404, change +1 pp p 0.855; MiniLM-L6 + KD: editorial +3 pp [-3, +9] p 0.401, human +2 pp [-4, +8] p 0.628, change -1 pp p 0.861; Model2Vec classifier + KD: editorial +0 pp [-3, +3] p 1.000, human +1 pp [-2, +5] p 0.768, change +1 pp p 0.661.

Gain teacher-like +13.6 pp vs others +8.5 pp; difference +5.1 pp [+0.8, +10.0], p 0.022.

**What changes for the paper:** The reviewer's pattern is real but it is not caused by the LLM: on the 300 LLM-adjudicated audit labels the teacher-like models gain 3.5 pp more than fastText and TF-IDF (+9.5 vs +6.0 pp, [+0.8, +6.5], p 0.012), and on the 100 blind human-only labels the same gap appears (+13.6 vs +8.5 pp, difference +5.1 pp [+0.8, +10.0], p 0.022). On the audit labels the gap between StaticKD and the teacher widens from −5.7 to −9.0 pp (change −3.3 pp, p 0.064) and StaticKD falls significantly behind MiniLM-L6 (−4.3 pp, p 0.039; editorial −1.7 pp, p 0.417); on the human labels no paired difference of StaticKD is significant (teacher −3 pp, p 0.404). The paper can state that cleaner labels raise all models, raise the teacher-like models somewhat more, and that this also holds for human labels without any LLM step; the ranking teacher > MiniLM-L6 ≥ Model2Vec ≈ StaticKD > DistilmBERT > fastText ≈ TF-IDF is unchanged. The 100-article sample is small (Wilson intervals 13 to 18 pp wide).

## A5. StaticKD with input truncated to 512 tokens

Token counts (StaticKD tokenizer, [UNK] removed): dev median 376, 34.2% > 512, 0.7% > 1,024; heldout median 496, 48.0% > 512, 3.9% > 1,024; test median 582, 58.6% > 512, 5.4% > 1,024. Procedure check: the same code with 1,024 tokens reproduces the stored dev logits (max |Δ| 0.0e+00).

| Set | StaticKD-512 macro-F1 [95% CI] | Acc. | StaticKD-1024 macro-F1 / acc. | Argmax agreement with 1,024 |
|---|---|---|---|---|
| dev | 0.759 [0.716, 0.789] | 0.797 | 0.761 / 0.800 | 97.8% |
| heldout | 0.788 [0.770, 0.800] | 0.814 | 0.783 / 0.811 | 95.3% |
| test | 0.545 [0.490, 0.605] | 0.640 | 0.545 / 0.641 | 94.5% |

Test macro-F1/language 0.621, % teacher 91.1%.

Paired Δ = StaticKD-512 − model (Holm over the 3 comparisons per set and metric):

| Set, metric | vs StaticKD-1024 | vs MiniLM-L6 512 | vs DistilmBERT 512 |
|---|---|---|---|
| dev macro_f1 | -0.002 [-0.013, +0.009] p 0.836 | -0.050 [-0.083, -0.018] p 0.003 | -0.014 [-0.050, +0.019] p 0.836 |
| dev accuracy | -0.003 [-0.011, +0.005] p 0.529 | -0.038 [-0.061, -0.016] p <0.001 | -0.016 [-0.040, +0.007] p 0.388 |
| test macro_f1 | +0.000 [-0.003, +0.004] p 1.000 | -0.035 [-0.049, -0.020] p <0.001 | -0.002 [-0.028, +0.021] p 1.000 |
| test accuracy | -0.001 [-0.004, +0.002] p 0.968 | -0.041 [-0.056, -0.024] p <0.001 | +0.003 [-0.037, +0.040] p 0.996 |

**What changes for the paper:** Truncating StaticKD to 512 tokens changes nothing measurable: test macro-F1 0.545 (Δ +0.000 [−0.003, +0.004]), accuracy 0.640 (Δ −0.001), dev 0.759 (Δ −0.002, p 0.836), although 58.6% of the test articles exceed 512 tokens. At the same 512-token budget MiniLM-L6 stays significantly ahead (test −0.035 [−0.049, −0.020], Holm p<0.001) and DistilmBERT stays indistinguishable (−0.002, p 1.000). The comparison with the 512-token transformers is therefore not driven by StaticKD's longer input.

## A6. Teacher output entropy and confidence on the distillation data

teacher softmax at tau = 1 of the stored logits (data/processed/teacher_labels/*.parquet); entropy in nats, max log(17) = 2.833.

| Part | n | Entropy mean (nats) | Entropy median | Max prob. mean | Max prob. median | Share max prob. > 0.9 | Share < 0.5 |
|---|---|---|---|---|---|---|---|
| emmediatopic_train | 20,000 | 0.120 | 0.036 | 0.970 | 0.996 | 92.7% | 0.7% |
| ccnews_all | 265,999 | 0.210 | 0.047 | 0.938 | 0.994 | 83.1% | 2.0% |
| ccnews | 109,842 | 0.221 | 0.052 | 0.935 | 0.994 | 82.1% | 2.2% |
| ccnews_extra | 156,157 | 0.202 | 0.044 | 0.941 | 0.995 | 83.7% | 1.8% |
| ccnews_in_emm_languages | 25,467 | 0.200 | 0.044 | 0.941 | 0.995 | 83.7% | 1.7% |
| ccnews_other_languages | 240,532 | 0.211 | 0.047 | 0.938 | 0.994 | 83.0% | 2.0% |
| all | 285,999 | 0.204 | 0.046 | 0.941 | 0.995 | 83.7% | 1.9% |

**What changes for the paper:** The teacher is more confident on its own fine-tuning data than on CC-News: on the 20,000 EMMediaTopic training articles mean max probability 0.970, 92.7% above 0.9, mean entropy 0.120 nats; on the 265,999 CC-News articles 0.938, 83.1% above 0.9, entropy 0.210 nats. The difference is tied to the corpus, not the language: CC-News articles in the four EMMediaTopic languages (25,467) give 0.941 / 83.7%, the other CC-News languages 0.938 / 83.0%. Medians are high everywhere (max probability 0.994–0.996), so most soft labels are close to one-hot; the EMMediaTopic part carries the least extra information beyond the arg-max label.

## A7. Filtering use case (test set)

Per-label precision / recall / F1:

| Label | Support | StaticKD P / R / F1 | Teacher P / R / F1 | MiniLM-L6 P / R / F1 |
|---|---|---|---|---|
| arts, culture, entertainment and media | 767 | 0.47 / 0.62 / 0.54 | 0.63 / 0.61 / 0.62 | 0.53 / 0.60 / 0.56 |
| conflict, war and peace | 160 | 0.25 / 0.72 / 0.37 | 0.26 / 0.84 / 0.39 | 0.24 / 0.82 / 0.38 |
| crime, law and justice | 914 | 0.64 / 0.74 / 0.69 | 0.66 / 0.82 / 0.73 | 0.69 / 0.75 / 0.71 |
| disaster, accident and emergency incident | 78 | 0.18 / 0.79 / 0.29 | 0.16 / 0.74 / 0.27 | 0.16 / 0.76 / 0.27 |
| economy, business and finance | 1909 | 0.82 / 0.58 / 0.68 | 0.86 / 0.66 / 0.74 | 0.82 / 0.61 / 0.70 |
| education | 617 | 0.78 / 0.73 / 0.75 | 0.78 / 0.76 / 0.77 | 0.79 / 0.76 / 0.77 |
| environment | 278 | 0.29 / 0.57 / 0.38 | 0.51 / 0.59 / 0.55 | 0.46 / 0.55 / 0.50 |
| health | 1590 | 0.84 / 0.61 / 0.71 | 0.88 / 0.67 / 0.76 | 0.80 / 0.67 / 0.73 |
| human interest | 301 | 0.21 / 0.62 / 0.32 | 0.26 / 0.71 / 0.38 | 0.23 / 0.68 / 0.34 |
| labour | 153 | 0.35 / 0.44 / 0.39 | 0.37 / 0.53 / 0.43 | 0.38 / 0.48 / 0.42 |
| lifestyle and leisure | 447 | 0.60 / 0.55 / 0.57 | 0.66 / 0.64 / 0.65 | 0.60 / 0.60 / 0.60 |
| politics | 2022 | 0.67 / 0.60 / 0.63 | 0.72 / 0.66 / 0.69 | 0.68 / 0.62 / 0.65 |
| religion | 516 | 0.69 / 0.34 / 0.46 | 0.82 / 0.48 / 0.61 | 0.81 / 0.29 / 0.42 |
| science and technology | 383 | 0.71 / 0.52 / 0.60 | 0.73 / 0.54 / 0.62 | 0.75 / 0.54 / 0.63 |
| society | 132 | 0.12 / 0.21 / 0.16 | 0.14 / 0.26 / 0.18 | 0.14 / 0.23 / 0.18 |
| sport | 2245 | 0.94 / 0.82 / 0.88 | 0.97 / 0.87 / 0.92 | 0.94 / 0.86 / 0.90 |
| weather | 232 | 0.87 / 0.82 / 0.84 | 0.87 / 0.84 / 0.86 | 0.86 / 0.82 / 0.84 |

Confidence threshold (keep predictions with max probability ≥ θ):

| θ | StaticKD coverage / precision | Teacher coverage / precision | MiniLM-L6 coverage / precision |
|---|---|---|---|
| 0.5 | 80.8% / 73.2% | 98.2% / 70.8% | 89.7% / 71.7% |
| 0.7 | 67.4% / 78.8% | 93.3% / 72.8% | 77.1% / 77.0% |
| 0.9 | 51.7% / 83.9% | 85.1% / 76.3% | 59.3% / 84.3% |

Per label at θ = 0.7 and 0.9 (StaticKD; precision of kept predictions / recall):

| Label | θ=0.7 P / R | θ=0.9 P / R | Teacher θ=0.9 P / R |
|---|---|---|---|
| arts, culture, entertainment and media | 0.74 / 0.54 (n 557) | 0.78 / 0.42 (n 417) | 0.70 / 0.54 (n 588) |
| conflict, war and peace | 0.36 / 0.59 (n 263) | 0.39 / 0.42 (n 170) | 0.31 / 0.76 (n 396) |
| crime, law and justice | 0.77 / 0.64 (n 759) | 0.81 / 0.52 (n 583) | 0.71 / 0.78 (n 1003) |
| disaster, accident and emergency incident | 0.25 / 0.72 (n 221) | 0.31 / 0.65 (n 164) | 0.20 / 0.74 (n 283) |
| economy, business and finance | 0.88 / 0.49 (n 1067) | 0.91 / 0.40 (n 852) | 0.88 / 0.61 (n 1326) |
| education | 0.84 / 0.65 (n 478) | 0.87 / 0.53 (n 380) | 0.81 / 0.74 (n 570) |
| environment | 0.57 / 0.45 (n 223) | 0.62 / 0.33 (n 146) | 0.55 / 0.48 (n 242) |
| health | 0.88 / 0.45 (n 808) | 0.90 / 0.32 (n 571) | 0.91 / 0.61 (n 1068) |
| human interest | 0.41 / 0.49 (n 361) | 0.55 / 0.36 (n 199) | 0.34 / 0.63 (n 553) |
| labour | 0.36 / 0.31 (n 135) | 0.41 / 0.24 (n 91) | 0.42 / 0.44 (n 163) |
| lifestyle and leisure | 0.78 / 0.38 (n 220) | 0.83 / 0.24 (n 128) | 0.83 / 0.52 (n 283) |
| politics | 0.79 / 0.50 (n 1271) | 0.85 / 0.41 (n 981) | 0.77 / 0.62 (n 1632) |
| religion | 0.80 / 0.14 (n 90) | 0.84 / 0.09 (n 56) | 0.86 / 0.37 (n 219) |
| science and technology | 0.83 / 0.43 (n 198) | 0.92 / 0.32 (n 132) | 0.79 / 0.47 (n 229) |
| society | 0.22 / 0.09 (n 55) | 0.21 / 0.05 (n 28) | 0.14 / 0.14 (n 125) |
| sport | 0.98 / 0.73 (n 1685) | 0.98 / 0.66 (n 1509) | 0.98 / 0.85 (n 1956) |
| weather | 0.90 / 0.76 (n 196) | 0.91 / 0.69 (n 176) | 0.91 / 0.79 (n 203) |

coverage = share of test articles with max probability >= threshold; precision_kept = accuracy of the kept predictions; per label: precision of the kept predictions of that label, coverage_of_predictions = share of that label's predictions that are kept, recall = kept and correct / gold articles of the label.

**What changes for the paper:** For a filtering use case StaticKD's confidence is usable: keeping predictions with max probability ≥ 0.9 retains 51.7% of the test articles at 83.9% precision (≥ 0.7: 67.4% at 78.8%), close to MiniLM-L6 (59.3% at 84.3%). The teacher keeps 85.1% at 76.3% because it is overconfident (A3). Weak labels stay weak for every model: society (F1 0.16), disaster (0.29), human interest (0.32) and conflict (0.37) have low precision; religion has low recall for StaticKD (0.34) and MiniLM-L6 (0.29) against 0.48 for the teacher. A compact supplementary table can be built from the per-label rows.

## A8. Operating regime (stored benchmark, single-core throughput)

| Model | Articles/s (1c) | Cores for 1M/day | Cores for 100M/day | RAM per process (MB) | RAM for 100M/day (GB, ceil processes) |
|---|---|---|---|---|---|
| StaticKD-compact | 601 | 0.0193 | 1.93 | 277 | 0.5 |
| Model2Vec classifier + KD | 568 | 0.0204 | 2.04 | 1,091 | 3.2 |
| fastText + KD, subword, quantized | 545 | 0.0213 | 2.13 | 300 | 0.9 |
| StaticKD (proposed) | 539 | 0.0215 | 2.15 | 501 | 1.5 |
| fastText + KD, subword | 489 | 0.0236 | 2.36 | 1,122 | 3.3 |
| SD-style + KD, 0.27B | 457 | 0.0253 | 2.53 | 998 | 2.9 |
| SD-style + KD, 0.13B | 456 | 0.0254 | 2.54 | 737 | 2.2 |
| TF-IDF + SVM + KD | 433 | 0.0267 | 2.67 | 828 | 2.4 |
| MiniLM-L6 + KD | 16.9 | 0.683 | 68.3 | 1,235 | 83.2 |
| DistilmBERT + KD | 5.05 | 2.29 | 229 | 1,059 | 236.9 |
| Teacher XLM-R-large, ONNX int8 | 0.79 | 14.7 | 1,465 | 1,375 | 1,968.8 |
| Teacher XLM-R-large | 0.343 | 33.7 | 3,374 | 2,950 | 9,721.9 |

median over the stored repetitions per (model, threads); SD-style from the later e1 session; cores = articles per day / 86,400 / single-core throughput (24 h at full load, no headroom); RAM per process = model RSS increase at 1 thread; one single-threaded process per core.

**What changes for the paper:** At the stored single-core speeds, 1 million articles per day need 0.021 cores for StaticKD, 0.68 for MiniLM-L6, 2.29 for DistilmBERT, 14.7 for the ONNX int8 teacher and 33.7 for the PyTorch teacher (as in the paper). 100 million per day need 2.15 StaticKD cores (3 processes × 501 MB ≈ 1.5 GB RAM) against 68 MiniLM-L6 cores, 229 DistilmBERT cores and 1,465 ONNX int8 teacher cores. The numbers assume 24 h at full load with no headroom and single-threaded processes.

## A9. Release check: test-set manifest

- release_repo (dist/static_kd_iptc_classifier): fields doc_id, source, label_origin, lang, label, label_id, source_label, url, domain, warc_file, source_row_id, text_sha1, n_words; CC-NEWS rows 6,050, with warc_file 6,050; byte-offset field: no; WARC-Record-ID field: no; example {'doc_id': 'u00003', 'url': 'https://ar.mehrnews.com/news/1955601/%D8%A8%D8%B2%D8%B4%DA%A9%DB%8C%D8%A7%D9%86-%D8%AE%D9%84%D8%A7%D9%84-%D8%AD%D8%AF%D9%8A%D8%AB%D9%87-%D9%85%D8%B9-%D8%B1%D8%A6%D9%8A%D8%B3-%D9%88%D8%B2%D8%B1%D8%A7%D8%A1-%D8%A3%D8%B1%D9%85%D9%8A%D9%86%D9%8A%D8%A7-%D9%86%D8%B1%D8%AD%D8%A8-%D8%A8%D8%A7%D9%84%D8%B3%D9%84%D8%A7%D9%85-%D8%A8%D9%8A%D9%86-%D8%A7%D9%84%D8%AC%D9%8A%D8%B1%D8%A7%D9%86', 'warc_file': 'CC-NEWS-20250316135544-01176', 'text_sha1': 'f7f5a460a2ac67f63f8f79fed1f83db45d5f85b6'}.
- scripts/build_dist.py output (dist/github): fields doc_id, source, label_origin, lang, label, label_id, source_label, url, domain, text_sha1, n_words; CC-NEWS rows 6,050, with warc_file 0; byte-offset field: no; WARC-Record-ID field: no; example {'doc_id': 'u00003', 'url': 'https://ar.mehrnews.com/news/1955601/%D8%A8%D8%B2%D8%B4%DA%A9%DB%8C%D8%A7%D9%86-%D8%AE%D9%84%D8%A7%D9%84-%D8%AD%D8%AF%D9%8A%D8%AB%D9%87-%D9%85%D8%B9-%D8%B1%D8%A6%D9%8A%D8%B3-%D9%88%D8%B2%D8%B1%D8%A7%D8%A1-%D8%A3%D8%B1%D9%85%D9%8A%D9%86%D9%8A%D8%A7-%D9%86%D8%B1%D8%AD%D8%A8-%D8%A8%D8%A7%D9%84%D8%B3%D9%84%D8%A7%D9%85-%D8%A8%D9%8A%D9%86-%D8%A7%D9%84%D8%AC%D9%8A%D8%B1%D8%A7%D9%86', 'text_sha1': 'f7f5a460a2ac67f63f8f79fed1f83db45d5f85b6'}.
- src/statickd/testset/jsonld.py stores per extracted page: url, canonical, domain, ..., warc_date (WARC-Date header), warc (WARC file name); it does not store the byte offset or the WARC-Record-ID of the record.
- `data/processed/testset_jsonld/unified_test.jsonl` fields: doc_id, source, label_origin, lang, label, source_label, headline, text, url, domain, lid_checked, train_overlap, label_id.

**What changes for the paper:** The release manifest (dist/static_kd_iptc_classifier) gives the WARC file name for all 6,050 CC-NEWS articles plus the URL and a SHA-1 of the text, but no byte offset and no WARC-Record-ID; the extraction code (src/statickd/testset/jsonld.py) does not store them either, and the older dist/github manifest written by scripts/build_dist.py has no WARC field at all. Re-identification therefore requires scanning the named WARC file for the URL. The paper must not claim offsets or record IDs; code was not changed.

## A10. Reference check: Zschache and Hartwig

The current entry `zschache2026energy` in paper_final/references.bib (Scientific Reports, vol. 16, 12717, 2026, doi 10.1038/s41598-026-45023-0) matches the published record. arXiv 2508.14170 (Johannes Zschache, Tilman Hartwig, "Comparing energy consumption and accuracy in text classification inference"; v1 19 Aug 2025, v2 21 Apr 2026) lists the journal reference "Sci Rep 16, 12717 (2026)" and the related DOI https://doi.org/10.1038/s41598-026-45023-0 (https://arxiv.org/abs/2508.14170). The Crossref record of that DOI (https://api.crossref.org/works/10.1038/s41598-026-45023-0) gives: title "Comparing energy consumption and accuracy in text classification inference", authors Johannes Zschache and Tilman Hartwig, Scientific Reports, volume 16, article number 12717, published online 17 April 2026, URL https://www.nature.com/articles/s41598-026-45023-0 (the nature.com page itself redirects to a login wall for automated fetches and was not read). The bib field `pages = {12717}` holds the article number; the title in the bib uses title case, the published title is sentence case.

**What changes for the paper:** nothing; the citation can stay as it is.

## B1. Model2Vec classifier with a linear head

**m2v_lin** (Model2Vec classifier, linear head (normalize=True)): Sequential( (0): Linear(in_features=256, out_features=17, bias=True) ), normalize=True, train 16.2 min on GPU. Pipeline 506 MB; folded fp16 table directory 34.0 MB ({'meta.json': 0.0, 'model.npz': 16.22, 'tokenizer.json': 17.75}). Fold checks (argmax agreement with the trained pipeline): dev: exact form (fp32 table ÷ document norm) 100.00% (max |Δlogit| 2.0e-05), fp16 table plain mean 95.20%, StaticKD inference code 95.20%; test: exact form (fp32 table ÷ document norm) 100.00% (max |Δlogit| 3.1e-05), fp16 table plain mean 96.06%, StaticKD inference code 96.06%.

**m2v_lin_nonorm** (Model2Vec classifier, linear head, normalize=False): Sequential( (0): Linear(in_features=256, out_features=17, bias=True) ), normalize=False, train 33.1 min on GPU. Pipeline 506 MB; folded fp16 table directory 34.0 MB ({'meta.json': 0.0, 'model.npz': 16.22, 'tokenizer.json': 17.75}). Fold checks (argmax agreement with the trained pipeline): dev: exact form (fp32 table) 100.00% (max |Δlogit| 2.0e-05), fp16 table plain mean 100.00%, StaticKD inference code 100.00%; test: exact form (fp32 table) 100.00% (max |Δlogit| 3.6e-05), fp16 table plain mean 99.99%, StaticKD inference code 99.98%.

| Model | Dev macro-F1 [95% CI] | Dev acc. | Held-out agr. | Test macro-F1 [95% CI] | Test acc. | Macro-F1/lang. | % teacher |
|---|---|---|---|---|---|---|---|
| Model2Vec classifier, linear head (normalize=True) | 0.748 [0.708, 0.777] | 0.787 | 0.804 | 0.536 [0.480, 0.602] | 0.625 | 0.602 | 89.7% |
|   same, folded table without the norm (approximation) | 0.738 [0.689, 0.770] | 0.782 | 0.799 | 0.527 [0.470, 0.597] | 0.619 | 0.594 | 88.1% |
| Model2Vec classifier, linear head, normalize=False | 0.770 [0.736, 0.798] | 0.803 | 0.800 | 0.524 [0.467, 0.597] | 0.608 | 0.587 | 87.6% |
|   same, folded fp16 table (StaticKD code) | 0.770 [0.736, 0.798] | 0.803 | 0.800 | 0.524 [0.467, 0.597] | 0.608 | 0.587 | 87.6% |
| Model2Vec classifier + KD | 0.744 [0.701, 0.772] | 0.780 | 0.805 | 0.547 [0.495, 0.599] | 0.642 | 0.619 | 91.5% |
| StaticKD (proposed) | 0.761 [0.717, 0.792] | 0.800 | 0.811 | 0.545 [0.490, 0.605] | 0.641 | 0.622 | 91.1% |

Paired tests (single comparisons, unadjusted):

| Model | Test Δ StaticKD − model [95% CI], p | same, stratified | Test Δ Model2Vec default − model, p | Dev Δ StaticKD − model, p | Held-out agr. Δ StaticKD − model, p |
|---|---|---|---|---|---|
| Model2Vec classifier, linear head (normalize=True) | +0.008 [-0.001, +0.016], 0.077 | [+0.003, +0.015], 0.005 | +0.010 [-0.005, +0.017], 0.166 | +0.013, 0.364 | +0.007, 0.010 |
|   same, folded table without the norm (approximation) | +0.018 [+0.004, +0.025], 0.008 | [+0.009, +0.024], <0.001 | +0.020 [-0.002, +0.029], 0.069 | +0.023, 0.089 | +0.012, <0.001 |
| Model2Vec classifier, linear head, normalize=False | +0.021 [+0.003, +0.029], 0.014 | [+0.011, +0.027], <0.001 | +0.023 [-0.003, +0.032], 0.090 | -0.009, 0.537 | +0.012, <0.001 |
|   same, folded fp16 table (StaticKD code) | +0.021 [+0.003, +0.029], 0.016 | [+0.011, +0.027], <0.001 | +0.023 [-0.003, +0.032], 0.091 | -0.009, 0.537 | +0.012, <0.001 |

**What changes for the paper:** With n_layers=0 and every other library default (normalize=True) the Model2Vec classifier reaches 0.536 test macro-F1 (accuracy 0.625), 0.748 dev and 0.804 held-out agreement; StaticKD is +0.008 [−0.001, +0.016] ahead (p 0.077; stratified p 0.005) and the MLP-head default is +0.010 ahead (p 0.166). This model cannot be folded exactly: the library L2-normalises the pooled vector, so the logit is (sum of table rows) ÷ ||pooled 256-dim embedding|| + b and the norm needs the full embeddings. The fold is exact in that form (100% argmax agreement, max |Δlogit| 3e-5), but the plain 17-logit table without the norm agrees on only 96.1% of the test articles and scores 0.527. Trained with normalize=False (one extra change, needed for a 17-logit table), the linear head folds exactly into a 34 MB fp16 table that runs in the StaticKD inference code (99.98% test argmax agreement with the pipeline; the residual comes from the order of truncation and [UNK] removal in the two tokenisation paths) but scores 0.524 on test (StaticKD +0.021 [+0.003, +0.029], p 0.016) and 0.770 on dev (StaticKD −0.009, p 0.537). So a Model2Vec-trained table of exactly StaticKD's size and speed does not reach StaticKD on the independent test labels. Caveats: one seed per variant (the final recipe's seed SD is 0.004), unadjusted p, and the two training procedures differ in several settings at once (hard vs soft labels, learned token weights, early stopping, 512 vs 1,024 tokens).

## B1b. Linear-head Model2Vec (normalize=False): three seeds and the folded 3-seed ensemble

Seeds 1 and 2 were trained exactly like seed 0 of B1 (m2v_lin_nonorm: n_layers=0, normalize=False, hard teacher labels, every other setting the library default; seed passed to fit(random_seed) and torch.manual_seed), folded exactly into fp16 tables and run with the StaticKD inference code. The ensemble averages the three tables and biases (fp32 mean -> fp16 table), as the StaticKD ensemble.

- m2v_lin_nonorm_s1: training 18.3 min on GPU; fold check (test): fp32 table 100.00% argmax agreement with the pipeline (max |Δlogit| 3.1e-05), fp16 table in the StaticKD code 99.98%; folded size 34.0 MB.
- m2v_lin_nonorm_s2: training 18.3 min on GPU; fold check (test): fp32 table 100.00% argmax agreement with the pipeline (max |Δlogit| 3.1e-05), fp16 table in the StaticKD code 99.99%; folded size 34.0 MB.
- ensemble: 34.0 MB; logits vs the mean of the member logits: dev max |Δ| 2.5e-03, argmax agreement 100.00%, test max |Δ| 4.1e-03, argmax agreement 100.00%.

| Model | Dev macro-F1 [95% CI] | Dev acc. | Held-out agr. | Test macro-F1 [95% CI] | Test acc. | Macro-F1/lang. | % teacher | Mean per-source macro-F1 [95% CI, stratified] |
|---|---|---|---|---|---|---|---|---|
| Model2Vec linear, seed 0 (folded) | 0.770 [0.736, 0.798] | 0.803 | 0.800 | 0.524 [0.467, 0.597] | 0.608 | 0.587 | 87.6% | 0.634 [0.589, 0.648] |
| Model2Vec linear, seed 1 (folded) | 0.757 [0.714, 0.785] | 0.794 | 0.799 | 0.520 [0.459, 0.596] | 0.603 | 0.581 | 86.9% | 0.621 [0.580, 0.638] |
| Model2Vec linear, seed 2 (folded) | 0.753 [0.711, 0.783] | 0.793 | 0.799 | 0.523 [0.465, 0.599] | 0.606 | 0.580 | 87.5% | 0.621 [0.578, 0.637] |
| Model2Vec linear, 3-seed folded ensemble | 0.753 [0.713, 0.784] | 0.795 | 0.800 | 0.525 [0.466, 0.599] | 0.608 | 0.585 | 87.7% | 0.627 [0.584, 0.642] |
| StaticKD (3-seed folded ensemble, paper) | 0.761 [0.717, 0.792] | 0.800 | 0.811 | 0.545 [0.490, 0.605] | 0.641 | 0.622 | 91.1% | 0.661 [0.614, 0.678] |

Paired tests, Δ = StaticKD − model (single comparisons, unadjusted; 2,000 resamples):

| Model | Test macro-F1 Δ [95% CI], p (site-cluster) | same, stratified | Test acc. Δ, p | Mean per-source Δ [95% CI], p (stratified) | Dev macro-F1 Δ, p | Held-out agr. Δ, p |
|---|---|---|---|---|---|---|
| Model2Vec linear, seed 0 (folded) | +0.021 [+0.003, +0.029], 0.016 | [+0.011, +0.027], <0.001 | +0.033, 0.010 | +0.027 [+0.009, +0.038], 0.006 | -0.009, 0.537 | +0.012, <0.001 |
| Model2Vec linear, seed 1 (folded) | +0.025 [+0.003, +0.034], 0.016 | [+0.008, +0.034], <0.001 | +0.038, 0.006 | +0.040 [+0.018, +0.051], <0.001 | +0.004, 0.731 | +0.012, <0.001 |
| Model2Vec linear, seed 2 (folded) | +0.022 [+0.003, +0.031], 0.033 | [+0.010, +0.029], <0.001 | +0.035, 0.011 | +0.039 [+0.019, +0.051], <0.001 | +0.008, 0.372 | +0.012, <0.001 |
| Model2Vec linear, 3-seed folded ensemble | +0.020 [+0.002, +0.028], 0.026 | [+0.008, +0.027], <0.001 | +0.033, 0.015 | +0.034 [+0.016, +0.045], <0.001 | +0.008, 0.418 | +0.011, <0.001 |

Single seeds against the single-seed runs of the final StaticKD recipe (3 vs 10 seeds; paired-seed bootstrap as the ablation table (mean over the 3 Model2Vec seeds minus mean over the 10 final-recipe seeds on the same resample; site clusters on test/held-out, documents on dev); Welch p unadjusted):

| Metric | Model2Vec linear, mean ± SD (3 seeds) | StaticKD recipe, mean ± SD (10 seeds) | Δ StaticKD − Model2Vec [95% CI] | p | Welch p |
|---|---|---|---|---|---|
| dev_macro | 0.760 ± 0.009 | 0.755 ± 0.010 | -0.005 [-0.023, +0.012] | 0.533 | 0.438 |
| heldout_acc | 0.799 ± 0.000 | 0.808 ± 0.001 | +0.008 [+0.003, +0.013] | <0.001 | <0.001 |
| test_macro | 0.522 ± 0.002 | 0.542 ± 0.004 | +0.020 [+0.000, +0.029] | 0.049 | <0.001 |
| test_acc | 0.606 ± 0.003 | 0.640 ± 0.004 | +0.034 [+0.003, +0.051] | 0.015 | <0.001 |

Test macro-F1 per language (mean over languages): Model2Vec linear seeds 0.587, 0.581, 0.580 (mean 0.583); StaticKD recipe seeds mean 0.622 ± 0.004.

**What changes for the paper:** The B1 result holds across seeds. The three single seeds of the foldable linear-head Model2Vec classifier score 0.522 ± 0.002 test macro-F1 against 0.542 ± 0.004 for the ten single-seed runs of the StaticKD recipe (Δ +0.020 [+0.000, +0.029], paired-seed bootstrap p 0.049; Welch p<0.001), 0.606 vs 0.640 accuracy (+0.034, p 0.015) and 0.799 vs 0.808 held-out agreement (p<0.001); on dev they are level (0.760 vs 0.755, p 0.533). The folded 3-seed ensemble gains almost nothing over its members (test 0.525, accuracy 0.608, 87.7% of the teacher) and stays below the StaticKD ensemble: +0.020 [+0.002, +0.028], p 0.026 (stratified [+0.008, +0.027], p<0.001); accuracy +0.033, p 0.015; mean of per-source macro-F1 0.627 vs 0.661 (+0.034 [+0.016, +0.045], p<0.001); per-language mean 0.585 vs 0.622; dev 0.753 vs 0.761 (p 0.418). At identical format, size and inference code (34 MB fp16 table), StaticKD's training recipe therefore gives about 0.02 macro-F1 and 0.03 accuracy more on the independent test labels than a Model2Vec-trained linear head. The paper can state this as a same-format comparison; the remaining caveat is that the two training procedures differ in several settings at once (soft vs hard labels, learned token weights, early stopping, token dropout, 512 vs 1,024 tokens), so the gap cannot be attributed to one component, and the test p-values are unadjusted single comparisons.

## B2. Size-matched multilingual data ablation (20,000 CC-News documents)

Subset: 20,000 of 265,999 CC-News documents, 69 of 69 languages (proportional to the CC-News language shares, largest remainder, >= 1 per language, sampling seed 0); by source {'ccnews_extra': 11762, 'ccnews': 8238}. Training: final recipe, 10 epochs (as emm_only), seeds [0, 1, 2, 3, 4]. Mean training time 25 s.

| Variant | n seeds | Dev macro-F1 | Held-out agr. | Test macro-F1 | Test acc. |
|---|---|---|---|---|---|
| Final recipe (one seed) | 10 | 0.755±0.010 | 0.808±0.001 | 0.542±0.004 | 0.640±0.004 |
| EMMediaTopic only (4 languages, 20k) | 5 | 0.723±0.006 | 0.730±0.001 | 0.491±0.002 | 0.583±0.002 |
| CC-News 20k, all languages (new) | 5 | 0.666±0.004 | 0.766±0.002 | 0.533±0.001 | 0.631±0.002 |

Paired-seed bootstrap (Δ = CC-News 20k − reference; no Holm) and Welch p:

| Reference | Metric | Δ [95% CI] | p | TOST 90% CI, p | Welch p |
|---|---|---|---|---|---|
| full | dev_macro | -0.089 [-0.116, -0.065] | <0.001 | [-0.112, -0.069], 1.000 | <0.001 |
| full | heldout_acc | -0.041 [-0.048, -0.035] | <0.001 | [-0.047, -0.036], 1.000 | <0.001 |
| full | test_macro | -0.009 [-0.019, +0.001] | 0.092 | [-0.017, -0.000], 0.432 | <0.001 |
| full | test_acc | -0.008 [-0.014, -0.001] | 0.025 | [-0.013, -0.003], 0.265 | <0.001 |
| emm_only | dev_macro | -0.058 [-0.086, -0.033] | <0.001 | [-0.080, -0.037], 1.000 | <0.001 |
| emm_only | heldout_acc | +0.036 [+0.024, +0.049] | <0.001 | [+0.026, +0.047], 1.000 | <0.001 |
| emm_only | test_macro | +0.042 [+0.024, +0.057] | <0.001 | [+0.026, +0.055], 1.000 | <0.001 |
| emm_only | test_acc | +0.048 [+0.019, +0.078] | <0.001 | [+0.023, +0.073], 1.000 | <0.001 |

Test macro-F1 by distillation data of the language (mean over seeds):

| Variant | Dev | sufficient (≥500), n=7,644 | few (<500), n=2,243 | none (0), n=2,857 |
|---|---|---|---|---|
| full | 0.755 | 0.615 | 0.482 | 0.403 |
| emm_only | 0.723 | 0.559 | 0.466 | 0.351 |
| ccnews20k | 0.666 | 0.597 | 0.482 | 0.406 |

**What changes for the paper:** Matching the size of the EMMediaTopic-only ablation (20,000 documents) but drawing the documents from CC-News in all 69 languages gives 0.533±0.001 test macro-F1 against 0.491±0.002 for EMMediaTopic only (+0.042 [+0.024, +0.057], p<0.001) and 0.542±0.004 for the full 285,999-document recipe (−0.009 [−0.019, +0.001], bootstrap p 0.092, not equivalent: TOST p 0.432; seed-level Welch p<0.001). About four fifths (82%) of the drop of the EMMediaTopic-only ablation on the test set is therefore due to language and source coverage rather than to the number of documents. The 20k multilingual model matches the full recipe in languages with few or no distillation data (0.482 and 0.406 vs 0.482 and 0.403) and trails it in well-covered languages (0.597 vs 0.615). Data size still matters for the in-distribution criteria: dev macro-F1 0.666 (full 0.755, EMMediaTopic-only 0.723) and held-out agreement 0.766 (full 0.808). The paper's reading of the emm_only ablation as a data-diversity effect is supported; it should not say that size is irrelevant.

## C. CPU benchmarks

Protocol of statickd.benchmark / run_benchmarks.py (fresh process per model and thread count, pinned P-cores, thread limits in the environment, 100/96 documents with batch 8 for the ONNX transformers, 500/3,000 with batch 64 for the light models). Three repetitions in ONE later session (the paper used three separate sessions); the table gives the median and the 1-core min–max. StaticKD, the fp32 Model2Vec classifier and the fp32 ONNX transformers were rerun in the same session as references.

| Model | Latency p50 / p95, 1c (ms) | Throughput 1c (art./s) [min–max] | Throughput 4c | Model RAM 1c (MB) | File (MB) |
|---|---|---|---|---|---|
| distilmbert_kd | 200 / 215 | 4.795 [4.732–4.905] | 16.56 | 1060 | 516 |
| distilmbert_kd_int8 | 78.5 / 87.8 | 11.9 [11.86–12] | 41.42 | 677 | 130 |
| m2v_lin | 2.17 / 3.68 | 527.5 [523.8–534.1] | 1784 | 1092 | 506 |
| m2v_lin_nonorm_folded | 2.22 / 4.08 | 492.9 [490–501.4] | 1738 | 503 | 34 |
| minilm_kd | 58.5 / 64.6 | 15.77 [15.73–16.16] | 55.33 | 1235 | 408 |
| minilm_kd_int8 | 29.9 / 33.2 | 30.78 [30.38–32.31] | 96.96 | 929 | 102 |
| minilm_kd_int8_gather_only | 60.9 / 67.3 | 15.21 [15.04–15.84] | 52.74 | 959 | 133 |
| model2vec_clf | 2.25 / 3.87 | 524.9 [492.5–525.1] | 1737 | 1091 | 507 |
| model2vec_clf_fp16 | 2.3 / 3.8 | 499.1 [496–511.1] | 1460 | 847 | 263 |
| model2vec_clf_int8 | 2.7 / 4.18 | 504.3 [487.4–510.8] | 1591 | 723 | 140 |
| statickd | 2.26 / 3.95 | 473.7 [472.5–479.1] | 1785 | 501 | 34 |

### C1. Model2Vec classifier in fp16 / int8

- fp16: directory 263 MB (fp32 507 MB), loaded dtype float16; test macro-F1 0.547 (fp32 0.547), accuracy 0.642 (fp32 0.642); argmax changes vs fp32 1 of 12,744; dev argmax changes 0; Δ macro-F1 fp16 − fp32 -0.000 [-0.000, +0.000], p 0.819; Δ StaticKD − fp16 -0.002 [-0.009, +0.010], p 0.806.
- int8: directory 140 MB (fp32 507 MB), loaded dtype int8; test macro-F1 0.546 (fp32 0.547), accuracy 0.639 (fp32 0.642); argmax changes vs fp32 324 of 12,744; dev argmax changes 26; Δ macro-F1 int8 − fp32 -0.000 [-0.004, +0.003], p 0.792; Δ StaticKD − int8 -0.002 [-0.008, +0.010], p 0.836.

### C2. MiniLM-L6 + KD and DistilmBERT + KD, ONNX int8 (quantize_dynamic)

| Model | ONNX fp32 / int8 (MB) | Test macro-F1 fp32 ONNX / int8 | Test acc. fp32 ONNX / int8 | Argmax changes int8 vs fp32 ONNX | Δ int8 − fp32 ONNX [95% CI], p | Δ StaticKD − int8 [95% CI], p | Δ acc. StaticKD − int8, p |
|---|---|---|---|---|---|---|---|
| minilm_kd | 408 / 102 | 0.565 / 0.436 | 0.667 / 0.544 | 3963 of 12,744 | -0.129 [-0.151, -0.105], <0.001 | +0.109 [+0.084, +0.128], <0.001 | +0.097, <0.001 |
| distilmbert_kd | 516 / 130 | 0.534 / 0.530 | 0.623 / 0.618 | 630 of 12,744 | -0.005 [-0.008, -0.002], 0.006 | +0.015 [-0.021, +0.045], 0.684 | +0.023, 0.582 |

Stored (PyTorch GPU) fp32 test scores and argmax changes of the ONNX fp32 CPU path vs the stored predictions: minilm_kd: stored 0.566, ONNX fp32 changes 1, int8 Δ vs stored -0.129 (p <0.001); distilmbert_kd: stored 0.534, ONNX fp32 changes 1, int8 Δ vs stored -0.005 (p 0.006). Quantized graph ops: minilm_kd {'Constant': 186, 'Mul': 105, 'Add': 66, 'Unsqueeze': 63, 'Cast': 46, 'MatMulInteger': 38, 'Reshape': 29, 'Concat': 28, 'DynamicQuantizeLinear': 26, 'Transpose': 24, 'Gather': 22, 'Shape': 21}; distilmbert_kd {'Constant': 177, 'Mul': 102, 'Add': 64, 'Unsqueeze': 61, 'Cast': 44, 'MatMulInteger': 38, 'Reshape': 27, 'DynamicQuantizeLinear': 26, 'Concat': 26, 'Transpose': 24, 'Gather': 19, 'Shape': 17}.

MiniLM-L6 quantization diagnostic (dev macro-F1; ONNX fp32 0.783, teacher-style int8 0.535): matmul_pc_rr {'op_types_to_quantize': ['MatMul'], 'per_channel': True, 'reduce_range': True} 0.582 (378 MB, 280 argmax changes); all_tensor {'per_channel': False, 'reduce_range': False} 0.424 (102 MB, 362 argmax changes); matmul_tensor {'op_types_to_quantize': ['MatMul'], 'per_channel': False, 'reduce_range': False} 0.434 (378 MB, 358 argmax changes); gather_only_pc_rr {'op_types_to_quantize': ['Gather'], 'per_channel': True, 'reduce_range': True} 0.784 (133 MB, 7 argmax changes). Best on dev: gather_only_pc_rr, test macro-F1 0.565, accuracy 0.666, 132 argmax changes vs ONNX fp32; Δ StaticKD − it -0.020 [-0.037, -0.006], p 0.002.

### C3. Folded linear-head Model2Vec table

The exactly folded linear-head Model2Vec table (m2v_lin_nonorm_folded, B1) runs in the unchanged StaticKD inference code (spec `static:`) and has the same file size (34 MB), model RAM (503 vs 501 MB) and speed (493 vs 474 articles/s on one core, 1,738 vs 1,785 on four) as StaticKD in this session; it differs only in accuracy (test macro-F1 0.524 vs 0.545, B1). The normalised linear-head pipeline (m2v_lin) runs at 528 articles/s with 1,092 MB RAM and a 506 MB directory, like the MLP-head Model2Vec classifier.

**What changes for the paper:** All speeds in this section come from one later session in which StaticKD ran at 474 articles/s on one core (paper table: 539) and the fp32 Model2Vec classifier at 525 (paper: 568); compare rows within this session, not with Table 9. (1) Model2Vec classifier: fp16 storage halves the file (507 -> 263 MB) and lowers model RAM from 1,091 to 847 MB with 1 changed prediction out of 12,744 and unchanged macro-F1 (0.547); int8 (model2vec's own global-scale quantization, saved and loaded through the pipeline, which has no quantize_to option) gives 140 MB and 723 MB RAM, 324 changed predictions and macro-F1 0.546 (Δ −0.000, p 0.792). Speed is unchanged on one core (499 and 504 vs 525 articles/s) and somewhat lower on four cores (1,460 and 1,591 vs 1,737). StaticKD still uses less RAM (501 MB) and a smaller file (34 MB); the paper's RAM/file comparison with Model2Vec should mention that the Model2Vec classifier's RAM can be cut to about 720–850 MB by storage precision without accuracy loss. (2) DistilmBERT int8 (teacher-style quantize_dynamic) keeps its accuracy (test 0.530 vs 0.534, Δ −0.005 [−0.008, −0.002]; still not different from StaticKD, Δ +0.015, p 0.684), runs 2.5× faster (11.9 vs 4.8 articles/s on one core, 41 vs 17 on four) and needs 677 instead of 1,060 MB RAM and 130 instead of 516 MB on disk; it remains about 40× slower than StaticKD. (3) MiniLM-L6 int8 with the same settings is 2× faster (30.8 vs 15.8 articles/s, 929 MB RAM, 102 MB) but collapses in accuracy (test 0.436 vs 0.565, Δ −0.129 [−0.151, −0.105]; StaticKD +0.109 ahead). The drop comes from quantizing the MatMul layers (MatMul-only 0.582 dev vs 0.783); quantizing only the embedding Gather keeps accuracy (dev 0.784, test 0.565, MiniLM still +0.020 ahead of StaticKD, p 0.002) but gives no speed-up (15.2 articles/s; 959 MB RAM, 133 MB file). Calibration-free dynamic int8 therefore narrows the speed gap only for DistilmBERT, and only by a factor of 2.5; the speed and RAM conclusions of the paper stand. The paper may state that int8 versions of the transformer students were tested and remain 15 times (MiniLM-L6, with the accuracy loss) to 40 times (DistilmBERT) slower than StaticKD on one core.

## Run times

Part A: 1 min 44 s (A1/A2 25 s, A3 16 s, A5 57 s, the rest < 2 s; A4 and A5 were re-run once after fixes, 0.2 s and 57 s). Part B1 (GPU): 66 min 55 s wall in total; training 16.2 min (m2v_lin) and 33.1 min (m2v_lin_nonorm, slowed by RAM pressure: the process held about 8 GB), the rest is inference on three sets and the fold checks. Part B2 (GPU): 21 min 36 s wall; 22–26 s of training per seed, the rest is inference. Part C: quantization and accuracy evaluation 23 min 0 s, MiniLM quantization diagnostic 4 min 30 s, CPU benchmarks (11 configurations × 2 thread counts × 3 repetitions) 16 min 1 s. Benchmarks started only after the editor's EDITOR_DONE flag existed and with no other job of this task running; other desktop programs (VPN client, browser, IDE) were open as before and could not be closed by this task. Follow-up B1b (GPU, seeds 1 and 2 and the ensemble): 45 min 13 s wall; training 18.3 min per seed.
