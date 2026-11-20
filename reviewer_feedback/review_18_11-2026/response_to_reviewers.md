# Response to Reviewers

**Manuscript:** StaticKD: Distilling a Multilingual IPTC News Topic Classifier into a Static Token-Logit Table for Lightweight CPU Inference

**Reviews of 16 November 2026:** Reviewer 1 (editor and senior reviewer, minor revision) and Reviewer 2 (major revision). Both reviews were written in Indonesian; the comments below are our English summaries of them.

We thank both reviewers for their careful reading. The revision adds three analyses that the reviews asked for. An article-level audit checks the editorial test labels, the cascade is measured end to end with a threshold chosen on the development set, and a Sparse-Distillation-style baseline is trained on the same data and budget as StaticKD. Section and table numbers refer to the revised manuscript.

> **Open items in this draft.** [E1] marks results of the Sparse-Distillation-style baseline that are still running. [A1] marks the repository, model and DOI links and the author biographies, which will be filled in before submission. Page numbers will be added to the final version.

---

## Reviewer 1

**1.1 Novelty.** *The token-logit table is a natural continuation of static embedding work such as Model2Vec. Sharpen the novelty claim in one paragraph: the manuscript's own data show that a Model2Vec classifier is about as accurate, so the real advantage lies in memory and deployment simplicity, and that should be the axis of the claim.*

We agree that the contribution is a lighter deployed model and not a new operator. The manuscript attributes the folding identity to Singh et al. (Section II-D and the synthesis at the end of Section II) and states the memory and deployment advantage over the Model2Vec classifier at three points. The abstract and Section VI-C report that the Model2Vec classifier is equally accurate within ±0.01 macro-F1 but needs a file 15 times larger and about twice the RAM. Section VII-D states that StaticKD folds into one table and needs only the `numpy` and `tokenizers` libraries. The conclusion repeats the comparison.

**1.2 Effect sizes and equivalence.** *Differences to the closest baselines are often small; every claim of superiority should state the effect size, and equivalence should be stated explicitly.*

Every comparison reports Δmacro-F1 with its 95% interval (Table 8), and equivalence is claimed only where TOST supports it (Model2Vec classifier; several ablation variants in Table 10). The revision states in Section V-B that TOST p-values are not adjusted for multiple comparisons, so readers can judge each equivalence claim on its own interval. The share of the teacher score that StaticKD keeps now has a bootstrap interval as well (91.1% [88.7–94.8%] on the test set, Section VI-B).

**1.3 Placeholders and artifacts.** *The [TODO] placeholders for the GitHub/Zenodo links, the model link and the author biographies must be filled, and the artifacts deposited before the final decision.*

The code, models, test-set manifest, mapping table, audit data and all predictions are in the release repository [A1: public URL and Zenodo DOI]. The Hugging Face model link and the biographies will be added [A1].

**1.4 Expectation of absolute performance.** *The abstract should place the moderate absolute macro-F1 relative to the teacher so that readers do not think the static table is "free".*

The abstract reports the StaticKD macro-F1 together with the teacher score (0.545 against 0.598, 91.1% of the teacher). The new label audit (Section VI-A) shows that part of the moderate absolute level comes from the labels. On the audited articles the accuracy of StaticKD rises from 72.3% against the editorial labels to 81.7% against the checked labels, and that of the teacher from 78.0% to 90.7%.

**1.5 Consistency of speed and memory figures.** *Make sure the speed and memory numbers agree between abstract, tables and discussion.*

All numbers are produced by one notebook from stored results. The revision labels the 682× figure as a throughput ratio in the abstract, Section VI-C and the conclusion, because the latency ratio (about 592×) is a different quantity. [A4: a final consistency check of every number against the notebook output will be done on the submitted version.]

---

## Reviewer 2

### Major comments

**2.1 Validity of the benchmark labels.** *Two thirds of the labels are newsroom sections or publisher categories mapped with LLM help; agreement with the teacher measures model fit, not label correctness. Annotate a sample blindly, report agreement and adjudication, and recompute the metrics on the gold subset.*

We added an article-level audit of 300 editorial labels, stratified by source (Section IV-D for the design, Section VI-A for the results, supplementary "Label audit" for the procedure).

- An LLM labeled each article from its text without the section name or the URL. A separate blind LLM pass decided every disagreement without knowing which label came from the newsroom and relabeled 60 agreeing articles as a spot check. The authors decided the 44 uncertain cases.
- The audit found **no mapping error**: every sampled section name maps to the IPTC topic it names. The remaining mismatches come from articles that newsrooms filed under another section, such as a car crash in a sports section.
- Editorial labels name the main topic of **80.4% [75.7–84.8%]** of the articles and one of two central topics of **86.5% [82.6–90.4%]** (population-weighted by source). Accuracy is higher for CC-NEWS sections (83.0%) than for the publisher categories of L3Cube-IndicNews (74.3%) and the Bangla news dataset (73.3%).
- **The ranking of the models does not change** against the audited labels; all models score higher against them.

We report the audit's limits in Section VII-E: the labeling and adjudication used an LLM, the authors decided the uncertain cases with LLM suggestions, and there is no second human annotator. A post-hoc recheck after unblinding corrected systematic LLM errors on celebrity and fashion articles; the pre-specified procedure gives 79.3% and 81.9%, and both sets of numbers are reported. The earlier claim that section labels are "not noisier than human labels" is now worded as indirect evidence, noting that the two English samples come from different sources (Section VI-A). The single-label proxy for a multi-label taxonomy remains a stated limitation (Section VII-E).

**2.2 Effective sample size.** *Two publisher groups hold 37.3% of the data and the effective number of sites is about 13.8; 27 languages have fewer than three effective sites. Define the estimand, test leave-publisher-out and site dominance, and do not treat 47 languages as 47 independent replications.*

Section VII-E now states that all test-set intervals apply to the sites and languages of this test set and that the per-language Wilcoxon tests treat languages as independent units although the 10 languages of L3Cube-IndicNews come from a single source. The manuscript already contains the two checks the reviewer asks for. A leave-one-source-out analysis drops each of the six test-set sources in turn, and a second check uses a subset with at most five articles per site and language–label pair (Section V-B, results in Section VI-B and supplementary Table S2). StaticKD stays between third and fifth of the nine models in every leave-one-out run, and only the order of StaticKD and the Model2Vec classifier changes on the capped subset.

**2.3 Novelty and the closest baseline.** *Static tables, average pooling and linear folding have precedents. Do not attribute gains to dark knowledge without evidence. Add the closest prior method under the same data and budget, e.g. Sparse Distillation, or explain the measurable difference.*

The manuscript does not claim a new folding operator; it attributes the identity to Singh et al. and positions StaticKD as a lighter deployed model (see 1.1). It also does not credit dark knowledge: hard teacher labels are equivalent to soft labels within ±0.01 and a higher temperature never helps (Sections VI-D and VII-B).

We added a Sparse-Distillation-style baseline trained on the same 285,999 documents, the same teacher logits and the same tokenizer as StaticKD. It keeps the defining ingredients of Sparse Distillation: unigram, bigram and trigram embeddings trained from scratch, averaged, and passed through a non-linear MLP head trained with KL divergence. Two table sizes (about 0.13B and 0.27B parameters) test whether table size limits accuracy; the original 1B-parameter English table does not fit the hardware used here, which we state in Section VII-E. [E1: results, Δ against StaticKD with interval, CPU cost, and the effect of table size.]

**2.4 Cluster inference and equivalence.** *Define how classes absent from a resample are handled, global macro-F1 versus the per-language mean, how the bootstrap p-value is computed, and the multiplicity correction for equivalence tests.*

Section V-B now states that macro-F1 in each resample is computed over the labels present in that resample, as in the point estimate, and that the two-sided bootstrap p-value is twice the share of resamples whose difference has the opposite sign to the observed one. TOST p-values are stated to be unadjusted. Table 7 reports global macro-F1 and the per-language mean in separate columns. Equivalence with the Model2Vec classifier is a statement about this test set, which the new estimand sentence in Section VII-E makes explicit; the subgroup without distillation data, where the Model2Vec classifier leads, is reported in Section VI-E.

**2.5 Reproduction and deployment claims.** *The 682× headline is a throughput ratio while latency, file size and RSS give other ratios; the cascade throughput is computed from component throughputs; artifact links are TODO. Release the artifacts, measure the cascade end to end, and restrict productivity claims to the tested hardware.*

- The 682× figure is now labeled as a throughput ratio throughout (see 1.5); latency, file size and RAM are reported separately in Table 9.
- The cascade threshold θ = 0.8 is now chosen on the development set, where it gives the highest cascade macro-F1 (0.822 against 0.823 for the teacher alone; Section V-C). On the test set the cascade is equivalent to the teacher in macro-F1 within ±0.01 (0.602 against 0.600, TOST p < 0.001). An end-to-end run on 448 test articles passed 35.0% of them to the teacher and reached 2.18 articles per second against 2.20 from the cost formula (Section VI-F).
- Section VII-E already restricts the efficiency results to one laptop and notes that absolute numbers will differ on a server while ratios measured under the same protocol remain informative.
- Artifacts: see 1.3 [A1].

### Statistical audit notes

| Note | Response |
|---|---|
| Report an interval for the 91.1% retention, not only the ratio | Added: 91.1% [88.7–94.8%] on test and 92.6% [88.4–96.3%] on dev, ratio taken inside each resample (Section VI-B). |
| 682× is a throughput ratio, 592× a latency ratio | Labeled as throughput in abstract, Section VI-C and conclusion. |
| Algebraic folding is not bitwise equality; check fp32 → fp16 | All reported StaticKD scores are computed with the averaged table stored in fp16 (Section III-E). Reloading the released model reproduces the stored test predictions exactly (0 of 12,744 labels differ). |
| Model2Vec equivalence is conditional on the TOST procedure; non-significance against DistilmBERT is not equivalence | Agreed. The manuscript claims equivalence only for the Model2Vec classifier and reports that the DistilmBERT interval is too wide to show equivalence (TOST p = 0.428, Section VI-B). |
| Efficiency claims must acknowledge the MiniLM-L6 gap; file size is not resident RAM | Section VI-B reports MiniLM-L6's significant lead (−0.021); Table 9 and the abstract give file size and RAM separately. |
| Cascade 0.602 vs 0.600 needs a paired interval and a development-chosen threshold | Done (see 2.5). |

### Consistency and reporting notes

| Note | Response |
|---|---|
| Name the two macro-F1 estimands in tables and captions | Table 7 has separate columns for global macro-F1 and macro-F1 per language; Section V-B defines both. |
| Site totals must explain the overlap between sources | The note under Table 3 explains the one site shared by CC-NEWS and MN-DS. |
| The teacher is not a mathematical upper bound on the student | Agreed and removed (Section VII-E); StaticKD beats the teacher in 7 of 47 languages. |
| Hard targets and frozen embeddings weaken the claim that all training components are needed | These results are reported as findings (Sections VI-D and VII-B, conclusion) and support a simpler recipe. |
| The English/French agreement check does not isolate label noise; ECE 0.139 shows weak calibration out of domain | Section VI-A now notes that the two English samples come from different sources, and the article-level audit gives a direct estimate. Section VI-F states that both models become overconfident on the out-of-distribution test set. |
| Release scraping details, licenses, IPTC version, mapping, model commits, seeds and CPU environment | Licenses are in supplementary Table S5 and hyperparameters in supplementary Table S1. The mapping table and its review, the seeds of every run and the CPU environment file are in the repository, and the CPU setup is in Table 6 [A1]. |

### Revision plan items (P0–P2)

| Item | Done in this revision |
|---|---|
| P0: independent validation of the IPTC labels | Article-level audit of 300 editorial labels (2.1). |
| P0: fix the macro-F1 definition and the bootstrap procedure | Section V-B (2.4). |
| P1: closest prior method under equal conditions | Sparse-Distillation-style baseline [E1] (2.3). |
| P1: real cascade measurement with a development-only threshold | Section V-C and VI-F (2.5). |
| P2: release the model, table, tokenizer, per-article predictions, cluster IDs, split manifest, license metadata and scripts | Release repository [A1]. |
