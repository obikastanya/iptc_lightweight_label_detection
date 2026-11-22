# Independent human annotation of editorial labels: protocol

Written 2026-11-23, before any annotation. This file is fixed once committed; any later change is added
as a dated "Deviation" section at the end and never edits the text above it.

## Purpose
Measure agreement between two human annotators who label articles independently and without access to
any label, and use their labels to estimate how often editorial labels (section labels and publisher
categories) name the main topic of an article. The study complements the LLM-assisted article-level audit
(results/label_audit/) with a sample that the audit did not use.

## Sample
- Population: the 8,650 test articles with editorial labels (label origin "publisher_section_jsonld" or
  "publisher_category"), minus the 300 articles of the label audit.
- Stratified random sample of 100 articles with seed 20261123, in the proportions of the audit:
  CC-NEWS 67, L3Cube-IndicNews 23, Bangla news 5, IndoNews 5 (items H001-H100).
- Calibration set: 10 further articles from the same population (CC-NEWS 6, L3Cube 2, Bangla 1,
  IndoNews 1; items C01-C10). Calibration items are never part of the analysis.
- Script: scripts/human_annotation/build_sheets.py. Answer key (editorial labels, sources, document
  ids): human_annotation/_key/ (not opened by the annotators).

## Annotators and blinding
- Annotators: two of the authors (Obi Kastanya and Ananta Dwi Prayoga Alwy). Neither has seen these 110
  articles before; the label audit used other articles.
- Each annotator gets a workbook with the title, the text (first 512 words), and the language of each
  article, in a random order that differs between the annotators. The workbook shows no editorial label,
  section name, URL, source dataset, LLM label, or model prediction.
- Each workbook also shows an English machine translation of the title and text as a reading aid. The
  translations were made by Claude Haiku 5.5, a different model from the one that produced the LLM labels,
  with the instruction to translate faithfully without summarizing or labeling. The original text stays
  the reference, and annotators may check doubtful passages with Google Translate. No LLM is used to decide
  a topic.
- The annotators do not discuss the items until both have submitted. They do not open
  human_annotation/_key/ or human_annotation/ai_reference/.

## Labels
For each article the annotator records:
- the main topic: exactly one of the 17 top-level IPTC Media Topics (official definitions in the workbook);
- all correct topics: every topic that is as central as the main topic, including the main topic
  (lenient set).
For broken or truncated texts the annotator decides from the title and the readable part of the text.

## Calibration
Before annotating, both annotators label the 10 calibration articles separately, then compare and discuss
them together to align their reading of the IPTC definitions. After calibration the guidelines are frozen.

## Submission
Each annotator submits the completed workbook before seeing the other's. The submitted files are copied to
human_annotation/submitted/ and committed to git (or their SHA-256 hashes are recorded) at submission time,
which gives a time stamp for each independent submission.

## Analysis plan (fixed now)
Computed on the 100 analysis items, before any discussion of disagreements:
1. Agreement between the two annotators on the main topic: Cohen's kappa with a 95% bootstrap interval
   (2,000 resamples of articles), Krippendorff's alpha, and percent agreement.
2. Correct rate of the editorial labels per annotator: strict (editorial label = main topic) and lenient
   (editorial label in the set of all correct topics), with 95% Wilson intervals, overall and by source.
   Overall rates are weighted by each source's share of the 8,650 editorial labels, as in the audit.
3. Agreement of each annotator with the blind LLM labels (Cohen's kappa), to compare the LLM annotator
   with human annotators.
4. Results by source are reported descriptively (small strata).
Rule for extension: if the pre-discussion kappa between the annotators is below 0.65, 50 further articles
are drawn from the same population with seed 20261124 and annotated under the same protocol, and all
analyses are reported on the 150 articles.

## Adjudication
After the analysis above, the two annotators discuss their disagreements and agree on one main topic and
one set of correct topics per article. The adjudicated labels give a further correct-rate estimate, which
is reported next to the per-annotator rates. The pre-discussion kappa remains the reported agreement.

## LLM labels
Blind LLM labels for the 100 items are produced from the same title and text (no section name, URL, or
source) by Claude Opus 5.5 and stored in human_annotation/ai_reference/ai_labels.xlsx. They are used only
for analysis item 3 and are not shown to the annotators.

## Release
The guidelines, both annotators' labels, the adjudicated labels, the LLM labels, and the analysis script are
released with the code. Article texts are not redistributed.
