# Article-level audit of editorial labels in the unified IPTC test set: protocol

Written 2026-11-18, before any sampling. This file is not edited after sampling; changes are
appended at the end as dated "Deviation" sections.

## 1. Purpose and population

Editorial labels (newsroom JSON-LD `articleSection` and publisher categories) were mapped to the
17 top-level IPTC Media Topics through a name-level mapping table. Only the mapping table was
checked by humans. This audit measures, on a random sample of articles, how often the mapped
editorial label matches the main topic of the article content. Nothing is removed from the test
set; the audit only reports numbers.

- Data: `data/processed/testset_jsonld/unified_test.jsonl` (12,744 rows).
- Audit population: rows with `label_origin` in {`publisher_section_jsonld`, `publisher_category`}
  (8,650 rows). Strata = `source`: ccnews_jsonld (6,050), l3cube_indic (2,361), bangla_plos (139),
  indonews (100).
- `data/processed/testset_jsonld/audit_sample.csv` is not used.
- Label strings are the 17 strings of `src/statickd/config.py::LABELS`. The repository string
  `politics` is the IPTC topic whose official name is `politics and government`; definitions come
  from the official IPTC file `src/statickd/testset/iptc_reference/mediatopic_en-GB.json`
  (level-1, non-retired concepts).

## 2. Sample

- Seed 20261118, `rng = numpy.random.default_rng(20261118)`.
- Stratified by source, fixed sizes: ccnews_jsonld 200, l3cube_indic 70, bangla_plos 15,
  indonews 15 (total 300). Within each stratum (processed in this order), rows are sorted by
  `doc_id` and drawn without replacement with `rng.choice`.
- The 300 sampled rows are shuffled with the same rng and given item_ids A001..A300 in shuffled
  order (item_id carries no information on source or label).
- `results/label_audit/sample_key.csv` (item_id, doc_id, source, lang, label, source_label, url)
  is written straight to file and not printed or read by the coordinating agent before
  `llm_labels_A.jsonl` exists.

## 3. Blind input

- `results/label_audit/blind_input.jsonl`: item_id, lang, headline, text. Text = the test-set
  `text` cut with `statickd.config.truncate` (first 512 whitespace words; first 2,000 characters
  for ja/zh/th/my/km/lo), i.e. the model input. Nothing else is stripped.
- Subagents get plain-text batch files rendered from this file (item_id, lang, headline, text;
  long lines wrapped for readability). The batch files are placed in a scratch directory outside
  the repository, and the subagent is told the absolute path of its single batch file only.
  Copies of all batch files and raw outputs are stored in `results/label_audit/batches/`.
- Blindness: no subagent ever receives label, source_label, source, url, domain, the doc_id, or
  any hint of which candidate came from the newsroom. Every labeling, adjudication, spot-check
  and translation call uses a fresh subagent.

## 4. Annotator A (blind labeling)

- Items in item_id order are split into batches of at most 50 items and at most 80,000
  characters of text (whichever limit is reached first), one fresh subagent per batch, run in
  parallel.
- Output validated: valid JSON per line, every item_id of the batch exactly once, `primary` and
  non-null `secondary` in the 17 strings (the string `politics and government` is normalised to
  `politics`), `secondary != primary` (else set to null), confidence in {high, medium, low},
  `unusable` boolean. Invalid or missing items are re-run in a new fresh subagent with the same
  prompt (only those items). Result: `results/label_audit/llm_labels_A.jsonl`.

### Prompt (Annotator A and Annotator B, verbatim; {placeholders} filled per batch)

```
You are an expert news-topic annotator. Read the file {batch_path}. It contains {n} news articles. Each article has an item_id, a language code, a headline (may be empty) and the article text (possibly truncated). Use only this file: do not open, list or search any other file or directory, and do not use web tools.

For each article choose the IPTC Media Topic (top level) that best describes the MAIN subject of the article content. Judge what the article is actually about, not where a newspaper would file it or which desk wrote it.

The 17 allowed labels (write them exactly as given before the colon) with their official IPTC definitions:
{label_block}

For every article output one JSON object with exactly these keys:
- "item_id": the item_id as given
- "primary": one label, exact string from the list
- "secondary": a second label (exact string) or null. Give one only if the article is genuinely about two topics in substantial measure; otherwise null.
- "confidence": "high", "medium" or "low" (your confidence in "primary")
- "rationale": English, at most 25 words
- "unusable": true if the text is a teaser, boilerplate (navigation, cookie or subscription text), not a news article, or too short to judge; otherwise false. Still give your best "primary" when unusable is true.

Write all objects to {out_path} as JSON Lines (one object per line, UTF-8, in item order, no other text) using the Write tool. Then reply with only the number of lines written.
```

`{label_block}` is one line per label, in the fixed order of `LABELS`:
`<label>: <official definition>`; the politics line reads
`politics: (official IPTC name "politics and government") <definition>`.

## 5. Comparison

- agree = `A.primary == label` (section label) and `A.unusable == false`.
- disagree = everything else (including all items A marks unusable).
- Lenient agreement (reported only) = label in {A.primary, A.secondary}.

## 6. Adjudication (blind)

- Every disagree item goes to a fresh blind adjudicator subagent (batches with the same limits
  as Section 4). Candidates: the section label and A.primary, shown as option_1/option_2 in an
  order drawn with `numpy.random.default_rng(20261119)` (one fair coin per item, items in
  item_id order). If the two candidates are identical (an unusable item with A.primary == label),
  the item is shown with option_1 only.
- The map (item_id, option_1_origin, option_2_origin) is kept only in
  `results/label_audit/adjudication_key.csv`. Output: `results/label_audit/adjudication.jsonl`.
- Validation and re-run rules as in Section 4.

### Prompt (adjudicator, verbatim)

```
You are an expert news-topic annotator acting as adjudicator. Read the file {batch_path}. It contains {n} news articles. Each article has an item_id, a language code, a headline (may be empty), the article text (possibly truncated) and two candidate IPTC labels, option_1 and option_2 (a few articles have only option_1). The candidates come from different annotators and are listed in random order; neither is privileged. Use only this file: do not open, list or search any other file or directory, and do not use web tools.

Decide which candidate best describes the MAIN subject of the article content. Judge what the article is actually about, not where a newspaper would file it or which desk wrote it.

The 17 IPTC labels (exact strings before the colon) with their official IPTC definitions:
{label_block}

Allowed decisions:
- "option_1": option_1 fits the main subject and option_2 does not (or fits clearly worse)
- "option_2": option_2 fits the main subject and option_1 does not (or fits clearly worse)
- "both_acceptable": the article is genuinely multi-topic and both labels fit its main content about equally well
- "neither": no candidate fits; give the correct label in "correct_label"
- "unusable": the text is a teaser, boilerplate, not a news article, or too short to judge
For an article with only option_1, decide "option_1" (it fits), "neither" (+ correct_label) or "unusable".

For every article output one JSON object with exactly these keys:
- "item_id"
- "decision": one of the five values above
- "correct_label": exact label string if decision is "neither", otherwise null
- "confidence": "high", "medium" or "low"
- "rationale": English, at most 25 words

Write all objects to {out_path} as JSON Lines (one object per line, UTF-8, in item order, no other text) using the Write tool. Then reply with only the number of lines written.
```

Adjudication outcome per item: section wins (decision = option holding the section label),
LLM wins (decision = option holding A.primary), both_acceptable, neither, unusable.

## 7. Spot-check of agreed items (Annotator B)

- 60 agreed items, drawn with `numpy.random.default_rng(20261120)`, allocated proportionally to
  the number of agreed items per source (largest-remainder rounding), with at least 5 per source
  where a source has at least 5 agreed items (taken from the largest stratum to keep 60 in total).
- Annotator B = fresh blind subagent(s), same prompt and batching as Annotator A, no candidates.
  Output: `results/label_audit/llm_labels_B.jsonl`.
- B confirms when B.primary == agreed label or B.secondary == agreed label.

## 8. Routing to the human

Only these items go to the human:
1. disagree items whose adjudication confidence is not high, or decision is neither, or decision
   is unusable;
2. spot-check items where B does not confirm.

For routed items with `lang` not in {en, id}, a fresh blind translator subagent translates the
headline and the first 200 words (first 1,000 characters for ja/zh/th/my/km/lo) into English.

### Prompt (translator, verbatim)

```
You are a professional translator. Read the file {batch_path}. It contains {n} news article excerpts, each with an item_id, a language code, a headline (may be empty) and a text excerpt. Use only this file: do not open, list or search any other file or directory, and do not use web tools.

Translate the headline and the text excerpt of every item faithfully into English. Do not summarise, add, omit or comment; keep names and numbers as they are. If the excerpt ends mid-sentence, end the translation at the same point.

For every item output one JSON object with exactly these keys: "item_id", "headline_en", "text_en". Write all objects to {out_path} as JSON Lines (one object per line, UTF-8, no other text) using the Write tool. Then reply with only the number of lines written.
```

`results/label_audit/verify_human.csv` (UTF-8 with BOM): item_id, task, lang, headline_en,
text_en, option_A, option_B, decision, other_label, note.
- task `choose_between` (routed adjudication items with two distinct candidates): option_A /
  option_B = section label and A.primary in an order drawn with
  `numpy.random.default_rng(20261121)`, independent of the adjudicator order. Allowed decision:
  `A`, `B`, `both`, `neither` (then the correct label in other_label), `unusable`.
- task `label_from_scratch` (routed spot-check items, and routed adjudication items with a single
  candidate): options empty. decision = one of the 17 labels or `unusable`; other_label = optional
  second label if the article is genuinely multi-topic.
- For en/id items the text columns hold the original headline and first 200 words.
- No agent suggestion, confidence, rationale or adjudicator choice is shown.
- `results/label_audit/iptc_labels_reference.csv`: the 17 labels with official definitions.

## 9. Final verdict per item

- Agreed item, not spot-checked: assumed correct (its error is covered by the spot-check rate).
- Agreed item, spot-checked, B confirms: correct.
- Agreed item, spot-checked, B does not confirm: human decision. label_from_scratch decision or
  other_label == agreed label -> correct; another label -> wrong; unusable -> unusable;
  empty -> pending.
- Disagree item, not routed: adjudicator outcome (section wins -> correct; LLM wins -> wrong;
  both_acceptable -> both_acceptable).
- Disagree item, routed: human decision mapped the same way (both -> both_acceptable; neither ->
  wrong; unusable -> unusable; empty -> pending). For label_from_scratch routed adjudication
  items: decision or other_label == section label -> correct (other_label match only ->
  both_acceptable), else wrong.

## 10. Metrics (`scripts/label_audit/analyze.py`; empty human decisions = pending)

1. Annotator A vs section label per stratum and pooled: n, agree (strict: primary), lenient
   (primary or secondary), A-unusable count.
2. Adjudication outcomes (section wins / LLM wins / both_acceptable / neither / unusable /
   pending), adjudicator confidence mix, and after human resolution.
3. Spot-check confirmation rate by B with Clopper-Pearson 95% CI; spot-check error rate after
   human resolution (pending excluded), with Clopper-Pearson 95% CI.
4. Estimated editorial-label accuracy per stratum s:
   acc_s = P_s(agree) * (1 - e) + P_s(disagree) * P_s(section correct | disagree),
   where e = pooled spot-check error rate after human resolution (pooled over strata because of
   the small spot-check), P_s(agree) over all audited items of s, and
   P_s(section correct | disagree) over resolved, usable disagree items of s. Strict: correct
   only; lenient: correct + both_acceptable. Population estimate = sum_s (N_s / 8,650) * acc_s.
   95% CI: stratified bootstrap (2,000 resamples of items within each stratum, seed 20261122;
   e recomputed from the spot-checked items in each resample). Unusable and pending items are
   reported separately.
5. Descriptive model check on the 300 items using `results/final/predictions/<model>__test.npz`
   (rows in the order of `unified_test.jsonl`; argmax of logits): teacher, statickd,
   model2vec_clf, MiniLM (`minilm_kd_t1`, the run selected for the paper), distilmbert_kd,
   tfidf_svm_kd, fastText (`ft_kd_subword`). Accuracy vs the section label, and vs the verified
   label (agreed -> section label; adjudicated -> the winning label; both_acceptable -> either
   label counts; neither -> the label given by the human or adjudicator; unusable and pending
   excluded).
6. Outputs: `results/label_audit/results.json`, `results/label_audit/report.md` (interim while
   human items are pending).

## Deviation / additions (2026-11-18, after the agent runs)

- Human sheet size: 44 rows (target about 40). The protocol routes every adjudication whose
  confidence is not high; the adjudicator returned medium or low confidence on 43 of the 68
  disagree items (no neither or unusable decisions), plus 1 spot-check item not confirmed by B.
  The routing rule was not changed.
- Translation batches used a 40,000-character limit (Section 8 named no limit).
- Additional files not named in the protocol: `human_routing.csv` (routed ids, reason, task),
  `human_options_key.csv` (origin of option_A/option_B in the human sheet; do not open while
  deciding), id lists `adjudication_input.jsonl`, `spotcheck_input.jsonl`,
  `translation_input.jsonl`, `item_status.csv` (per-item status), `agent_tool_log.csv` (every
  tool call of the audit subagents, as a blindness check), and `batches/` (rendered batch files
  and raw agent outputs).
- Added to Section 10: a "provisional" estimate that uses the LLM verdict (adjudicator decision,
  or Annotator B's label for a non-confirmed spot-check item, counted as an error) in place of
  pending human decisions. It is reported next to the interim estimate, never instead of it.
- Section 10.4: when a stratum has no resolved usable disagree item, P_s(section correct |
  disagree) falls back to the pooled value over strata (flagged in the report).
- Model check also reports accuracy against the section label on the same subset of items that
  has a verified label, so both columns compare the same items.

## Deviation (2026-11-18, human review)

The author decided all 44 routed items in an interview held by the main Claude session instead of filling the CSV blind. For each item the session showed a one-sentence summary, the two candidate labels without their origin, and its own recommended answer with a short reason; the session had not opened `human_options_key.csv`, `adjudication_key.csv` or `sample_key.csv`. The author accepted the recommendation for 43 items and chose differently for 1 (A008). The human decisions are therefore an author review of LLM suggestions and must be reported as such, not as independent blind human annotation. No second human annotator was used (author decision), MasakhaNEWS was not audited (author decision), and the lenient definition was kept as specified above (author decision).

## Deviation (2026-11-19, post-hoc recheck)

After all decisions were final, the 55 "wrong" and 7 "both_acceptable" items were unblinded and reread by a separate LLM session (`scripts/label_audit/recheck.py`, `recheck_disagreements.csv`, `recheck_summary.md`). It found no mapping errors (all section names map to the IPTC topic they name; the mapping table reproduces every label of the 8,650 audited-population articles), 42 true mismatches between the newsroom section and the article's main topic, 16 items where the editorial label is a legitimate main or co-main topic, and 3 annotation errors that come from systematic LLM tendencies (celebrity stories labelled arts instead of human interest; fashion labelled lifestyle instead of arts). The author decided the three items the recheck flagged: A027 both_acceptable, A032 correct, A222 both_acceptable (`scripts/label_audit/final_status.py`). Estimated editorial-label accuracy: pre-specified procedure 79.3% strict / 81.9% lenient; after the recheck and the author's decisions 80.4% (75.7-84.8) strict / 86.5% (82.6-90.4) lenient. Both sets of numbers must be reported, with the second marked as post-hoc.

## Deviation (2026-11-22, author verification of all items)

The author states that the authors manually verified the final label of all 300 audited articles from the title and an English translation of the opening text and found every final status correct. The statement was recorded on 2026-11-22 as the column `author_verified` in `final_status.csv`; no item changed. Cohen's kappa between Annotator A and the editorial label is 0.75 (232 of 300 agree; an unusable item counts as a disagreement).
