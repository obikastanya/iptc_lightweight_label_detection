# Article-level audit of editorial labels

Generated 2026-11-18 by `scripts/label_audit/analyze.py`. Protocol: `protocol.md`.

Sample: 300 articles, stratified by source; population ccnews_jsonld 6,050, l3cube_indic 2,361, bangla_plos 139, indonews 100.

## 1. Annotator A (blind LLM) vs editorial label

| stratum | n | agree (strict) | agree (lenient) | A unusable | disagree |
|---|---|---|---|---|---|
| bangla_plos | 15 | 10 (66.7%) | 11 (73.3%) | 1 | 5 |
| ccnews_jsonld | 200 | 159 (79.5%) | 179 (89.5%) | 1 | 41 |
| indonews | 15 | 12 (80.0%) | 12 (80.0%) | 0 | 3 |
| l3cube_indic | 70 | 51 (72.9%) | 53 (75.7%) | 0 | 19 |
| all | 300 | 232 (77.3%) | 255 (85.0%) | 2 | 68 |

## 2. Adjudication of disagreements (blind LLM adjudicator)

| outcome | n | high | medium | low |
|---|---|---|---|---|
| llm_wins | 59 | 24 | 34 | 1 |
| both_acceptable | 5 | 0 | 2 | 3 |
| section_wins | 4 | 1 | 2 | 1 |

Final status of disagree items (human decision where routed; pending = not yet decided): wrong 55, both_acceptable 7, correct 6.

## 3. Spot-check of agreed items (blind Annotator B)

60 agreed items (ccnews_jsonld 37, l3cube_indic 13, indonews 5, bangla_plos 5). B confirmed 59/60 = 98.3% (Clopper-Pearson 95% CI 91.1-100.0%). After human review: 0 errors in 60 resolved items (error rate 0.0%, CI 0.0-6.0%); 0 pending.

## 4. Estimated editorial-label accuracy (population-weighted)

acc_s = P_s(agree)(1 - e) + P_s(disagree) P_s(section correct | disagree); strict = section label is the main topic; lenient also counts both_acceptable. 95% CI: stratified bootstrap, 2,000 resamples.

| estimate | strict | 95% CI | lenient | 95% CI | spot-check error e |
|---|---|---|---|---|---|
| interim (pending excluded) | 79.3% | 74.4-83.9% | 81.9% | 77.1-86.3% | 0.0% |
| provisional (LLM verdict for pending) | 79.3% | 74.4-83.9% | 81.9% | 77.1-86.3% | 0.0% |

Per stratum (interim):

| stratum | P(agree) | P(correct given disagree) strict / lenient | resolved disagree | acc strict | acc lenient |
|---|---|---|---|---|---|
| ccnews_jsonld | 79.5% | 12.2% / 22.0% | 41 | 82.0% | 84.0% |
| l3cube_indic | 72.9% | 0.0% / 15.8% | 19 | 72.9% | 77.1% |
| bangla_plos | 66.7% | 20.0% / 20.0% | 5 | 73.3% | 73.3% |
| indonews | 80.0% | 0.0% / 0.0% | 3 | 80.0% | 80.0% |

## 5. Descriptive model check on the audited items

Accuracy (argmax of stored test logits) against the editorial label on all audited items, against the editorial label on the verified subset, and against the verified label (both_acceptable: either label counts; unusable and pending excluded). Provisional = LLM verdicts for pending items.

| model | vs section (all) | vs section (verified subset) | vs verified | n verified | vs provisional | n prov. |
|---|---|---|---|---|---|---|
| teacher | 78.0% | 78.0% | 88.7% | 300 | 88.7% | 300 |
| StaticKD (final) | 72.3% | 72.3% | 80.3% | 300 | 80.3% | 300 |
| Model2Vec classifier + KD | 74.0% | 74.0% | 81.0% | 300 | 81.0% | 300 |
| MiniLM-L6 + KD | 74.0% | 74.0% | 84.3% | 300 | 84.3% | 300 |
| DistilmBERT + KD | 69.3% | 69.3% | 72.3% | 300 | 72.3% | 300 |
| TF-IDF + SVM + KD | 60.7% | 60.7% | 65.3% | 300 | 65.3% | 300 |
| fastText + KD | 61.3% | 61.3% | 65.7% | 300 | 65.7% | 300 |

## Notes

- All LLM roles (Annotator A, adjudicator, Annotator B, translator) were fresh subagents that saw only item_id, language, headline and text (plus two unlabelled candidates for the adjudicator). The agent tool log in `agent_tool_log.csv` shows each subagent read only its own batch file.
- Annotator A, the adjudicator and Annotator B are the same underlying LLM; their agreement is not independent human agreement.
- The agreed items outside the spot-check are assumed correct; the spot-check error rate e is pooled over strata.
- The teacher was fine-tuned on GPT-4o labels, so LLM-verified labels may share its view of the taxonomy. The model check against verified labels can favour the teacher and its students.
