# Module 4, Part 3: Evaluating RAG — Detailed Learning Document

**Schedule:** Days 45–51  
**Prerequisite:** [Part 1: Build a Simple RAG](../Module_4.1/Module-4-Part-1-Building-Simple-RAG-Detailed.md) and [Part 2: Advanced RAG](../Module_4.2/Module-4-Part-2-Advanced-RAG-Detailed.md)  
**Companion:** [OpenAI practical experiments](Module-4-Part-3-OpenAI-Practical-Experiments.ipynb)  
**Roadmap:** [Part 3 in GenAI.md](../../GenAI.md#part-3-evaluating-rag-days-4551)

A RAG answer can sound confident while the search missed a key passage, the prompt omitted an exception, the model invented a claim, or the citation points to the wrong source. Evaluation makes those different failures visible so that you can improve the right component. This part develops an evaluation process; it does not guarantee that a system is safe, correct, or production-ready.

## Learning outcomes

- Read real examples, label failure patterns, and choose measurements that reflect them.
- Build, version, and split a grounded question-and-answer test set.
- Calculate retrieval hit rate, recall and precision, including multi-passage cases.
- Separate retrieval, context assembly, answer support, relevance and citation checks.
- Calibrate a model judge against human labels and inspect disagreements.
- Compare configurations on the same examples, with cost and latency alongside quality.
- Run an automatic regression gate that rejects incomplete or degraded results.

## Contents

1. [Running example and setup](#running-example-and-setup)
2. [4.17 Why evals and reading failures](#unit-417-why-evals-and-reading-your-failures--day-45)
3. [4.18 Test sets and scoring](#unit-418-test-sets-and-scoring--days-4647)
4. [4.19 Measuring retrieval](#unit-419-measuring-retrieval--day-48)
5. [4.20 Evaluating RAG answers](#unit-420-evaluating-rag-answers--days-4950)
6. [4.21 Automatic regression tests](#unit-421-automatic-regression-tests--day-51)
7. [Capstone and self-check](#capstone-and-self-check)

## Running example and setup

Continue the fictional Northstar policies and product guides from Parts 1–2. All questions, answers and policy facts in the notebook are synthetic. A typical record links a question to expected evidence IDs, a reference answer and an `answerable` label. The notebook is self-contained, compares three retrievers, and includes manually labeled synthetic judge examples and a small regression-gate script.

Use Python and a Jupyter kernel with `openai`, `python-dotenv`, and `numpy`. Configure `OPENAI_API_KEY` through an environment variable or a Git-ignored `.env`, never in notebook source or outputs. A VS Code kernel may not inherit a key exported in a separate terminal: restart the kernel or load `.env` from an ancestor directory. Choose models available to your OpenAI project using `OPENAI_MODEL`, `OPENAI_JUDGE_MODEL`, and `OPENAI_EMBEDDING_MODEL`. Paid calls are off by default. The full optional comparison involves many embedding and model calls; consult [current pricing](https://developers.openai.com/api/docs/pricing) before enabling them. Sent evidence, answers and questions leave your machine; use authorized synthetic data.

**Important distinction:** the notebook's sample support labels are prewritten exercise labels, not reviews of live model outputs or evidence that a real judge has been calibrated. Replace them with independent expert labels before drawing conclusions.

---

## Unit 4.17: Why Evals, and Reading Your Failures — Day 45

### Why a few nice examples are misleading

An answer can be fluent and still wrong. Output may vary between runs. A single average can also hide failures for one product, language or question type. Real evaluation uses examples selected before testing, explicit criteria, and repeated inspection. Human feedback and operational monitoring complement automated scoring; neither alone captures every failure.

### Start by reading 20–30 answers

Collect representative, authorized cases: common questions, rare identifiers, missing information, multi-part questions, conflicting or outdated sources, different languages and follow-ups. Include retrieved passages and source IDs, not just the final answer. Redact sensitive data and protect logs. For each case record:

| Field | What to inspect |
| --- | --- |
| Question and expected outcome | Is the user's actual need and answerability clear? |
| Retrieved IDs and passages | Is every necessary fact present? Are irrelevant or stale chunks included? |
| Assembled context | Did truncation, filtering, formatting or deduplication remove a needed exception? |
| Final answer and citations | Does each factual claim follow from its cited source? Is it complete and responsive? |
| Latency, usage and route | Was the outcome worth the time and cost? |

Group failures into a few useful buckets: extraction/index freshness, missing retrieval, context assembly, unsupported answer, incomplete answer, incorrect citation, and operational failure. More than one may apply. Tallying a bucket is a diagnosis aid, not proof of root cause: inspect the actual trace before deciding what to fix.

### The loop

Define success → sample and label examples → run the system → score → inspect failures → change one thing → rerun the same examples → evaluate on held-out cases. For the exercise, inspect 20 cases and record three to five dominant groups. Do not classify missing or failed API runs as successful answers; report coverage separately.

**Try it:** In the notebook, review the 20 deliberately constructed answer examples. Assign your own issue labels before looking at the provided exercise labels, then tally and compare.

---

## Unit 4.18: Test Sets and Scoring — Days 46–47

### Build a golden set

A golden set is a reviewed collection of cases with clear expected behavior. It can include:

```json
{
  "id": "leave-01",
  "question": "How early must planned leave be requested?",
  "answerable": true,
  "expected_source_ids": ["leave"],
  "reference_answer": "At least 10 calendar days before the first day away.",
  "source_version": "northstar-v1",
  "tags": ["policy", "deadline"]
}
```

For an unanswerable case, `answerable` is false, expected IDs may be empty, and the desired behavior is to admit that the corpus does not contain the answer. Verify the absence against the **specified corpus and version**; no source in the test set is not the same as no source anywhere. For multi-part questions, record every needed source. Preserve alternative valid evidence when more than one passage supports the same claim; simple ID overlap alone can underestimate success.

Write cases by hand first. A model can suggest variants or edge cases, but a person should verify every question, answer, evidence span, and label before treating it as a reference. A synthetic generated question can embed a false premise or require knowledge absent from its source.

### Split to avoid tuning on the exam

- **Development set:** inspect and tune prompts, search settings and judge criteria.
- **Held-out set:** lock after labeling; evaluate selected configurations after development. Avoid tuning repeatedly on the same held-out results.
- **Judge calibration set:** independently label representative answers and compare judge decisions. Keep separate examples for testing the finalized judge when possible.

Avoid near-duplicate or same-source leakage between partitions when that would make the intended generalization test too easy. For a small learning set, show its size and state that the results are unstable. Consider slice coverage and real traffic before drawing conclusions.

### Choose the right scorer

| Scorer | Suitable for | Limitation |
| --- | --- | --- |
| Code | JSON schema, required field, exact ID, citation ID exists, arithmetic. | Valid format and exact string match do not prove meaning or support. |
| Model judge | Semantic support or relevance when a simple rule is insufficient. | Can be inconsistent, biased, prompt-injected or wrong. |
| Human | Nuance, policy interpretation, calibration and disagreement review. | Expensive, may disagree, needs a rubric. |

A reference-based answer-correctness check compares to an answer key. A reference-free **support** check asks if the answer is justified by the supplied evidence; an answer can be supported by a wrong or stale source, and a correct answer can still be unsupported. Evaluate factual correctness and document freshness separately.

### Build and calibrate a support judge

Define one binary decision precisely: “Are all substantive factual claims in the answer supported by the provided evidence?” Exclude or handle explicit abstentions consistently. Show question, evidence and answer as data; ask for a pass/fail label and short explanation. An invalid or missing judge result is **unknown**, not pass. Do not let a refusal, empty answer or unsupported speculation pass merely because it contains a citation.

Have a person label at least 20 diverse answers *before* seeing judge outputs. Compare decisions with a confusion matrix:

| | Human support pass | Human support fail |
| --- | --- | --- |
| Judge pass | True positive | False positive: unsupported answer accepted |
| Judge fail | False negative: supported answer rejected | True negative |

Agreement is `(true positives + true negatives) / labeled examples`. Also report false-positive and false-negative counts and class balance: high agreement can hide a judge that always predicts the majority class. Review all disagreements and missing judgments; revise the rubric on calibration data, then test on separate data. Twenty cases are a starting exercise, not sufficient validation for a high-stakes deployment.

**Try it:** Use the notebook's 20 labeled teaching examples, optionally run the judge, print agreement and disagreements, and replace exercise labels with your own independent review.

---

## Unit 4.19: Measuring Retrieval — Day 48

### Common metrics and their denominators

Suppose the gold relevant passage IDs for a question are `{A, B}`, and the top five retrieved are `[A, C, D, E, F]`.

- **Hit@5** = 1 if at least one relevant ID occurs in the first five; here 1. Averaged over queries, it measures how often a hit occurs.
- **Recall@5** = `|retrieved top 5 ∩ relevant| / |relevant|`; here `1/2 = 0.5`. It measures how much labeled evidence was found. With one relevant passage per question, mean recall and hit rate coincide, which can hide incomplete multi-hop retrieval.
- **Precision@5** is often `|relevant in top 5| / 5`; here `1/5 = 0.2`. If the search returns fewer than five, state whether you divide by k or actual returned count. The notebook uses **actual returned count**, and reports empty retrieval as zero for answerable cases. Use the same convention across comparisons.

Deduplicate repeated chunk IDs before calculating; duplicates are not extra relevant evidence. Unanswerable cases with no labeled relevant passages have undefined recall: exclude them from retrieval recall and report their number separately. Evaluate whether the application abstains on them at the answer level. A passage can be useful even if a simple gold set omitted it: adjudicate unexpected candidates and update labels carefully.

### Compare retrievers fairly

Keep the same source and chunk IDs, corpus version, access filters, gold set and k. Compare semantic, hybrid and hybrid plus reranking on identical queries. A reranker only orders its candidate pool; measure candidate recall before reranking so you can see whether it had a chance to recover the relevant passage. Use a stable tie rule, record failures and timeouts, and do not silently drop hard questions.

Report both per-case and macro-averaged metrics. Also record candidate depth, model and index version, latency, embedding/generation tokens and estimated cost. A small apparent gain may be noise; use a larger representative set, paired comparisons and repeated runs where variation matters. Part 2's methods are options to test, not automatically superior configurations.

**Try it:** Measure Recall@5 for semantic, hybrid and hybrid + reranking. The notebook calls a general model as a simple reranker; it is not a specialized trained cross-encoder. Preserve the same hybrid candidate set for the rerank comparison. Compare source IDs and count exceptions.

---

## Unit 4.20: Evaluating RAG Answers — Days 49–50

### Three core checks

1. **Retrieval:** Are the necessary passages present? Measure evidence coverage independently of the generator.
2. **Support / faithfulness:** Are the answer's claims supported by the passages actually provided? A correct-looking claim may come from model memory instead.
3. **Answer relevance and completeness:** Does the answer address all parts of the question and state missing information? A supported but unrelated answer is still not useful.

Also check citation ID validity, whether each cited passage supports its associated claim, correct abstention, source freshness, format, and user-facing tone. Citation validity alone cannot establish entailment. Search quality alone cannot establish answer quality. A judge's decision should not be confused with ground truth.

### Isolate the broken stage

| Observation | Likely investigation |
| --- | --- |
| Needed source missing from retrieval | Parsing, query, filters, embedding, ranking and index freshness. |
| Source retrieved but missing in prompt | Context assembly, token budgeting, truncation and deduplication. |
| Correct passage supplied but answer unsupported | Prompt/model settings, ambiguity and generation behavior. |
| Correct and supported but wrong citation | Citation generation, mapping and claim association. |
| Good quality but slow or expensive | Routing, retry policy, reranking and context size. |

A useful **oracle-context test** supplies the gold passages directly to the generator. If the answer still fails, retrieval is not the only problem; if the oracle answer succeeds but the ordinary pipeline fails, inspect retrieval and context assembly. Oracle tests are diagnostic, not production results. Compare the same questions and instructions, and ensure the gold passages themselves are correct.

### Cost and speed

Measure wall-clock time across the complete request, including retrieval, network and any retries. Record API request count, input/output tokens, embedding usage, judge costs and separately the one-time indexing cost. Use current per-model prices; do not confuse an estimate with an invoice. Report averages and, for enough observations, high-percentile latency. Label cached usage and any failures. Never omit expensive failed requests just because they lack usage data.

### Evaluation libraries

[Ragas](https://docs.ragas.io/) and [DeepEval](https://deepeval.com/docs/getting-started) can help organize RAG metrics and experiments. Inspect each metric's definition, inputs, model calls, expected label format and version before comparing its score to another library or your manual rubric. Library metrics and model judges require calibration and are not guarantees. The notebook implements core checks directly so their denominators remain visible.

**Try it:** Evaluate the same cases through ordinary retrieval and oracle context. For five failures, label the earliest observed failure, supporting evidence, uncertainty, and a possible fix. Check all three core criteria and record incomplete/failed calls separately.

---

## Unit 4.21: Automatic Regression Tests — Day 51

### What to save

For every run record: dataset and corpus versions, source/chunk IDs, questions and gold labels, prompt version, model names and settings, retrieval configuration and candidate depth, timestamps, generated answers, citations, scores, latency, token usage, errors and software versions. Protect private examples and logs. Exact model snapshots and fixed settings help reproducibility but do not guarantee identical outputs; a model name alone may be insufficient.

### A gate with explicit rules

Use the same case IDs and compatible metric/rubric versions for baseline and candidate. Fail closed on missing, duplicate, invalid or errored cases. Compare actual per-case scored outcomes, not just a manually typed summary. Example: candidate support pass rate must not fall by more than 0.05 from baseline; fail if required labels are unknown. Add separate budgets for citation correctness, retrieval recall, cost and latency where justified. An average can mask a critical slice regression, so keep protected-slice and must-pass checks when applicable.

The companion [regression script](rag_regression_gate.py) compares binary support results for identical case IDs. It validates dataset/rubric versions and missing values, reports a paired score comparison and exits nonzero when it fails. It is an example gate, not a comprehensive quality or security gate. Evaluate real candidates with calibrated scorers before supplying them; changing prompts alone does not create valid labels.

```bash
python rag_regression_gate.py --baseline baseline.json --candidate candidate.json --max-drop 0.05
```

The notebook creates synthetic baseline and deliberately worse candidate files in a temporary directory and runs that command, demonstrating both success and failure without an API call. For a genuine bad-prompt comparison, explicitly enable the paid experiment, collect the normal and deliberately weak prompt outputs, judge/review both on the same cases, and inspect the results. A weak prompt is not guaranteed to score worse in every stochastic run; the gate should act on measured labels, never assume a prompt is bad.

### Regression test practice

1. Freeze a reviewed baseline and a dataset/rubric version.
2. Change one prompt, retriever or model setting; run the same cases.
3. Produce scores and verify all expected cases completed; fail on unknowns and missing entries.
4. Run the gate; investigate every regression and severe slice failure.
5. Add newly discovered failures to development data, then evaluate on an independent held-out set.
6. Review quality, estimated cost and latency together; approve a tradeoff deliberately.

No finite gate makes degradation impossible: blind spots, distribution drift, scorer errors, data changes and nondeterminism can evade it. Monitor production with appropriate privacy controls and periodically refresh the test set.

---

## Capstone and self-check

Build an evaluation report for your Part 1/2 assistant:

1. Review at least 20 answers and group failures into 3–5 categories; preserve evidence for each.
2. Curate a versioned golden set with answerable, unanswerable, multi-source, rare-code and ambiguous questions. Verify all references; split development and held-out cases.
3. Compare semantic, hybrid and hybrid + reranking using Recall@5, precision and hit rate, using the same candidate depth and dataset.
4. Independently label 20 answers for support, run the judge, show confusion counts and inspect disagreements. Retest a revised judge on new cases.
5. Compare retrieval, context assembly, support and completeness; run an oracle-context test and diagnose the five worst failures.
6. Capture quality, latency and cost; save settings, failures and data versions.
7. Run the regression script against a baseline, then prove it fails on deliberately degraded scored results. Optionally run the paid weak-prompt experiment and evaluate its actual outputs.
8. Freeze your choices and report held-out results and limitations; do not present toy synthetic results as production evidence.

**Self-check:**

- What is the difference between hit@5 and recall@5 when two sources are needed?
- Why is recall undefined for a question with no relevant passage?
- Can an answer be supported but irrelevant, or correct but unsupported?
- How can you distinguish retrieval failure from model failure?
- Why can 90% judge agreement still hide a serious problem?
- What should a regression gate do with a missing case or unknown label?
- What information is missing from token usage when estimating the full cost of a system?

## Further reading

- [OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices) and [Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create).
- [OpenAI model catalog](https://developers.openai.com/api/docs/models), [pricing](https://developers.openai.com/api/docs/pricing), and [deprecations](https://developers.openai.com/api/docs/deprecations). OpenAI's documentation notes a transition away from its hosted Evals platform; the exercises here use the Responses API and local scoring instead.
- [Ragas documentation](https://docs.ragas.io/) and [DeepEval documentation](https://deepeval.com/docs/getting-started). Consult current versions and metric definitions.

APIs and packages change. Record versions and verify documentation when you rerun the experiments.
