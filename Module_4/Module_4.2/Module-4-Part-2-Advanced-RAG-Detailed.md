# Module 4, Part 2: Advanced RAG — Detailed Learning Document

**Schedule:** Days 37–44  
**Prerequisite:** [Part 1: Build a Simple RAG](../Module_4.1/Module-4-Part-1-Building-Simple-RAG-Detailed.md), Python and OpenAI API basics  
**Companion:** [Advanced RAG practical notebook](Module-4-Part-2-OpenAI-Practical-Experiments.ipynb)  
**Roadmap:** [Module 4, Part 2](../../GenAI.md#part-2-advanced-rag-days-3744)

Part 1 built a baseline that retrieves passages and answers with citations. Part 2 adds techniques for specific failures. Start with observed failures and measure a baseline; an advanced technique is useful only when it improves an outcome enough to justify its cost and complexity. Parts 3 and 4 cover deeper evaluation and security.

## Learning outcomes

- Combine lexical and semantic retrieval using reciprocal rank fusion (RRF).
- Search small chunks, return useful context, and apply validated metadata filters.
- Rerank candidates, transform questions, and compare their results and costs.
- Route questions, perform bounded correction and multi-hop retrieval.
- Recognize when structured data needs database queries and safely demonstrate text-to-SQL.
- Explain when graph relationships help and select improvements from actual errors.

## Contents

1. [Experimental setup](#experimental-setup)
2. [4.9 Keyword and hybrid search](#unit-49-keyword-and-hybrid-search--day-37)
3. [4.10 Smarter chunks and filters](#unit-410-smarter-chunks-and-filters--day-38)
4. [4.11 Reranking](#unit-411-reranking--day-39)
5. [4.12 Query transformation](#unit-412-query-transformation--day-40)
6. [4.13 Routing and adaptive RAG](#unit-413-routing-and-adaptive-rag--day-41)
7. [4.14 Self-correcting RAG](#unit-414-self-correcting-rag--day-42)
8. [4.15 RAG over structured data](#unit-415-rag-over-structured-data--day-43)
9. [4.16 Graph RAG and pattern selection](#unit-416-graph-rag-and-pattern-selection--day-44)
10. [Capstone and assessment](#capstone-and-assessment)

## Experimental setup

Continue the fictional Northstar company example from Part 1. The companion notebook includes its own small corpus, labeled query sets and a synthetic expenses table. Values and policies are invented. It uses OpenAI embeddings and the Responses API for semantic search and selected model-driven experiments, with local BM25 and SQLite. Install dependencies in the notebook kernel, keep `OPENAI_API_KEY` in a Git-ignored `.env`, and configure `OPENAI_MODEL` and `OPENAI_EMBEDDING_MODEL` for your account. If VS Code does not inherit a terminal variable, restart the kernel or load `.env` from an ancestor directory. Check [current prices](https://developers.openai.com/api/docs/pricing); API calls and optional model reranking can multiply costs. Paid calls are disabled until enabled in the notebook. Text sent for embedding, generation or reranking leaves your machine.

### A fair comparison

Use the same corpus version, question set, expected source IDs, user access scope and result size for each comparison. Record rankings before looking at answers. For a question with at least one relevant source, **hit@k** is 1 if any expected source appears in the first k results, otherwise 0. Average it across questions. This is a small diagnostic, not a complete retrieval evaluation: it does not measure all relevant evidence, ranking quality, or answer support. For multi-part questions label *all* needed sources and inspect coverage separately. Save a held-out set for the final comparison, and record latency, number of calls and tokens in addition to quality.

```text
Question → initial search (semantic, keyword, or both) → optional rerank
         → evidence selection → grounded answer + citations → verify support
                         ↑ optional rewriting, routing, correction
```

A citation points to evidence, but verify the cited passage actually supports the claim. Do not send a document to a model unless the caller is authorized to access it.

---

## Unit 4.9: Keyword and Hybrid Search — Day 37

### Why use keyword search?

Semantic search finds passages with related meaning, but can miss exact product names, IDs, serial numbers, quotations and rare terms. Keyword search uses the actual terms and often handles identifiers better. It can miss paraphrases or synonyms. A good tokenizer preserves the identifiers and languages in your collection; punctuation, case, hyphens, stop-word processing and stemming affect results.

**BM25** ranks documents using term occurrence, term rarity across the corpus and document-length normalization. It is not simply a count of matching words. Its values are not probabilities and are generally not on the same scale as embedding similarities. Implementations may differ in tokenization, scoring and handling very small corpora.

### Combine with reciprocal rank fusion

Run semantic and keyword searches independently, then merge by rank:

```text
RRF_score(document) = Σ 1 / (K + rank_in_each_list)
```

Ranks start at 1. Documents absent from a list contribute zero for that list. `K` is a smoothing constant (often set to 60 as a starting point), distinct from the number of final results. Stable IDs must refer to the same retrieval units across lists; deduplicate within each list before combining. RRF avoids treating incomparable raw scores as directly additive, but still depends on list depth, duplicate handling and which documents each search returns. It can dilute an excellent exact match; measure it.

**Example:** semantic returns `A, B`, keyword returns `B, C`. With K=60, B scores `1/62 + 1/61`, A `1/61`, C `1/62`. B ranks first because both methods retrieved it.

**Practice:** Test five questions containing different exact names or codes. Record expected ID and top three from semantic, BM25 and hybrid. Inspect misses and tokenization. Use [the RRF paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) for further detail.

---

## Unit 4.10: Smarter Chunks and Filters — Day 38

### Search small, send larger context

Index short child passages that pinpoint a fact. Store a `parent_id` for each and retrieve its surrounding section for generation. Example: a child contains “10 calendar days”; its parent also explains which leave requests are affected and who approves them. Deduplicate parent sections if several child matches point to the same parent. Preserve child and parent source IDs and offsets so a cited claim can be traced back to the source. Oversized parents may add noise or exceed context limits; enforce a token budget.

### Contextualized chunks and tables

An isolated passage such as “Submit within 30 days” is ambiguous. Prefix the searchable text with `Document: Expense policy; Section: submission deadline`. Keep this prefix separate from the original evidence and record how it was produced. LLM-generated summaries can be wrong or leak information across access boundaries; verify them and inherit source permissions.

For tables, keep the headers with each row, retain units and specify the table's subject and date. A verbal description may help semantic search, but the original row is the authority. Do not use a language model to add missing values. Use SQL or calculations for aggregation, as in Unit 4.15.

### Derive filters from questions

A question such as “For product Atlas in the 2025 guide, what is error NX-7421?” supplies potential `product`, `year`, and `document_type` constraints. Extract filters with explicit allowed fields and values, validate them, and apply the same filter to every retrieval branch. If the model is unsure, omit a filter or ask for clarification rather than inventing one. An overly strict filter can hide the right document, especially when metadata is incomplete; show no-result behavior and compare to unfiltered search.

User-derived filters are relevance preferences. Mandatory tenant and access filters must come from trusted identity and be enforced independently. Never allow an extracted filter to relax those constraints.

**Practice:** Use three questions with explicit metadata, compare unfiltered and filtered IDs, and inspect both correct and over-restrictive filters. Measure whether the parent context still contains the answer and its exception.

---

## Unit 4.11: Reranking — Day 39

### Two-stage retrieval

A first-stage retriever returns a broad candidate set, for example 10–50 passages. A reranker then considers the question and each candidate together and orders them; return only the top few to the generator. A cross-encoder or model-based grader can capture interactions that a separate query/document embedding may miss. It cannot recover a passage that was never in the candidate set.

```text
retrieve top 20 → rerank those 20 against the question → pass top 3 to generation
```

The numbers are examples, not universal settings. Candidate depth, reranking model and top-k affect latency, cost and quality. A model can misjudge relevance or obey instructions in a passage; it should treat documents as data. Applying a reranker to an already accurate, small set might not improve anything.

**Practice:** Pick five failures without tuning on your final holdout. Save initial rankings. Rerank the same candidates, inspect top three and expected IDs, then measure added calls, tokens and elapsed time. A relevancy grade is an estimate, not proof that a factual answer follows. The notebook uses a general model as a simple reranker, not a specialized cross-encoder; compare against a dedicated reranking model when appropriate.

---

## Unit 4.12: Query Transformation — Day 40

Users often provide underspecified, long or compound questions. Transforming a question changes the search representation while the original question remains the authority for answering.

| Technique | How it works | Failure mode |
| --- | --- | --- |
| Rewrite | Make a concise standalone search query. | Drops an ID, date, negation or qualification. |
| Multi-query | Generate different query formulations; search each. | More calls and redundant or irrelevant variants. |
| RAG-Fusion | Fuse the multi-query rankings with RRF. | Good passages absent from all lists stay absent. |
| HyDE | Generate a hypothetical answer and search using its embedding. | The hypothesis can invent facts; it is a search query, never evidence. |
| Decomposition | Split a multi-part question; retrieve for each part and combine with citations. | A subquestion may omit a necessary dependency or invent an assumption. |

Include the original query in the search variants to reduce rewrite risk. Validate transformed output, preserve identifiers, and limit the number and size of queries. For multi-part questions, record which sources each subquestion requires; a single hit@k can hide missing evidence for one part.

**Practice:** Compare plain search, multi-query fusion and decomposition on ambiguous and multi-part questions. Show original queries, rewritten queries, result IDs, evidence coverage, total embedding/generation calls, latency and cost. HyDE text is not a source and must not appear as a citation or be passed as supporting evidence.

---

## Unit 4.13: Routing and Adaptive RAG — Day 41

A router selects a path before the rest of the pipeline runs. A greeting may need no retrieval, a straightforward policy question may use simple search, and an ambiguous multi-part question may need decomposition. A structured-data question may require SQL instead of searching documents.

| Route | Example | Risk |
| --- | --- | --- |
| Direct | “Hello!” | Accidentally answer a private policy question from memory. |
| Simple retrieval | “What is the expense deadline?” | Wrong or stale passage. |
| Transform and retrieve | “Compare the two leave rules.” | Missed subquestion or added cost. |
| Structured query | “Total approved expenses by category?” | Wrong query, missing permissions or unsafe execution. |
| Clarify | “What is the limit?” with no subject | Extra turn, but avoids guessing. |

Use deterministic rules when they suffice; model routing can help when intent is ambiguous. Validate labels and provide a safe fallback. Routing does not replace authorization and must not permit a private question to bypass retrieval. Measure route accuracy, false direct-route rate and per-route costs on 20 labeled examples. Include tricky follow-ups and ambiguous questions. Track the expected and actual route and review every mismatch. A model can be confidently wrong, so keep correction or clarification paths.

---

## Unit 4.14: Self-Correcting RAG — Day 42

A corrective loop checks retrieved evidence, makes a bounded adjustment when it is weak, then decides whether enough evidence exists to answer. A separate support check inspects the draft answer against evidence. These labels describe a teaching pattern, not a claim that a few prompts reproduce published Corrective RAG or Self-RAG systems.

1. Retrieve initial candidates and record source IDs and scores.
2. Assess relevance to the specific question; inspect the grader's output and handle malformed results.
3. If evidence is weak, rewrite while preserving IDs and retry at most once or another fixed limit.
4. Generate only from retrieved source text. If still weak, abstain or request clarification.
5. Check each claim and citation against the retrieved evidence. On failure, correct within a fixed budget or abstain.

An LLM grader can misclassify or be influenced by malicious text, and model self-approval does not prove support. Human-reviewed tests and independent evaluation remain necessary. Log the initial and retried queries, evidence, verdict, stop reason and call count. A retry that yields more text is not necessarily an improvement.

**Multi-hop:** a question may require two related facts: find which team owns a product, then look up that team's contact procedure. Retain the source for each hop, verify the intermediate entity and stop if it is ambiguous. Do not turn an unsupported intermediate guess into a new fact.

**Practice:** select five weak questions. Record baseline and retry results, whether needed evidence was found, final answer support, retries, cost and latency. Cap retries even if repeated rewrites seem promising.

---

## Unit 4.15: RAG over Structured Data — Day 43

Vector search is helpful for finding relevant rows or descriptions, but generation from a few retrieved rows is unreliable for counting, summing, grouping and range filters. For those tasks, query the authoritative table and show the result and query or calculation.

### Text-to-SQL flow

1. Provide a minimal schema: table/column names, types, descriptions, example categorical values, and examples of intended queries.
2. The model proposes a query. Treat generated SQL as untrusted input; validate it and execute only under a restricted database identity.
3. Use read-only access, table/column/function allowlists, resource limits, result limits, and audit logs. Prefer parameterized, vetted query templates where practical.
4. Execute, inspect errors or surprising results, and optionally ask for a bounded repair. Never expose unrestricted database errors or secrets to an untrusted model.
5. Explain using the returned rows, show the query and units, and disclose missing data or zero results.

A `LIMIT` controls returned rows, not the amount of data scanned. SQL parsing with regular expressions is not a reliable authorization boundary. Restrict permissions at the database level, handle timeouts, and validate tenant access independently. A SQLite demo is not a complete security solution for an exposed production database.

### Five-table-question experiment

Use a synthetic CSV of expenses, load it into a local database, and compare document search with database answers to: count rows, sum approved amounts, sum a category, filter by employee and count pending rows. Show the rows and the exact query. The notebook demonstrates hand-written baseline SQL and an optional model-generated query in a deliberately constrained, in-memory sandbox with query-only and authorizer checks. Never connect the demo executor to a real or privileged database. Compare against known totals and check whether the model selected the right filters and aggregation. Empty rows and errors must not be explained as successful facts.

---

## Unit 4.16: Graph RAG and Choosing a Pattern — Day 44

A knowledge graph represents **entities** as nodes and **relationships** as edges, such as `Atlas → owned_by → Platform` and `Platform → supported_by → Service Desk`. Relationship questions can traverse two or more edges rather than hoping a single chunk contains the entire answer. Store source citations with every edge and inspect entity resolution: two similar names may refer to different entities. A wrong edge can contaminate all downstream paths.

Some Graph RAG systems also create community summaries for broad collection-wide questions. A tiny hand-built graph demonstrates traversal but does not implement full graph extraction, clustering or global summarization. Graph construction, updates, deletions, confidence, access control, evaluation and model calls add complexity. Preserve source permissions along every edge and summary; graph connectivity alone does not authorize a user to see a relationship.

### Choose from observed failures

| Observed failure | Candidate change | What to measure |
| --- | --- | --- |
| Exact code missing | Keyword or hybrid search | Correct-ID hit@k and wrong-code matches. |
| Relevant passage retrieved but too low | Rerank | Top-k support, latency and cost. |
| Long or multi-part query misses evidence | Rewrite or decompose | Coverage for each subquestion, preservation of constraints. |
| A good small match lacks context | Parent/child chunks | Relevant context and token usage. |
| Weak evidence | Bounded corrective retry | Recovery rate and false positives. |
| Count or sum is wrong | Structured database query | Exact result, SQL correctness and permissions. |
| Several relationships must be connected | Graph traversal | Edge provenance, complete path and correctness. |
| Every query goes through expensive search | Routing | Route accuracy, missed private queries, latency. |

Each addition can introduce failure; keep the simple baseline if the improvement is not measurable on unseen cases.

---

## Capstone and assessment

Build an adaptive assistant over a small synthetic product manual and expenses dataset. Preserve the Part 1 baseline. Create at least 20 route-labeled questions and separate development from held-out questions. Include five exact-code questions, three metadata-filter examples, five poor baseline retrieval cases, ambiguous and multi-part questions, at least five aggregate questions, and a two-hop relationship question. Labels may overlap, but document the overlap.

1. Establish baseline source hit@3 and inspect citation support; preserve results.
2. Compare keyword, semantic and hybrid on the same code questions.
3. Try parent context and filters; inspect three filtered cases including an empty result.
4. Rerank identical candidate sets for five weak questions and measure the added cost.
5. Compare direct retrieval, multi-query fusion and decomposition; inspect evidence for every part.
6. Measure the 20-question router and review false-direct routes.
7. Add one bounded retry; compare failures before and after with stop reasons.
8. Compare five SQL results to independently calculated totals and show the SQL.
9. Walk a graph path with source IDs and map each remaining failure to a possible improvement.
10. Freeze your choices, run the held-out cases once, and report what improved, what worsened, cost, latency and limitations.

### Questions to check your understanding

1. Why can BM25 find an identifier that semantic search misses, and why can it miss a paraphrase?
2. If two ranking lists use different scores, why can RRF combine them without normalizing their scores?
3. Can reranking recover a passage absent from the candidate pool?
4. Why might returning parent sections reduce answer errors but increase cost?
5. What happens when a filter is over-specific or metadata is missing?
6. Why must HyDE text never be treated as retrieved evidence?
7. How would you catch a router that sends a private policy question directly to the model?
8. What ends a correction loop when retrieval remains weak?
9. Why is `SELECT` plus `LIMIT` alone insufficient as a SQL safety policy?
10. How do you verify both edges in a two-hop answer?

### Completion checklist

- [ ] I compared techniques on the same labeled cases and saved both successful and failed examples.
- [ ] I tracked citation support, latency and API use as well as retrieval hits.
- [ ] I can explain that inferred filters, model graders and prompt instructions are not authorization.
- [ ] I measured or explicitly marked untested the model-driven experiments.
- [ ] I preserved a held-out set and documented the remaining failure modes.

## Further reading

- [OpenAI embeddings guide](https://developers.openai.com/api/docs/guides/embeddings), [Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create), [models](https://developers.openai.com/api/docs/models), and [pricing](https://developers.openai.com/api/docs/pricing).
- [Original RRF paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), [HyDE paper](https://arxiv.org/abs/2212.10496), [Corrective RAG paper](https://arxiv.org/abs/2401.15884), and [Self-RAG paper](https://arxiv.org/abs/2310.11511).
- [SQLite authorizer](https://www.sqlite.org/c3ref/set_authorizer.html) and [SQLite query-only pragma](https://www.sqlite.org/pragma.html#pragma_query_only).

APIs, package interfaces, model availability and prices change; verify against current documentation before use.
