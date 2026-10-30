# Module 4, Part 1: Build a Simple RAG — Detailed Learning Document

**Schedule:** Days 25–36  
**Prerequisites:** Modules 1–3, basic Python, OpenAI API usage and prompt engineering  
**Companion:** [Part 1 practical experiments](Module-4-Part-1-OpenAI-Practical-Experiments.ipynb)  
**Roadmap:** [Module 4 in GenAI.md](../../GenAI.md#module-4-rag-days-2556)

This is the first of four RAG parts: (1) build a simple system, (2) advanced RAG, (3) evaluation, and (4) security. Complete this part before adding advanced techniques. The companion notebook develops a small question-answering system using synthetic documents and OpenAI embeddings and generation.

## Learning outcomes

By the end you should be able to explain when retrieval helps; load, clean, inspect and split documents; embed and search chunks; use a vector store; generate answers with citations; and recognize missing evidence. You should also be able to tell whether a failure happened during extraction, retrieval, or generation.

## Contents

1. [The example and setup](#the-example-and-setup)
2. [4.1 When to use RAG](#unit-41-when-to-use-rag--day-25)
3. [4.2 Loading and cleaning documents](#unit-42-loading-and-cleaning-documents--days-2627)
4. [4.3 Parsing messy documents](#unit-43-parsing-messy-documents--day-28)
5. [4.4 Chunking](#unit-44-chunking--day-29)
6. [4.5 Embeddings and semantic search](#unit-45-embeddings-and-semantic-search--days-3031)
7. [4.6 Vector databases](#unit-46-vector-databases--days-3233)
8. [4.7 Your first RAG pipeline](#unit-47-your-first-rag-pipeline--days-3435)
9. [4.8 Citations and saying I do not know](#unit-48-citations-and-saying-i-do-not-know--day-36)
10. [Capstone, assessment and resources](#capstone-project)

## The example and setup

Imagine a fictional company, Northstar. Its internal handbook has separate articles on leave, expenses, and account support. A user asks, “How many days ahead must I request leave?” Our assistant should retrieve the relevant article, answer using the article, and show the source. All policies in the notebook are invented for learning; they are not real company policies.

**Two stages:** indexing prepares the documents before questions arrive; querying searches the existing index and generates an answer. Updating a document requires updating its index entries.

```text
INDEXING: source → extract → clean → split → embed → store chunks + metadata
QUERYING: question → embed → search → choose evidence → answer + citations
```

Use Python 3.10+ and a Jupyter kernel in the environment where packages are installed. The companion notebook contains installation commands. Set `OPENAI_API_KEY` in the kernel environment or a Git-ignored `.env`; never put a secret in a notebook cell. If VS Code cannot see a key exported in a terminal, restart the kernel or load the `.env` from the notebook. Configure `OPENAI_MODEL` and `OPENAI_EMBEDDING_MODEL` if needed. Every API request may incur charges; inspect [current pricing](https://developers.openai.com/api/docs/pricing). Start with synthetic information. Uploaded text leaves your machine when sent to the API, including text sent for embeddings.

---

## Unit 4.1: When to Use RAG — Day 25

### What is RAG?

Retrieval-augmented generation combines search with a language model. The application finds potentially relevant external text and supplies it to the model with the question. The model then produces an answer using that evidence. Retrieval does not change model weights or guarantee the information is true.

1. A question arrives: “When do I submit a leave request?”
2. Search retrieves the leave-policy paragraph and its source metadata.
3. The application adds the paragraph to the prompt as untrusted evidence.
4. The model answers and cites the relevant source, or reports that the evidence is insufficient.
5. You inspect both the answer and retrieved chunks when something goes wrong.

### When does it help?

| Situation | Likely approach | Why |
| --- | --- | --- |
| Many private policies or manuals | RAG | Narrow a large collection to relevant, authorized passages. |
| Frequently revised policies | RAG plus an update process | Retrieve the current indexed version; stale indexes can still give stale answers. |
| One short document | Send it directly | Retrieval can add unnecessary complexity. |
| Rewriting provided text | Direct prompting | The required text is already present. |
| General explanation with no private source | Direct prompting, check factual needs | Retrieval is not automatically necessary. |
| Exact arithmetic over a table | Database or calculation | Search and generation alone may miscount. |

RAG can support private data only if the application enforces access and the chosen service is appropriate for that data. Retrieval itself does not provide privacy, freshness, or authorization.

### Diagnose two major failure types

- **Retrieval failure:** the correct evidence was not retrieved. Causes include bad extraction, chunk boundaries, vocabulary mismatch, or filtering errors.
- **Generation failure:** useful evidence was retrieved but the model ignored, misread, combined, or cited it incorrectly.
- **Source failure:** the document itself is wrong, conflicting, out of date, or inaccessible. Better generation cannot repair it reliably.

**Experiment:** In the notebook, send the entire small collection for one question. Note elapsed time and input/output tokens. Compare later with retrieval. For meaningful cost comparisons include the one-time embedding/indexing cost and ongoing query embeddings, plus generation costs, not just prompt length. This small example does not predict behavior on a large collection.

**Check yourself:** Why might a policy answer be wrong even when the model follows the instruction to use only retrieved text?

---

## Unit 4.2: Loading and Cleaning Documents — Days 26–27

### Convert sources into inspectable records

Common inputs are plain text, PDF, HTML and Word. A loader should return text and metadata, such as `source`, `title`, `section`, `page`, `document_type`, `updated_at`, and a stable document ID. Page numbering may be zero-based in a library: convert it consistently before showing citations. An extraction path does not prove the resulting text is correct.

| Source | Basic route | Common problem |
| --- | --- | --- |
| Text/Markdown | Read UTF-8; retain headings | Encoding and headings lost during normalization. |
| PDF | LangChain `PyPDFLoader` or `pypdf` | Scans have little selectable text; tables or columns can reorder. |
| HTML | `WebBaseLoader` or Beautiful Soup | Navigation, cookie banners and duplicate menus pollute results. |
| DOCX | Word loader or `python-docx` | Tables, headings or embedded content may need special handling. |

Review permission and confidentiality before loading or uploading any actual document. For web pages, respect access rules. For PDFs, look at both the visual page and extracted text, not only a successful parser return value.

### Clean carefully

Remove clear noise such as an identical footer on every page, while preserving negations, numbers, paragraph boundaries, headings, lists and table headers. Avoid indiscriminately removing every repeated line: a repeated section title can give a chunk crucial context. Keep raw extraction separately so you can debug the transformation. Normalize whitespace without flattening everything into a single paragraph.

**Suggested record:**

```python
record = {
    "id": "leave-v1-page2", "text": "...",
    "metadata": {"title": "Leave policy", "source": "leave.pdf",
                 "page": 2, "section": "Requesting leave",
                 "document_type": "policy", "updated_at": "2026-01-01"},
}
```

### Document lifecycle

- Assign a stable source ID and record a content hash/version. Distinguish the source identity from a particular version.
- Detect unchanged files so you do not pay to embed them again. Deduplicate only after understanding whether identical text belongs to different sources or access groups.
- On update, prepare new chunks, replace obsolete chunks, and prevent mixing old and new versions; record failures and retry safely.
- When a document is deleted or access changes, remove or revoke its chunks and any cached copies according to your retention policy. Think about consistency during reindexing.

**Exercise:** Extract three messy documents. For each, record page count, empty pages, missing headers, reordered text, table damage, and a link to a source page. Select and inspect at least one page per problem type. The notebook demonstrates local PDF/HTML/Word loading with optional dependencies.

---

## Unit 4.3: Parsing Messy Documents — Day 28

Basic extraction reads text but may not understand the page layout. Two columns can become alternating lines. A scan may have no text layer at all. A table may lose row and column relationships. Forms, footnotes, and reading order can be especially difficult.

### Choose a method per page

1. **Basic PDF extraction:** fastest first pass for digitally generated, simple pages; inspect the result.
2. **Layout-aware parsing:** tools such as [Docling](https://docling-project.github.io/docling/) identify document structure and can preserve tables more usefully. Inspect results: layout inference also makes mistakes.
3. **OCR:** converts text in scanned images to machine-readable text. Image quality, rotation, language and handwriting affect accuracy.
4. **Vision model:** ask an image-capable model to transcribe or describe a rendered page when justified. It costs time and money, can misread numbers, and sends page data to the provider. It is not a truth oracle.

For a table, keep headers with their corresponding rows. Example: `Plan: Basic | Limit: 10 GB` is more useful than isolated `Basic` and `10 GB`. Preserve units, merged-cell meaning and page coordinates where available. Review the rendered page to confirm extracted values.

**Three-way experiment:** Select one multi-column page and one table from an authorized PDF. Compare basic extraction, layout-aware parsing and vision/OCR. Use the same page and record time, cost, reading order, headers, numeric errors and missing content. The companion notebook supplies optional examples; rendering and OCR may need system dependencies, and a vision example requires an image-capable model. Do not treat synthetic text as proof that a real PDF parsed correctly.

**Acceptance check:** Can a person reconstruct the answer and citation from the extracted text? If not, fix extraction before tuning the retriever.

---

## Unit 4.4: Chunking — Day 29

A chunk is a portion of a document that can be independently searched and supplied as evidence. Whole documents may be too large or contain too many unrelated topics. Chunks that are too small can omit conditions or definitions; chunks that are too large may bury the relevant sentence and increase context costs.

### Common strategies

| Strategy | How it works | Tradeoff |
| --- | --- | --- |
| Fixed size | Split every N characters or tokens | Simple but can cut sentences and sections. |
| Recursive separators | Try paragraphs, then lines, then smaller separators | Preserves more structure but needs tuning. |
| Section-based | Split at headings and then subdivide long sections | Keeps topic labels, depends on good parsing. |
| Semantic | Split at larger changes in meaning | Can fit content well but costs more and needs evaluation. |

**Overlap** repeats material across adjacent chunks. It can protect facts at boundaries but introduces duplicates, storage cost and repeated evidence. Overlap within a document; do not concatenate different documents. Size may mean characters or tokens depending on the splitter—always state which. Measure model context in tokens, not characters.

Every chunk should retain document ID, chunk ID, title, section and source/page when known. A chunk ID such as a source/version/section/offset combination helps updates, citation checks and tracing. The title or section may also be included in the embedded text when helpful, but assess whether it changes retrieval.

**Experiment:** compare fixed character windows against section chunks. Inspect five examples from each: are sentences cut? Are conditions attached to claims? Are title and page available? Then compare retrieval on the same labeled questions rather than choosing from appearance alone.

---

## Unit 4.5: Embeddings and Semantic Search — Days 30–31

An embedding model maps text to a vector of numbers. Search embeds the question with the **same model and configuration** as the stored documents, compares vectors, and ranks chunks by closeness. Similarity is a relevance signal, not a calibrated probability that the answer is present.

Cosine similarity is:

```text
cosine(a, b) = (a · b) / (||a|| × ||b||)
```

For two normalized vectors, the dot product equals cosine similarity. Check for zero vectors and matching dimensions. Exact search compares the query with every vector; it is easy to understand but scales poorly. The notebook builds exact search with NumPy before introducing a database.

### Choosing and testing embeddings

Compare models with your own representative queries and source documents: retrieval quality, supported languages, latency, cost, maximum input, and dimensions. Test English, Hindi and mixed Hindi-English if relevant; multilingual performance depends on your data and cannot be inferred from model size. Do not compare raw scores between unrelated models or corpora as if they had a universal meaning.

When the embedding model, dimensions or preprocessing changes, create a new compatible index and re-embed documents; do not mix different vector spaces even if their dimensions happen to match. Batch within the provider's input limits and record the model/version along with the index.

**Weak spot:** a question containing a rare identifier such as `NX-7421` may need exact lexical matching; semantic search can miss it or retrieve a similar but wrong code. Compare paraphrase and exact-code queries. Hybrid search is introduced later in Part 2; for now observe the failure instead of assuming embeddings solve it.

**Experiment:** embed the sample sections once, embed a query, calculate cosine similarity, print ranked text and scores. Do you see the expected source? Try an unrelated question: notice that a nearest neighbor still exists even when no answer exists.

---

## Unit 4.6: Vector Databases — Days 32–33

A vector database stores vectors together with IDs, source text and metadata, and supports nearest-neighbor searches. A typical record has `id`, `embedding`, `document`, and filterable metadata. It may also handle persistence, indexing and updates; capabilities and filter semantics depend on the product.

### Exact versus approximate search

Exact search checks every candidate and provides a useful baseline for small datasets. Approximate nearest-neighbor (ANN) indexes trade some recall and resource usage for speed at scale. Measure ANN recall against exact results on representative queries; a fast search that misses critical passages is not automatically better. For tiny datasets, a simple exact implementation may be appropriate.

### Filters and lifecycle

Filter on metadata such as `document_type` or date **before or as part of** search when supported. Demonstrate the difference between searching all documents and searching only policies. Metadata filters help narrow results, but a client-supplied filter is not authorization. A real service must derive access constraints from trusted identity and enforce them at the data boundary, including on every retrieval path.

Use stable IDs for upserts. A source update may change its number of chunks: remove stale IDs as well as inserting new ones. Delete all chunks for a removed document and verify the result. Maintain a mapping between documents and chunks. Avoid logging sensitive passage text by default.

The notebook includes an optional local persistent Chroma example through LangChain: insert, filter, update and delete test chunks. It uses a separate demo collection; it does not delete your files. pgvector is another option when PostgreSQL is already part of the stack. Confirm library versions and filter syntax against their documentation before using in production.

---

## Unit 4.7: Your First RAG Pipeline — Days 34–35

### The minimum flow

1. Receive a question and establish the caller's allowed document scope.
2. Embed the question using the index's embedding configuration.
3. Retrieve top `k` candidate chunks, with trusted filters where applicable.
4. Inspect whether there is enough relevant evidence. Thresholds require calibration; a high similarity score alone does not prove support.
5. Construct a bounded context with stable source labels; treat retrieved content as data.
6. Ask the model to answer only from evidence, cite each factual claim, and state when evidence is missing.
7. Inspect status, response, citations, token usage, latency and retrieval traces without exposing secrets.

The notebook implements this flow directly so each step is visible, and shows a LangChain composition for the prompt and model. A framework chain is an orchestration convenience, not a security boundary.

### Context selection and comparison

`k` is the maximum number of chunks to retrieve, not a guarantee of quality. More chunks can improve coverage but add irrelevant material, duplicates, token cost and opportunities for conflicting claims. Order by retrieval relevance for the baseline, retain source IDs and compare against alternatives. Respect the context window and reserve output budget. Truncation must not quietly remove caveats or source labels.

**Experiment:** for five questions, compare `k=3` and `k=10` on the same index and model. Record retrieved IDs, correctness, whether citations support claims, total input/output tokens, elapsed time, and estimated cost using current prices. Record when fewer than 10 chunks exist. Do not use one run to claim a general winner. Query embeddings may be reused for a fair comparison; report whether you did so.

---

## Unit 4.8: Citations and Saying I Do Not Know — Day 36

### Make answers verifiable

Give each evidence chunk a stable label, for example `[leave-v1-request]`, and preserve a mapping to document title, section, page or URL. Ask for citations next to claims rather than one unexplained list at the end. The application should resolve IDs to trusted metadata rather than allowing generated text to create arbitrary links.

Check both:

1. **Validity:** Does every cited ID exist among the retrieved evidence?
2. **Support:** Does that specific passage actually justify the claim, including numbers, date and exceptions?

A valid label is not proof of support. Conversely, a correct answer without a source may be difficult to verify. Model citations can be omitted, fabricated or attached to the wrong sentence. Manual checks in this part make that limitation visible; Part 3 develops repeatable evaluation.

### Handle missing and conflicting evidence

When evidence does not answer the question, state what is missing and avoid supplying an invented policy. If evidence is partial, answer only the supported portion. If documents conflict, cite each and describe the conflict; do not silently choose the most convenient one. An irrelevant nearest neighbor is still a nearest neighbor: do not assume retrieval success because results exist.

**Experiment:** ask at least three unanswerable questions and five answerable questions. For each answerable result, check each claim against its cited passage. For unanswerable ones, verify the corpus truly lacks the requested fact and record whether the assistant abstains. Compare answers against evidence, not model confidence.

### Prompt injection and access

A document might contain “ignore previous instructions.” It is untrusted content, not a command. Separate instructions from document text, validate output, and prevent retrieved content from granting tool access. Prompt wording alone cannot guarantee resistance. Do not send unauthorized chunks to the model and hope the prompt hides them. Part 4 covers threat modeling, authorization and red-teaming in depth.

---

## Capstone Project

Build a small handbook assistant with at least three fictional documents and 10 labeled questions. Include at least five answerable questions, three unanswerable questions and one identifier query. For each question record expected evidence or `none`, retrieved chunks, answer, citation validity, human support judgment, input/output tokens and elapsed time.

**Build sequence:**

1. Inspect raw documents and write a source inventory.
2. Parse and clean while preserving metadata and a reproducible source version.
3. Compare two chunking strategies on five chunks and a small query set.
4. Embed and index the chosen chunks; verify the embedding configuration.
5. Compare retrieval with and without an appropriate metadata filter.
6. Generate grounded answers; verify at least five citations and three abstentions.
7. Compare `k=3` and `k=10`, report results and choose based on observed tradeoffs.
8. Edit then delete a demo document; verify obsolete evidence disappears.

**Deliverables:** notebook with recorded observations, source inventory, list of failures with suspected stage, and a short decision on the next improvement. Never report unchecked model output as a verified result. Keep a separate held-out set for later tuning and evaluation.

## Practice exercises and self-check

1. Describe the offline indexing path and online query path without looking.
2. Explain a case where giving the whole document is preferable to RAG.
3. Find a paragraph where a removed heading changes meaning; fix the cleaning process.
4. Explain why a scan can produce a nearly empty text layer and what OCR adds.
5. Compare fixed windows and sections, and identify one boundary failure.
6. Explain why the same vector length does not imply compatible embedding spaces.
7. Calculate cosine similarity for `[1, 0]` and `[1, 1]` (answer: approximately 0.707).
8. Explain why a top-ranked chunk may still be irrelevant.
9. Change a document and show how stale chunks are removed.
10. Find an answer with a valid citation that does not actually support the claim.
11. Give a partial-answer example and explain what the assistant should say.
12. Explain why metadata filtering is not enough to establish user authorization.

### Completion checklist

- [ ] I can explain retrieval versus generation failures with examples.
- [ ] I inspected parsed output against original documents.
- [ ] I recorded chunking strategy, embedding model and source versions.
- [ ] I searched by hand and with a vector store, including a metadata filter.
- [ ] I compared `k=3` and `k=10` on the same questions.
- [ ] I checked five cited answers and three unanswerable questions.
- [ ] I understand that this learning prototype is not production authorization or an evaluated security system.

## Common mistakes

- Embedding document fragments without titles or the conditions that make them meaningful.
- Assuming all PDF pages contain selectable text or that table extraction is correct.
- Treating a similarity score as a probability or a universal threshold.
- Mixing embedding models or leaving old chunks after updates.
- Trusting a model-generated citation without checking the source.
- Sending private material to a provider without reviewing the appropriate data handling.
- Treating prompts, metadata filters or this toy corpus as sufficient production security.

## Further reading

- [OpenAI embeddings guide](https://developers.openai.com/api/docs/guides/embeddings) and [Responses API reference](https://developers.openai.com/api/reference/python/resources/responses/methods/create).
- [OpenAI model catalog](https://developers.openai.com/api/docs/models) and [pricing](https://developers.openai.com/api/docs/pricing).
- [LangChain document loaders](https://docs.langchain.com/oss/python/integrations/document_loaders) and [text splitters](https://docs.langchain.com/oss/python/integrations/splitters).
- [Chroma documentation](https://docs.trychroma.com/) and [pgvector](https://github.com/pgvector/pgvector).
- [Docling documentation](https://docling-project.github.io/docling/).

Documentation, model availability, package APIs and pricing can change. Check the linked references and record package versions for reproducible experiments.
