# AI Engineering Roadmap: Complete Topic Outline

**Source:** CampusX "100 Days of AI Engineering" (ai-engineer-roadmap-2026.vercel.app/100-days/), plus one added module from the full roadmap, Level 9 (AI System Design).

**Structure:** 7 modules → units → subtopics (with a short description of each) → a hands-on exercise ("Try it") and an expected outcome.

> Note: the descriptions below are condensed, paraphrased summaries written for training use, not copies of the site's text. Refer to the original site for full lessons.

## Table of Contents

| #   | Module                                     | Days      | Units         |
| --- | ------------------------------------------ | --------- | ------------- |
| 1   | LLM Fundamentals                           | 1–6       | 6             |
| 2   | Building with LLM APIs                     | 7–17      | 9             |
| 3   | Prompt Engineering                         | 18–24     | 8             |
| 4   | RAG (Build, Advanced, Evaluate, Secure)    | 25–56     | 26            |
| 5   | Agents (Build, Advanced, Evaluate, Secure) | 57–83     | 22            |
| 6   | Shipping to Production                     | 84–100    | 15            |
| 7   | **AI System Design (added, from Level 9)** | extension | 10 + capstone |

---

# MODULE 1: LLM Fundamentals (Days 1–6)

**Prerequisites:** basic Python, calling an API, an API key for one model provider.
**By the end:** explain why LLMs give fluent wrong answers, estimate cost and latency of a request, and choose a model from measurements rather than rankings.

## Unit 1.1: What an LLM Does (Day 1)

- **Next-token prediction:** the model predicts one token, appends it, and repeats.
- **What an LLM is not:** not a database, search engine or fixed program; it doesn't look up stored answers.
- **The path through the model:** text → tokens → numbers → layers → likelihood of each next token.
- **What parameter counts mean:** 8B/70B describe learned size; bigger isn't automatically better for your task.
- **Why chat models behave as they do:** instruction and human-feedback training explains following instructions, refusing and hedging.
- **Try it:** send the same prompt five times and note how answers vary.
- **Outcome:** explain in two sentences how an LLM answers and why it can sound confident yet be wrong.

## Unit 1.2: Tokens, Context and Latency (Day 2)

- **Tokens, not words:** models read and bill in tokens; different models split text differently.
- **Why some text costs more:** Hindi, mixed Hindi-English, code and numbers often use more tokens than plain English.
- **Input vs output tokens:** both are billed; output usually costs more and is generated one token at a time.
- **Latency you can feel:** time-to-first-token and generation speed decide how fast it feels.
- **What fills the context window:** instructions, history, documents, examples, tool results and the answer all share one limit.
- **Context is not knowledge or memory:** the window is temporary; training knowledge is fixed; long-term memory must be built.
- **Bigger windows aren't better windows:** facts in the middle of long inputs get missed and extra text dilutes focus.
- **Try it:** count tokens for English, Hindi, code and numbers, then estimate the cost of 10,000 requests.
- **Outcome:** estimate cost and speed, and decide between more context or better-chosen context.

## Unit 1.3: Beyond Text: Images, Audio and PDFs (Day 3)

- **Models that see and hear:** many accept images, audio and documents and reply in text.
- **Images cost tokens too:** cost rises with resolution; crop or resize.
- **PDFs come in two kinds:** digital PDFs have text; scanned PDFs are images of pages.
- **Audio:** transcribe first, or send audio directly to keep tone and emphasis.
- **What models read well:** text in images, a chart's main message, named form fields.
- **Where they slip:** counting objects, tiny text, exact table values, invented details.
- **Image or extracted text?:** clean text is usually cheaper and more accurate; use images when layout, charts or handwriting matter.
- **Try it:** send one chart as an image and as extracted text; compare answers and tokens.
- **Outcome:** choose the right input type and predict cost and failure points.

## Unit 1.4: Controlling Output (Day 4)

- **Temperature:** lower = consistent, higher = varied.
- **Length limits and stop points:** cap answer length and endpoints to control cost and format.
- **Match randomness to the task:** low for extraction, classification and facts; higher for brainstorming and creative work.
- **Reasoning models:** think longer first; help on hard math, code, planning and multi-step problems.
- **When reasoning is wasted:** extraction, rewriting, simple classification and summaries are better served by a fast model.
- **The trade-off:** more reasoning can raise quality but costs more and is slower; some settings (e.g., temperature) may be ignored.
- **Try it:** compare low/high temperature on two tasks, and a fast vs reasoning model on three hard problems.
- **Outcome:** choose settings and model type by task, not guesswork.

## Unit 1.5: Where LLMs Fail (Day 5)

- **Common strengths:** summarizing, rewriting, extracting, classifying, coding, translating, handling messy text.
- **Common failure modes:** making things up, arithmetic slips, shaky reasoning, ignoring instructions, wording sensitivity, outdated knowledge, inconsistent answers.
- **Why models make things up:** they produce likely continuations, not verified facts.
- **Ask "how reliably?", not "can it?":** measure success rate on your task.
- **Try it:** write 10 questions (known, unknowable, false-premise, missing-info) and record answers and apparent confidence.
- **Outcome:** predict which tasks need safeguards (right documents, output checks, human review).

## Unit 1.6: Choosing a Model (Day 6)

- **Closed vs open models:** API models give top capability with less control; open models give control and privacy with more work.
- **No single best model:** depends on task, budget, speed and data rules.
- **Knowledge cutoffs:** anything newer must be supplied via documents or tools.
- **Pin your version:** model names can change or retire; choose a specific version and re-test.
- **Leaderboards narrow the field; your examples decide:** shortlist by rankings, then test on your own questions.
- **Five things to compare:** quality on your task, speed, cost per request, input capacity, privacy/feature needs.
- **How to choose:** define task → shortlist → test on own examples → compare quality/speed/cost → study failures → decide.
- **Outcome:** choose a model using results from your own examples.

---

# MODULE 2: Building with LLM APIs (Days 7–17)

**Prerequisite:** Module 1. **Framework used:** LangChain.
**By the end:** build an assistant that holds a conversation, streams, returns trustworthy structured data, uses tools safely, recovers from failures and reports its cost.

## Unit 2.1: LangChain Basics and Conversations (Day 7)

- **What LangChain is for:** one consistent way to call models, build prompts, use tools and chain steps across providers.
- **Chat models:** connect to your chosen model; switching providers is a one-line change.
- **Messages and roles:** a conversation is a list of system, user and assistant messages.
- **The model has no memory:** each call starts fresh, so your app sends history every time.
- **Prompt templates:** write once with placeholders instead of building strings by hand.
- **Keep your keys secret:** store keys outside code, never in browser-side code.
- **Read the whole response:** check token usage and stop reason, not just the text.
- **Try it:** a three-turn conversation with a template; drop history on turn three and see what's forgotten.
- **Outcome:** call any model via LangChain and explain why it "remembers" only what you send.

## Unit 2.2: Open and Local Models (Day 8)

- **Open models:** weights are published; check licenses before building.
- **Run one on your own machine:** tools like Ollama do this in one command.
- **What your computer can handle:** memory limits model size; compressed versions trade quality for fit.
- **Same code, different model:** swap in a local model by changing the chat model.
- **Hosted open models:** providers host them and charge per token.
- **The trade-offs:** free and private, but slower and less capable than top API models.
- **A practical habit:** use cheap/local models while building, your chosen model for real use.
- **Try it:** run a small open model on 10 project questions next to your Module 1 model.
- **Outcome:** run, swap and judge open models.

## Unit 2.3: Streaming (Day 9)

- **Show answers as they arrive:** piece-by-piece output removes the wait.
- **Streaming in LangChain:** existing models, templates and chains stream without extra setup.
- **Perceived speed:** time-to-first-token matters more than total time.
- **Put the pieces back together:** reassemble the stream to save, check or log the full answer.
- **When streams break:** handle cancellations, dropped connections and half-finished answers.
- **Try it:** time the first word with and without streaming.
- **Outcome:** stream answers and still end up with a complete saved response.

## Unit 2.4: Structured Outputs (Days 10–11)

- **Don't just ask for JSON:** polite requests sometimes fail; use structured-output support.
- **Describe the shape you want:** schema with fields, required values, allowed values, nested lists.
- **Check before you use it:** validate every structured answer.
- **Valid isn't the same as correct:** well-formed output can still be wrong.
- **Generate → check → decide:** on failure, retry, ask for a fix, or reject with a clear error.
- **Try it:** extract name, date, amount and category from 10 messy receipts; count valid vs valid-but-wrong.
- **Outcome:** reliable structured data from a model, with failure handling.

## Unit 2.5: Tool Calling (Days 12–13)

- **The model asks; your code acts:** the model picks a tool and inputs; your app runs it.
- **Defining tools in LangChain:** function + name + "when to use" description + typed inputs.
- **The full loop:** user message → tool request → your code runs it → result back → final answer.
- **Matching results to requests:** each request has an ID; return results with the correct ID.
- **More than one call:** several calls per turn, repeated calls, and dependent calls.
- **When tools fail:** handle missing/invalid inputs and failures; send a useful error back.
- **Designing good tools:** precise names, few inputs, clear descriptions; avoid overlap.
- **Never trust tool inputs:** validate, especially before changing or deleting data.
- **Tool descriptions are instructions:** write them as carefully as prompts.
- **Try it:** weather + currency tools; ask a question needing both, then one needing neither.
- **Outcome:** build the complete tool-calling loop safely.

## Unit 2.6: Chaining Steps Together (Day 14)

- **Beyond one call:** real features combine model calls, tools and ordinary code.
- **Chains in LangChain:** prompt + model + output step as one reusable, streamable unit.
- **Three common patterns:** fixed sequence, routing, parallel-then-combine.
- **When to split a task:** small focused steps are more reliable and debuggable.
- **Chains vs agents:** in a chain your code decides the steps; in an agent the model does.
- **Try it:** classify a question by type, then route it to a type-specific prompt.
- **Outcome:** design features as chains and know when that beats one prompt.

## Unit 2.7: Handling Failures (Day 15)

- **What goes wrong:** wrong keys, bad requests, rate limits, server errors, timeouts, oversized inputs, malformed answers.
- **Retry or give up:** retry temporary errors; fail fast on permanent ones.
- **Retries in LangChain:** capped automatic retries and a time limit on every call.
- **Backup models:** fallbacks may answer differently.
- **Don't overload the API:** limit concurrent requests.
- **Fail gracefully:** show a clear message, not an error.
- **Try it:** use a wrong model name, then a tiny timeout; confirm the right retry/fallback fires.
- **Outcome:** an app that survives everyday API failures without crashing or retrying forever.

## Unit 2.8: Processing at Scale (Day 16)

- **One at a time is slow:** send independent requests concurrently.
- **Stay within your limits:** cap concurrency; LangChain batch lets you set it.
- **Batch APIs:** cheaper option for non-urgent jobs.
- **Don't lose progress:** retry failed items individually and save results as you go.
- **Estimate before you run:** compute tokens and cost of big jobs first.
- **Try it:** summarize 200 texts sequentially vs concurrently; compare time, errors, cost.
- **Outcome:** process large volumes quickly and cheaply without hitting limits or losing work.

## Unit 2.9: Cost, Logging and Tracing (Day 17)

- **Record every request:** model, tokens in/out, time, retries, errors, tools used, estimated cost.
- **Think in traces:** one user question can trigger many calls; tracing groups them.
- **Tracing LangChain apps:** turn on tracing (e.g., LangSmith) or use callbacks.
- **Find the problem step:** see which step failed, used most tokens or took longest.
- **Try it:** run 10 questions with tracing and report total cost, average time and the costliest question.
- **Outcome:** know exactly what your app costs and where it's slow or failing.

---

# MODULE 3: Prompt Engineering (Days 18–24)

**Prerequisite:** Module 2.
**By the end:** write clear prompts and system prompts, use examples and reasoning only where they help, test prompt versions, version them cheaply, and explain why wording alone can't keep an app safe.

## Unit 3.1: Anatomy of a Prompt (Day 18)

- **The parts of a prompt:** instructions, context, input, constraints, optional examples, desired answer format.
- **Roles and priority:** system instructions set global rules; user messages ask within them.
- **Separate instructions from data:** clearly mark where instructions end and documents/user text begin.
- **Where long text goes:** long documents first, labeled, then your question and instructions.
- **Templates keep prompts consistent:** use LangChain prompt templates.
- **Try it:** rewrite a one-line prompt with all six parts and compare answers on three questions.
- **Outcome:** write a well-structured prompt and explain each part.

## Unit 3.2: Zero-Shot and Few-Shot Prompting (Day 19)

- **When instructions are enough:** clear instructions without examples are cheaper and often fine.
- **When examples help:** when format, tone or judgment is hard to describe.
- **Designing good examples:** show exact behavior/format, including a tricky case.
- **Quality, variety and order matter:** poor or repetitive examples get copied; test ordering effects.
- **Few-shot templates:** keep examples separate from instructions and easy to swap.
- **Try it:** classify 10 support messages by urgency with 0 vs 3 examples.
- **Outcome:** decide if a task needs examples and write ones that actually help.

## Unit 3.3: System Prompts and Output Control (Day 20)

- **What belongs in a system prompt:** purpose, behavior, prohibitions, what good answers look like.
- **Stable rules vs per-request instructions:** constant rules in the system prompt; task details in the user message.
- **Plan for the gaps:** specify behavior for missing info, ambiguity and out-of-scope requests.
- **Light guardrails, not a rulebook:** a few clear rules beat a long list.
- **Controlling style, length and format:** state tone, length, layout; show an example if needed.
- **When prompting isn't enough:** use structured outputs when code depends on format.
- **Try it:** ask three unanswerable questions, add one rule for that case, ask again.
- **Outcome:** a system prompt that keeps the assistant on task and handles gaps sensibly.

## Unit 3.4: Reasoning and Breaking Tasks Down (Day 21)

- **When to ask for reasoning:** multi-step logic, planning, math, code.
- **When it's a waste:** extraction, classification, rewriting.
- **Break big tasks into steps:** inside one prompt or as a chain.
- **Prompting reasoning models:** skip "think step by step"; give goal, context, constraints and what good looks like.
- **Try it:** compare a normal prompt vs a reasoning prompt on a multi-step question (accuracy, tokens, time).
- **Outcome:** choose when to request reasoning and how to decompose hard tasks.

## Unit 3.5: Testing and Improving Prompts (Day 22)

- **Why prompts fail:** vague instructions, contradictory rules, poor examples, missing context, unclear format.
- **Build a small test set:** normal questions, edge cases, past failures, each with expected answers.
- **Score answers simply:** code checks for exact answers, simple rule checks, model grader only where judgment is needed.
- **Compare on the same questions:** judge on the whole set, never one impressive answer.
- **Check consistency:** run each version multiple times.
- **Try it:** score two system-prompt versions on 10 questions, each run twice.
- **Outcome:** pick the better prompt with numbers.

## Unit 3.6: Automatic Prompt Improvement (Day 23)

- **Let a model improve your prompt:** give it the prompt, failures and what good looks like.
- **An improvement loop:** generate versions → score → keep best → repeat.
- **Choosing examples automatically:** let test results pick the best few-shot examples.
- **Optimization tools:** e.g., DSPy searches for instructions/examples that score well.
- **Don't overfit:** hold out questions the optimizer never sees.
- **Read what you get:** understand the winning prompt so it stays maintainable.
- **Try it:** get three improved versions; score them on your test set and 5 held-back questions.
- **Outcome:** improve prompts with a model or tool and confirm on unseen questions.

## Unit 3.7: Versions, Cost and Caching (Day 24)

- **Version prompts like code:** save each version with the model it ran on.
- **Compare on more than quality:** track tokens, cost and response time.
- **Trim what isn't helping:** remove repeated instructions and unneeded context.
- **Prompt caching:** put fixed text first and changing text last so providers can discount repeated prefixes.
- **Try it:** reorder the system prompt (fixed first) and compare cost/latency over 10 requests.
- **Outcome:** a versioned prompt history and cheaper prompts without quality loss.

## Unit 3.8: Prompt Injection Basics (Day 24)

- **What prompt injection is:** user or document text that tries to override your instructions.
- **Treat outside text as untrusted:** documents, web pages and tool results can all carry hijack attempts.
- **Spot override attempts:** requests to reveal the system prompt, change role or take forbidden actions.
- **Wording isn't security:** "never do X" can be talked around; protect sensitive actions in code.
- **More later:** deeper defenses appear in RAG and Agents modules.
- **Try it:** plant an "ignore previous instructions" line in a document and ask a normal question.
- **Outcome:** recognize injection and explain why code, not prompts, must protect what matters.

---

# MODULE 4: RAG (Days 25–56)

**Prerequisite:** Module 3. 26 units in 4 parts.
**By the end:** build a RAG system that finds the right passages, cites sources and admits when it doesn't know; improve it with advanced patterns; prove with test sets which changes helped; and secure it against hidden instructions, unauthorized access and data leaks.

## Part 1: Build a Simple RAG (Days 25–36)

### Unit 4.1: When to Use RAG (Day 25)

- **The RAG pipeline:** question → find relevant passages → add to prompt → model answers from them.
- **What RAG solves:** private documents, fast-changing information, collections too large to send whole, answers needing sources.
- **When not to use RAG:** the model already knows enough, the task is pure rewriting, or everything fits in one prompt.
- **Two kinds of failure:** wrong passages retrieved, or right passages misused by the model.
- **Try it:** send a long document whole with a question; note tokens, cost, time.
- **Outcome:** decide if a problem truly needs RAG and explain each pipeline step.

### Unit 4.2: Loading and Cleaning Documents (Days 26–27)

- **Getting text out:** load PDFs, web pages, Word files with LangChain loaders; spot scans needing OCR.
- **Clean without losing structure:** drop repeated headers/footers, keep headings, lists, tables.
- **Keep the details:** store title, source, section, date with the text for filters and citations.
- **Changes over time:** handle duplicates, updated versions, deleted files.
- **Try it:** load your three messiest documents and note where extraction broke.
- **Outcome:** clean text with useful metadata, ready for search.

### Unit 4.3: Parsing Messy Documents (Day 28)

- **Why basic loaders fail:** multi-column pages, tables, forms and scans come out scrambled or empty.
- **Parsers that understand layout:** tools like Unstructured or Docling detect headings, columns and tables.
- **Scanned pages:** OCR, or a vision model reading the page image.
- **Tables:** extract as tables, keeping column headers with each row.
- **Check the result:** compare extraction with the original on sample pages; choose a parser per document type.
- **Cost and speed:** vision-based parsing is slower and pricier; use only where needed.
- **Try it:** parse your messiest PDF three ways (basic, layout-aware, vision) and compare a table and a multi-column page.
- **Outcome:** trustworthy text from messy real-world documents.

### Unit 4.4: Chunking (Day 29)

- **Why split documents:** whole documents are too big to search precisely or send.
- **Ways to split:** fixed size, paragraphs/sentences, document sections, meaning changes.
- **Size and overlap:** small chunks are precise but lack context; large ones add noise; slight overlap avoids cutting ideas.
- **Keep the labels:** each chunk carries its document title and section.
- **Try it:** split by fixed size vs sections; judge five chunks from each.
- **Outcome:** choose a chunking approach that fits your documents and questions.

### Unit 4.5: Embeddings and Semantic Search (Days 30–31)

- **What an embedding is:** a list of numbers capturing meaning; similar content sits close together.
- **Searching by meaning:** embed the question and find the nearest passages.
- **Build it once by hand:** search a small set without a database to see what a vector DB does.
- **Choosing an embedding model:** compare quality, cost, speed, language support (e.g., Hindi / mixed Hindi-English).
- **Don't mix models:** embeddings from different models aren't comparable; switching means re-embedding.
- **Where meaning-based search fails:** exact names, product codes, numbers, rare terms.
- **Try it:** search a paraphrased question vs one with an exact product code.
- **Outcome:** explain semantic search, pick an embedding model, predict weak spots.

### Unit 4.6: Vector Databases (Days 32–33)

- **Storing passages:** save text + embedding + metadata in a vector DB (e.g., pgvector) via LangChain.
- **Search with filters:** limit results by document type, date, etc.
- **Why databases use shortcuts:** approximate search trades a little accuracy for speed at scale.
- **Keeping it current:** add, update, delete passages without rebuilding everything.
- **Try it:** load all passages; run one question with and without a document-type filter.
- **Outcome:** store your collection and search it fast with filters.

### Unit 4.7: Your First RAG Pipeline (Days 34–35)

- **Put the pieces together:** search + prompt + model as one LangChain chain.
- **Prompt for grounded answers:** answer only from the provided passages.
- **When search comes back weak:** handle irrelevant or thin results instead of letting the model guess.
- **How many passages, in what order:** more isn't always better; most relevant first; watch total length.
- **Try it:** five questions with 3 vs 10 passages; compare answers, tokens, cost.
- **Outcome:** a working RAG pipeline answering from your documents.

### Unit 4.8: Citations and Saying "I Don't Know" (Day 36)

- **Answer from the evidence:** base answers on retrieved passages, not model memory.
- **Cite your sources:** link claims to document and section.
- **A citation isn't proof:** verify the cited passage supports the claim.
- **Say when the answer isn't there:** admit gaps instead of inventing.
- **Keep evidence and explanation apart:** distinguish document content from model additions.
- **Try it:** ask three unanswerable questions; verify five cited answers.
- **Outcome:** verifiable answers and an assistant that admits ignorance.

## Part 2: Advanced RAG (Days 37–44)

_Each unit tackles a specific way simple RAG fails._

### Unit 4.9: Keyword and Hybrid Search (Day 37)

- **Keyword search still matters:** BM25 finds exact names, codes, phrases.
- **Combine both:** hybrid search merges semantic and keyword results.
- **Merging rankings fairly:** reciprocal rank fusion combines by position, not raw score.
- **When hybrid helps most:** names, technical terms, IDs, numbers, exact wording.
- **Try it:** run five exact-name/code questions through semantic, keyword and hybrid search.
- **Outcome:** add hybrid search and know which questions it helps.

### Unit 4.10: Smarter Chunks and Filters (Day 38)

- **Search small, send big:** search precise chunks, send the surrounding section.
- **Give chunks their context:** prepend a short line naming the document and section.
- **Tables and structured content:** keep tables whole and describe them in words.
- **Filters from the question:** have the model extract filters (product, date range, type) before searching.
- **Try it:** compare plain search vs question-derived filters on three specific questions.
- **Outcome:** passages and filters that help search find, and the model see, the right content.

### Unit 4.11: Reranking (Day 39)

- **Search wide, then choose carefully:** retrieve many candidates, rerank to the best few.
- **Why it works:** a reranker reads question and passage together, so it judges relevance better.
- **The cost:** adds time and money per question.
- **When it doesn't help:** if search is already precise.
- **Try it:** compare top-3 passages with and without reranking on five poorly answered questions.
- **Outcome:** add reranking and explain its gains and costs.

### Unit 4.12: Query Transformation (Day 40)

- **Questions aren't always good search queries:** users write vague, long, multi-part questions.
- **Rewrite the question:** model turns it into a clearer search query.
- **Multiple queries:** generate several versions, search each, combine.
- **RAG-Fusion:** merge multi-query results with reciprocal rank fusion.
- **Search with a draft answer (HyDE):** write a hypothetical answer and search for similar passages.
- **Break it down:** split multi-part questions into sub-questions, answer separately, combine.
- **Weigh the cost:** each technique adds calls and latency; use selectively.
- **Try it:** compare plain search, multiple queries and decomposition on ambiguous and multi-part questions.
- **Outcome:** turn hard questions into good searches and pick the right technique.

### Unit 4.13: Routing and Adaptive RAG (Day 41)

- **Not every question needs retrieval:** greetings, follow-ups and general questions can skip it.
- **Route to the right source:** choose the right collection or tool before searching.
- **Adaptive RAG:** pick a strategy per question: direct answer, simple search, or heavier techniques.
- **Cost and speed:** save on easy questions, spend where it helps.
- **Try it:** build a router for 20 questions (direct / simple search / transform and search).
- **Outcome:** RAG that adapts effort to each question.

### Unit 4.14: Self-Correcting RAG (Day 42)

- **Check what you found (Corrective RAG):** grade retrieved passages; if weak, rewrite and retry or use another source.
- **Check what you wrote (Self-RAG):** verify the answer is supported and addresses the question.
- **Search in steps (multi-hop):** search, read, search again for missing facts.
- **Know when to stop:** cap retries.
- **A step toward agents:** the system decides its own next step.
- **Try it:** add a grading + retry step and test on your five worst questions.
- **Outcome:** RAG that notices bad retrieval or unsupported answers and corrects itself within limits.

### Unit 4.15: RAG over Structured Data (Day 43)

- **Not all knowledge is in documents:** spreadsheets and databases need counting, summing, filtering that vector search does poorly.
- **Text-to-SQL:** model writes the query, your code runs it, model explains the result.
- **Give it the map:** table structure, column descriptions, example values, sample question/query pairs.
- **Keep it safe:** read-only access, allowed tables only, query checks, row limits.
- **When the query fails:** detect errors/odd results and let the model fix within a retry limit.
- **Show your working:** show the query or figures behind the answer.
- **Documents or database?:** route with the Unit 4.13 router.
- **Try it:** load a spreadsheet into a small DB; answer five count/total/filter questions via text-to-SQL vs document search.
- **Outcome:** accurate, safe answers from tables, routed to the right source.

### Unit 4.16: Graph RAG and Choosing a Pattern (Day 44)

- **What Graph RAG is:** extract entities and relationships into a graph and search through connections.
- **When it helps:** relationship questions and collection-wide questions (e.g., main themes).
- **What it costs:** many model calls to build, harder to maintain.
- **Choosing a pattern (start from your failures):**
  - Missed exact terms → hybrid search
  - Right passages ranked low → reranking
  - Vague questions → query transformation
  - Weak retrieval → corrective RAG
  - Numbers in tables → text-to-SQL
  - Relationship questions → Graph RAG
- **More isn't better:** each pattern adds cost, delay and failure points.
- **Try it:** map each wrong answer from simple RAG to the pattern most likely to fix it.
- **Outcome:** judge Graph RAG's worth and choose patterns from real failures.

## Part 3: Evaluating RAG (Days 45–51)

### Unit 4.17: Why Evals, and Reading Your Failures (Day 45)

- **Why "it looks fine" isn't enough:** teams that skip testing ship embarrassing failures (e.g., a chatbot inventing a refund policy).
- **AI testing is different:** outputs vary and answers are rarely simply right/wrong; use example sets and scores.
- **Read before you measure:** review 20–30 real answers, note issues, group them.
- **Measure what actually fails:** let the groups drive what to measure and fix.
- **The evaluation loop:** define good → build examples → run → score → inspect failures → improve → repeat.
- **Try it:** run 20 questions, note each flaw, group into 3–5 failure types.
- **Outcome:** explain why systematic testing is needed and find the most common failures.

### Unit 4.18: Test Sets and Scoring (Days 46–47)

- **A golden test set:** questions with correct answers and source passages, from your own documents.
- **Growing it quickly:** hand-write, then use a model to suggest more; verify each one.
- **Three ways to score:** code checks, a model judge for meaning, your own review for subtle cases.
- **With or without an answer key:** some checks compare to a reference; others (e.g., "supported by passages?") don't need one.
- **Writing a good judge:** one clear question at a time, pass/fail rather than 1–10.
- **Check the judge:** score 20 answers yourself, compare, and refine until you mostly agree.
- **Try it:** build a "supported by passages?" judge and measure agreement on 20 answers.
- **Outcome:** a golden test set and verified scorers.

### Unit 4.19: Measuring Retrieval (Day 48)

- **Test the search on its own:** no prompt can fix missing passages.
- **Recall@k:** how often the right passage is in the top k.
- **Precision:** how many retrieved passages are actually useful.
- **Compare choices fairly:** test chunk sizes, embedding models, search types, reranking and query techniques on the same set.
- **Try it:** measure Recall@5 for semantic, hybrid and hybrid + reranking.
- **Outcome:** know whether search finds what users need and which Part 2 techniques helped.

### Unit 4.20: Evaluating RAG Answers (Days 49–50)

- **Test each part, then the whole:** search, model, full pipeline, then user experience.
- **Test the model on its own:** give it correct passages; if it fails, search isn't the problem.
- **The three core RAG checks:** relevant passages found? answer supported? answer addresses the question?
- **Find where it broke:** search, assembled prompt, or model answer.
- **What users care about:** correct, complete answers with working citations, right tone and format.
- **Cost and speed per question:** track alongside quality, especially for self-correction.
- **Evaluation tools:** libraries such as DeepEval or RAGAS; understand what each score means.
- **Try it:** score the test set on the three core checks; diagnose your five worst answers.
- **Outcome:** evaluate RAG part by part and pinpoint failure sources.

### Unit 4.21: Automatic Regression Tests (Day 51)

- **Save everything together:** questions, prompts, settings, model versions and results for reproducibility.
- **Compare against a baseline:** judge every change against the current best.
- **Test on every change:** auto-run evals on prompt/setting/model changes; stop if quality drops.
- **Grow the test set:** add every new failure.
- **Watch quality, cost and speed together:** trade-offs need deliberate decisions.
- **Try it:** one command that fails if score drops below baseline; verify with a deliberately worse prompt.
- **Outcome:** impossible to degrade the app without noticing.

## Part 4: Securing a RAG Application (Days 52–56)

### Unit 4.22: Threats to a RAG App (Day 52)

- **Where untrusted text gets in:** user questions, documents, uploads, web pages, anything the app didn't write.
- **Documents are data, never instructions:** retrieved text can contain commands; never obey it.
- **Model output is untrusted too:** it can carry injected content into links, tool calls, other systems.
- **A standard checklist:** OWASP Top 10 for LLM Applications.
- **Try it:** sketch data flow; mark where untrusted text enters and where harm could occur.
- **Outcome:** map attack surfaces before defending.

### Unit 4.23: Hidden Instructions and Poisoned Documents (Day 53)

- **Direct and indirect injection:** typed by a user, or hidden in retrieved content.
- **Poisoned documents:** anyone who can add/edit documents can plant content to steer answers.
- **Leaking through answers:** injected instructions can push private info into links or images.
- **Defenses that work in code:** separate documents from instructions, limit capabilities, check outputs, control who adds documents.
- **Why "ignore malicious instructions" fails:** prompts help slightly but can be bypassed.
- **Try it:** plant three documents (alter answers, misuse a tool, embed a leaking link) and see which succeed.
- **Outcome:** defend against injection with code-level controls.

### Unit 4.24: Who Can See What (Day 54)

- **Not every user should see every document:** store access rights with passages.
- **Filter during search, not after:** restricted passages must never reach the model.
- **Keep users and customers apart:** prevent cross-tenant retrieval.
- **Other places data leaks:** cached answers, conversation memory, citations showing restricted titles.
- **Try it:** two users with different access; ask the same restricted question as each and check answers and citations.
- **Outcome:** each user sees only permitted information.

### Unit 4.25: Personal Data and Secrets (Day 55)

- **Find it before you index it:** scan for phone numbers, emails, IDs, passwords, keys.
- **Remove or mask what you don't need:** before indexing.
- **Keep it out of everything else:** logs, traces, caches, test sets.
- **Deleting means everywhere:** index, caches, logs and test sets.
- **Know your provider's terms:** data retention and usage; basics of India's DPDP law for Indian users.
- **Try it:** scan your collection and recent logs for sensitive data.
- **Outcome:** keep personal data and secrets out of the system and remove them completely when required.

### Unit 4.26: Red-Teaming Your RAG (Day 56)

- **Attack your own app:** injection, poisoned documents, permission bypass, data extraction, rule evasion.
- **Input and output checks:** one extra layer, never the only defense.
- **Measure how often attacks work:** track success rates and leaks before and after fixes.
- **Every successful attack becomes a test:** add to the regression suite.
- **Security vs quality:** re-check quality after each defense.
- **Try it:** write 15 attacks, record successes, fix, re-run.
- **Outcome:** test defenses like an attacker and keep permanent tests.

---

# MODULE 5: Agents (Days 57–83)

**Prerequisite:** Module 4. 22 units in 4 parts. **Framework used:** LangGraph (plus MCP).
**By the end:** decide when a problem needs an agent; build one that plans, uses tools and MCP servers, remembers, manages context and cooperates with other agents; measure its behavior; and restrict what it can do.

## Part 1: Build an Agent (Days 57–63)

### Unit 5.1: What Is an Agent? (Day 57)

- **The agent loop:** goal → decide → act → observe → repeat until done.
- **Who decides what happens next:** in a chain your code decides; in an agent the model does.
- **Where agents help:** open-ended, multi-step tasks you can't map in advance.
- **When a chain is better:** if you can write the steps down, a chain is simpler, cheaper, more reliable.
- **What an agent is made of:** model, instructions, tools, memory, loop, stopping rules.
- **Try it:** list five user tasks and mark which need an agent vs a chain.
- **Outcome:** decide if an agent is justified and name its parts.

### Unit 5.2: The Agent Loop (Day 58)

- **Reason, then act (ReAct):** alternate deciding and doing, using each result to choose the next step.
- **Build the loop yourself once:** a simple loop that calls tools until done.
- **When to stop:** max steps, finish condition, repetition detection.
- **Step by step vs plan first:** incremental decisions vs writing a plan then following it.
- **Try it:** a three-tool-call task, then an impossible task to confirm the step limit.
- **Outcome:** explain and build a loop that always stops.

### Unit 5.3: Building Agents with LangGraph (Days 59–60)

- **Why LangGraph:** explicit step control, saved progress, pause for human approval.
- **State, nodes and edges:** state holds knowledge; nodes do work; edges choose what's next.
- **Branches and loops:** conditional edges and repeating tool-use loops.
- **Ready-made vs custom:** start with the prebuilt tool-using agent, go custom when needed.
- **See what happened:** inspect state after a run to understand the path.
- **Try it:** rebuild the 5.2 loop as a graph and add a branch for simple questions.
- **Outcome:** a controllable, debuggable LangGraph agent.

### Unit 5.4: Tools for Agents (Day 61)

- **Your RAG pipeline becomes a tool:** wrap Module 4 RAG as a search tool.
- **Tools the agent can choose between:** names, descriptions, inputs matter more with many tools.
- **Keep tool results small:** return only what's needed.
- **When tools fail:** useful errors so the agent can retry, switch tools or give up gracefully.
- **Looking vs changing:** separate read-only tools from state-changing ones; treat the latter carefully.
- **Try it:** give the agent a RAG tool and an action tool (e.g., create a support request).
- **Outcome:** a well-designed tool set including your RAG system.

### Unit 5.5: Planning and Reflection (Day 62)

- **Plan before acting:** a short plan for longer tasks.
- **Change the plan:** revise on failure or new information.
- **Check its own work:** review results and retry if they fall short.
- **The cost of thinking:** more planning/checking helps complex tasks but adds time and cost.
- **Try it:** multi-part task with and without planning; compare success, steps, cost.
- **Outcome:** use planning and self-checking only where useful.

### Unit 5.6: Saving Progress and Asking for Approval (Day 63)

- **Save as you go:** LangGraph persists state after each step.
- **Resume, don't restart:** continue interrupted runs.
- **Pause for a human:** show intended action and continue only after approval.
- **Safe to retry:** actions should be idempotent (no duplicate support requests).
- **Try it:** approval step before a state-changing tool; approve one, reject another.
- **Outcome:** agents that persist, resume and wait for approval.

## Part 2: Advanced Agents (Days 64–76)

### Unit 5.7: MCP Fundamentals (Day 64)

- **Why MCP exists:** a standard way to connect AI apps to tools and data; one integration works across apps.
- **Host, client and server:** the app (host) uses a client to talk to servers exposing a system's capabilities.
- **What a server can offer:** tools, resources (readable data), prompts.
- **Tool calling vs MCP:** tool calling is how a model asks for an action; MCP is how tools/data are packaged and shared.
- **Using existing servers:** file systems, code repositories, databases.
- **Connecting your agent:** LangChain MCP support loads server tools into a LangGraph agent.
- **Try it:** connect to a file-system MCP server and have the agent find and summarize a file.
- **Outcome:** explain MCP and use existing servers.

### Unit 5.8: Building MCP Servers (Days 65–66)

- **Build your own server:** expose a real system (e.g., support requests).
- **Tools, resources and prompts:** actions as tools, reference data as resources, common tasks as prompts.
- **Good server design:** clear names/descriptions, typed inputs, useful errors, small results.
- **Local and remote servers:** remote servers need authentication.
- **Test before you connect:** use an MCP inspector.
- **Several servers, one agent:** avoid tool-name clashes.
- **Changing a server safely:** adding/renaming tools affects all agents; version and retest.
- **Try it:** MCP server with two tools and one resource; connect your agent and an MCP-compatible app.
- **Outcome:** package your systems as MCP servers.

### Unit 5.9: Memory (Days 67–68)

- **Short-term and long-term:** per-task/conversation vs across conversations.
- **Kinds of long-term memory:** user facts, event records, lessons learned.
- **Where memories live:** a database or LangGraph's memory store, separated per user.
- **When to save:** during the conversation or in the background.
- **What to save:** extract structured useful facts; avoid duplicates and saving everything.
- **When facts change:** update or replace outdated memories; decide conflict rules.
- **Recalling the right memories:** by relevance and recency; add only the few that help.
- **Forgetting:** expiry, and full deletion on request (including backups and logs).
- **Is memory helping?:** check recall accuracy and detect wrong/outdated memories.
- **Try it:** store a preference and a fact, verify in a new conversation, then change the fact.
- **Outcome:** agents that remember accurately and forget on request.

### Unit 5.10: Context Engineering Basics (Day 69)

- **Context is a budget:** instructions, history, passages, memories, tool descriptions and results compete for space.
- **Long runs go wrong:** goal drift, lost details, rising cost.
- **See what's really in there:** inspect the exact context each step.
- **Split the budget:** rough limits per component.
- **Build it on purpose:** decide what's always included, looked up, summarized or dropped.
- **Order and label:** keep instructions, evidence, memories and user input clearly separated.
- **Try it:** log context at steps 1, 10, 20; measure growth and mark noise.
- **Outcome:** diagnose context problems and plan context contents.

### Unit 5.11: Compaction, Offloading and Tool Context (Days 70–71)

- **Summarize without losing the plot:** keep goal, constraints, decisions, unfinished work.
- **Rolling summaries:** update incrementally.
- **Keep exact details exact:** store numbers/IDs separately.
- **Write things down:** scratchpad or file notes outside the context.
- **Trim tool results:** summarize/filter before returning; keep the full version elsewhere.
- **Only the tools it needs:** load relevant tools per step.
- **Use caching:** fixed instructions/tool descriptions first; measure savings.
- **Give sub-tasks their own context:** delegate to separate agents.
- **Try it:** run a 20+ step task three ways (keep all / summarize / summarize + trim + notes).
- **Outcome:** focused, affordable long agent runs.

### Unit 5.12: Harness Engineering (Day 72)

- **What a harness is:** everything around the model: loop, instructions, tools, context handling, memory, permissions, approvals, error handling, output checks.
- **The harness matters as much as the model:** the same model can succeed or fail depending on it.
- **Designing the tool set:** few general vs many narrow tools; control result detail; remove poorly used tools.
- **Let the agent check its own work:** run tests, validate schemas, re-check sources; feed errors back clearly.
- **Instructions on demand:** short core instructions, task-specific guidance loaded when needed.
- **Fit the harness to the model:** re-test when changing models.
- **Change one thing at a time:** treat harness changes like code changes.
- **Learn from existing harnesses:** study well-known coding agents.
- **Try it:** three harness changes (tool descriptions, self-check step, on-demand instructions), each measured on the same 10 tasks.
- **Outcome:** improve agents via the harness and prove it with measurements.

### Unit 5.13: Multi-Agent Patterns (Day 73)

- **Why more than one agent:** different tools, instructions, models or permissions; smaller contexts.
- **Supervisor and specialists:** one plans and delegates, then combines.
- **Handoffs:** sequential transfer between agents.
- **Parallel specialists:** simultaneous work, merged results.
- **Teams of teams:** supervisors managing supervisors.
- **What goes wrong:** duplicated work, miscommunication, lost info, conflicts, higher cost.
- **One strong agent is often enough:** split only when clearly beneficial.
- **Try it:** sketch single-agent vs supervisor designs and predict cost.
- **Outcome:** choose a multi-agent pattern or decide against one.

### Unit 5.14: Building Multi-Agent Systems (Days 74–75)

- **Build a supervisor:** in LangGraph, routing work to specialists.
- **Handoffs between agents:** pass tasks with needed context.
- **What each agent sees:** shared state vs message passing; share only what's needed.
- **Right tools, right model:** minimal tools/permissions; cheaper models for simple specialists.
- **Combining results:** merge outputs and resolve disagreements.
- **When a specialist fails:** retry, reassign or escalate to a person.
- **Track the cost:** steps, tokens, time, cost per agent and per task.
- **Agents from other teams (A2A):** an emerging standard for cross-team agent communication, like MCP for tools.
- **Try it:** supervisor with 2–3 specialists vs single agent on ten tasks.
- **Outcome:** a working multi-agent system judged against a single agent.

### Unit 5.15: Reliability and User Experience (Day 76)

- **Common agent failures:** wrong tool, bad inputs, tool outages, repetition, incomplete plans, running too long.
- **Budgets:** limit steps, time, retries, cost.
- **Know when to hand over:** fall back to simpler paths or a person.
- **Show progress:** which tool is in use and what was found.
- **Keep users in control:** cancel, pause, resume, redirect.
- **Try it:** stream progress and add a cost cap.
- **Outcome:** agents that fail predictably and keep users informed.

## Part 3: Evaluating Agents (Days 77–79)

### Unit 5.16: Why Agent Evals Are Different (Day 77)

- **The final answer isn't enough:** right answers can come from wrong paths; inspect steps.
- **Read the traces:** review 20 runs step by step.
- **Group the failures:** wrong tool, bad inputs, loops, early give-up, lost context, bad handoff.
- **What to measure:** decisions, tool use, path, task success, recovery, memory, efficiency, safety.
- **Try it:** read 20 traces, note issues, group them.
- **Outcome:** find common failures from actual behavior.

### Unit 5.17: Evaluating Tool Use and Task Success (Day 78)

- **An agent test set:** tasks with success criteria and expected tools.
- **Right tool, right inputs:** check choice and parameters.
- **A sensible path:** unnecessary calls, repeats, wrong order, stopping early.
- **Did it get the job done?:** task completion rate.
- **Judging the path:** model judge, verified against your own scoring.
- **Try it:** 15-task test set scored on tool choice, inputs, success.
- **Outcome:** measure decision quality and task completion.

### Unit 5.18: Testing Recovery, Memory and Efficiency (Day 79)

- **Break things on purpose:** failing tools, bad data, unavailable tools, missing info.
- **Did it notice and recover?:** retry, alternate tool, replan, ask for help.
- **Test memory:** correct recall, no outdated or cross-user memories.
- **Efficiency:** steps, calls, tokens, time, cost per task.
- **Try it:** fail the search tool on three tasks, return empty results on two, run three memory-dependent tasks.
- **Outcome:** know how the agent handles trouble, whether memory is trustworthy, and per-task cost.

## Part 4: Securing Agents (Days 80–83)

### Unit 5.19: Agent Threats and Least Privilege (Day 80)

- **The dangerous combination:** untrusted content + access to private data or powerful tools + a way to send data out.
- **Least privilege:** only the tools and permissions the task needs.
- **Check tool use in code:** validate tool, inputs and permissions outside the model.
- **Memory can be attacked:** injected memories can affect future conversations; never leak across users.
- **Limits stop runaways:** step/time/cost limits.
- **Try it:** list each agent's tools/permissions, write the attacker's options, restrict.
- **Outcome:** explain agent attacks and minimize access.

### Unit 5.20: Approval Gates and Sandboxing (Day 81)

- **Approval for anything irreversible:** messages, record changes/deletes, payments, running code.
- **Run generated code in a sandbox:** E2B, Modal or containers; never inside your app.
- **Restrict the sandbox:** files, network, credentials, runtime, compute.
- **Generated commands are untrusted:** queries, shell commands, URLs.
- **Limit the damage:** even a fully tricked agent should do little harm.
- **Try it:** sandboxed Python tool; confirm outside-file reads and network calls are blocked.
- **Outcome:** safe powerful actions with approvals and isolation.

### Unit 5.21: Securing MCP and Tools (Day 82)

- **Untrusted MCP servers:** treat third-party servers like third-party code.
- **Hidden instructions in tools:** descriptions and results can carry injected instructions.
- **Servers that change:** approved servers can later change tools; review updates.
- **Over-broad access:** narrowest permissions and credentials.
- **Hijacked tool use:** your code checks (5.19) stop injected tool calls.
- **Try it:** hide instructions in a tool description and a tool result; see if behavior changes.
- **Outcome:** connect to MCP servers without giving up control.

### Unit 5.22: Red-Teaming Your Agent (Day 83)

- **Attack your own agent:** misuse tools, exceed permissions, reveal secrets, act without approval, poison memory, loop/spend.
- **Attack through every door:** messages, documents, tool results, MCP descriptions, memories.
- **Measure unsafe actions:** attack success rate and attempted forbidden actions.
- **Every successful attack becomes a test.**
- **Safety vs usefulness:** ensure defenses don't cripple the agent.
- **Try it:** 10 attacks across all entry points; fix and re-run.
- **Outcome:** test like an attacker and keep permanent tests.

---

# MODULE 6: Shipping to Production (Days 84–100)

**Prerequisite:** Module 5, your agent, and your full regression suite. 15 units.
**By the end:** turn a working AI app into a live service: justify its design, deploy it, block worse releases, roll out gradually, monitor quality/cost/speed, cut costs (caching, open models, fine-tuning), handle incidents and confirm end-to-end security.

### Unit 6.1: Designing an AI System (Day 84)

- **Start from the requirements:** users, volume, response time, quality bar, budget, data rules.
- **Rough numbers first:** convert users into requests/minute, tokens/day, cost/month.
- **Choose the simplest design that works:** prompting, RAG, agent or a mix; justify each extra piece.
- **The core trade-offs:** quality, speed and cost pull against each other.
- **Plan for failure:** what happens when the model, a tool or a provider is down or slow.
- **Write it down:** record the design, rejected alternatives and reasons.
- **Try it:** one-page design with requirements, traffic, monthly cost estimate, diagram and three decisions with rejected alternatives.
- **Outcome:** a defensible design that starts from requirements, not favorite tools.

### Unit 6.2: From Prototype to Service (Day 85)

- **A backend for your agent:** put it behind a web API.
- **Stream through the API:** send answers and progress as they happen.
- **Know who's asking:** per-user identity for permissions, memory, cost tracking.
- **Keep the app stateless:** store conversations, progress and memories in a database.
- **Long tasks in the background:** background jobs with progress reporting.
- **Protect the API:** authentication and per-user rate limits.
- **Try it:** streaming endpoint; call it as two users and verify separate conversations and memories.
- **Outcome:** an agent running as a multi-user service.

### Unit 6.3: A Real Frontend (Day 86)

- **A chat interface:** streaming answers.
- **Show the evidence:** clickable citations.
- **Show what the agent is doing:** progress, tool in use, findings.
- **Approval in the interface:** show the proposed action; approve or reject.
- **Collect feedback:** ratings and "what went wrong".
- **Handle failure in the interface:** clear messages, retry, cancel.
- **Try it:** chat page with streaming, citations, approval prompt and thumbs up/down.
- **Outcome:** an interface showing what the agent knows, does and wants to do.

### Unit 6.4: Packaging and Deploying (Days 87–88)

- **Package everything:** containerize.
- **Run all the parts together:** app + database + vector DB started with one command.
- **Deploy to a managed host:** a service that runs containers for you.
- **Secrets in the right place:** host secret storage, never code or containers.
- **Separate environments:** dev, test, live with separate settings, data and keys.
- **Health checks:** host restarts unhealthy apps.
- **Data changes on deploy:** plan database migrations and re-indexing.
- **Try it:** deploy to test then live; change a setting only in test and confirm live is unaffected.
- **Outcome:** reliable deployment with safe, separated secrets and settings.

### Unit 6.5: Automatic Checks on Every Change (Day 89)

- **Test on every change:** CI (e.g., GitHub Actions) on each push.
- **Your regression suite becomes a gate:** RAG quality/security and agent quality/security tests block worse releases.
- **Fast checks and full checks:** quick subset per change, full suite per release.
- **Prompts and models count as code:** same checks for prompt, model or setting changes.
- **Release only what passed:** deploy exactly the verified version.
- **Try it:** quick-subset checks on every push; push a worse prompt and confirm it's blocked.
- **Outcome:** impossible to release a version worse than the current one.

### Unit 6.6: Releasing Safely (Day 90)

- **Every release can be traced:** record code, prompt, model and settings in use.
- **Release to a few users first:** canary a small traffic share.
- **Test in the shadows:** run the new version on real questions without showing answers.
- **Compare versions with real users:** A/B on quality, ratings, cost.
- **Turn features on and off:** feature flags without redeploying.
- **Roll back fast:** return to the previous version in minutes.
- **Try it:** release a prompt to 10% of traffic, compare, then roll out or back.
- **Outcome:** gradual, comparable, reversible releases.

### Unit 6.7: Watching It in Production (Days 91–92)

- **Trace live traffic:** follow any user's problem step by step.
- **The numbers to watch:** requests, errors, typical and slowest response time, tokens, cost per request/user.
- **Check quality on live traffic:** score a sample of real answers with your judges.
- **Listen to users:** ratings, comments, rephrasing, abandoned tasks.
- **Spot changes over time:** drift in questions, documents or model behavior.
- **Alerts:** thresholds for errors, latency, cost, quality.
- **Close the loop:** feed real failures and poor ratings into the test set.
- **Keep private data out:** mask personal details in traces and logs.
- **Try it:** dashboard (requests, errors, latency, cost, live quality score), one alert triggered on purpose, three poorly rated answers added to the test set.
- **Outcome:** know real-user performance and turn it into improvements.

### Unit 6.8: Cost, Speed and Scale (Day 93)

- **Find where time and money go:** break down search, model calls, tools.
- **Caching:** repeated answers, fixed prompt sections, search results.
- **Answer similar questions once:** semantic caching; decide what must never come from cache.
- **Cheaper models for easy work:** route simple requests to a fast, cheap model.
- **Do things at the same time:** parallelize independent steps.
- **Limits per user:** cap usage to prevent runaway bills.
- **Try it:** add caching and routing; measure cost, latency and quality before/after.
- **Outcome:** faster, cheaper app with proof and no quality loss.

### Unit 6.9: Model Gateways and Managed Platforms (Day 94)

- **One gateway for all models:** holds keys, tracks usage, switches providers.
- **Backup providers:** automatic failover; the backup may answer differently.
- **Changing models is a release:** swap without code changes but test first.
- **Cloud AI platforms:** serve models inside your own cloud account when public APIs aren't allowed.
- **Choosing a path:** compare public APIs, cloud platforms and self-hosting on data rules, cost, effort, access to new models.
- **Try it:** gateway in front of two providers; fail the main one and confirm the backup meets latency limits.
- **Outcome:** avoid single-provider dependence and know when a cloud platform fits.

### Unit 6.10: Serving Open Models Yourself (Day 95)

- **Why host your own:** predictable high-volume cost, data control, version control.
- **The hosting ladder:** open model via provider → rented GPUs → own servers.
- **Serving software:** tools like vLLM batch requests efficiently.
- **Smaller models, less memory:** quantization trades quality for memory.
- **Sizing the hardware:** model size, precision and input length decide GPU memory.
- **Per token or per hour:** per-token for low/uneven traffic; hourly GPUs only if kept busy.
- **Cold starts:** scaled-down deployments answer first requests slowly.
- **Try it:** serve a small open model; compare quality, latency, cost per 1,000 requests vs hosted.
- **Outcome:** decide when self-hosting pays off and how to host.

### Unit 6.11: Should You Fine-Tune? (Day 96)

- **What fine-tuning changes:** teaches behavior, format, style or narrow skills; poor for changing facts (use RAG).
- **Prompt, RAG or fine-tune:** prompt first, RAG for knowledge, fine-tune only when evals show those are exhausted.
- **When it wins:** consistent format/tone, high-volume narrow tasks, specialist language, replacing a large model with a small one.
- **What it costs:** data, training, serving, retraining when base models change.
- **Data is most of the work:** gather from traces, clean, de-duplicate, remove personal data, format as conversations.
- **Quality beats quantity:** a few hundred careful examples often beat thousands of messy ones.
- **Keep a held-out set:** never train on test data.
- **Examples from a stronger model:** can help; check provider terms.
- **Try it:** 200 training + 50 held-out examples for a narrow task; measure the prompted baseline.
- **Outcome:** decide if fine-tuning is justified and prepare a clean dataset and fair baseline.

### Unit 6.12: Fine-Tuning with LoRA (Day 97)

- **Full vs lightweight tuning:** full updates all weights; LoRA trains a small add-on for most of the benefit.
- **Even less memory:** QLoRA tunes a compressed model on a modest GPU.
- **Choosing a base model:** small open model good at your task with a suitable license.
- **Tools:** Hugging Face PEFT, Unsloth; free or rented GPUs suffice.
- **Key settings:** learning rate, number of epochs, add-on size; change one at a time.
- **Watch for overfitting:** stop when held-out results worsen.
- **Managed fine-tuning:** providers tune from an uploaded dataset; check where the model lives and portability.
- **Forgetting other skills:** heavy tuning can degrade other abilities.
- **Try it:** LoRA fine-tune on 200 examples using a free notebook GPU; log training and held-out results.
- **Outcome:** run a LoRA fine-tune, choose sensible settings, spot overfitting.

### Unit 6.13: Evaluating, Serving and Distilling Fine-Tuned Models (Day 98)

- **Beat the baseline or don't ship it:** keep only if clearly better or much cheaper at the same quality.
- **Check what else changed:** run the full regression suite.
- **Cost and speed:** compare with the large model it might replace.
- **Learning from preferences:** DPO trains from better/worse answer pairs.
- **Distillation:** train a small model on a strong model's answers for one task.
- **Serving a tuned model:** merge the LoRA add-on or serve several add-ons on one base (e.g., vLLM).
- **Treat it like any release:** version, release gradually, retrain when base model or data changes.
- **Try it:** compare tuned vs prompted model on accuracy, cost, latency; run regression suite; decide.
- **Outcome:** prove with numbers whether a tuned or distilled model earns its place.

### Unit 6.14: When Things Break (Day 99)

- **Common incidents:** provider outage, slowness, cost spike, quality drop, looping agent.
- **Decide what to do:** roll back, switch provider, disable a feature, or investigate.
- **Runbooks:** short step-by-step guides for likely incidents.
- **Find the cause:** traces, dashboards, version records.
- **Learn from it:** blameless review and a new test.
- **Try it:** runbooks for a provider outage and a quality drop; simulate one and follow it.
- **Outcome:** calm, fast incident response that reduces repeats.

### Unit 6.15: Whole-App Security Review (Day 100)

- **Threat model the whole system:** frontend, API, RAG, agent, MCP servers, memory, models, logs.
- **Secrets and personal data everywhere:** code, containers, logs, traces, caches, test sets, fine-tuning data.
- **Checks at every boundary:** user input, documents, tool results, model output.
- **The API and frontend:** authentication, per-user limits, permissions; prevent data exfiltration via links or images.
- **A final red-team:** attack the live app through every door.
- **Use the checklist:** OWASP Top 10 for LLM Applications.
- **Try it:** full security suite on the deployed app plus five new API/frontend attacks.
- **Outcome:** confirm the complete deployed app is secure end to end.

---

# MODULE 7: AI System Design (Added module, from the Full Roadmap, Level 9)

**Source:** Full roadmap Level 9 "AI System Design" (Phase 5, Production). Not part of the 100-day track, where only Unit 6.1 touches this topic; here it is covered in depth.
**Goal:** design a complete production AI application and defend every architectural choice across quality, latency, cost, reliability and user experience.
**Why it matters:** interview weight 9/10 (dedicated design round for mid-level and senior roles), on-the-job weight 8/10 (grows with seniority).
**Prerequisite:** AI Security (Level 8) and Modules 1–6 above.
**Structure:** 9 modules (M9.1–M9.9) + a capstone (M9.10).

## M7.1 (M9.1): The AI System Design Framework

- **Starting from requirements and constraints:** use case, success criteria, constraints, traffic, latency and quality needs, privacy, cost budget.
- **Prompting, RAG, agents or fine-tuning:** choose deliberately, or combine.
- **Quality ↔ latency ↔ cost:** the central trade-offs, plus complexity and reliability.
- **Baseline before sophistication:** build a simple baseline first.
- **Outcome:** turn a product requirement into a defensible architecture, not a favorite technology.

## M7.2 (M9.2): Context and Knowledge Architecture

- **Where knowledge should come from:** model knowledge, prompt/context, RAG, memory or external tools.
- **The full context pipeline:** ingestion → retrieval → reranking → context assembly → generation.
- **Combining retrieval approaches:** vector, keyword, graph and others where justified.
- **Freshness, access control and provenance:** updates, permissions, scale, source tracking.
- **Retrieved dynamically or held in context:** what to fetch on demand vs keep permanently.
- **Outcome:** a knowledge layer that supplies the right information at the right time.

## M7.3 (M9.3): Model Routing and Gateways

- **What a model gateway does:** abstraction layer between application code and model providers.
- **Routing on task, cost and risk:** by task type, capability, latency, context length, cost, modality or risk.
- **Model cascades:** cheaper/faster models for easy requests, stronger ones for hard ones.
- **Provider fallbacks and quality risk:** backups, and the quality risk of switching models.
- **Centralizing auth, limits and tracking:** authentication, usage tracking, retries, rate limits, provider abstraction.
- **Outcome:** applications not hard-coded to a single model.

## M7.4 (M9.4): Latency and Inference Optimization

- **Breaking latency into components:** retrieval, time-to-first-token, generation, tool calls, network.
- **Streaming for perceived latency.**
- **Parallelizing independent work:** avoid unnecessary sequential LLM calls.
- **When a smaller model will do.**
- **Context reduction and batching:** prompt/context trimming, batching, inference-service optimizations.
- **Throughput and utilization at scale.**
- **Outcome:** locate where an app is slow and pick the right fix instead of just upgrading infrastructure.

## M7.5 (M9.5): Caching Architecture

- **The four caches in an AI system:** exact-response, prompt/context, retrieval, and semantic caching.
- **What is safe to cache:** plus expiration and invalidation strategies.
- **Semantic caching:** when different queries can share an answer.
- **Freshness vs latency and cost:** the trade-off between freshness, quality, latency and cost.
- **Measuring the hit rate:** measure real savings rather than assuming.
- **Outcome:** caching as an architectural optimization, not an afterthought.

## M7.6 (M9.6): Reliability, Guardrails and Graceful Degradation

- **Designing for every failure:** model failures, provider outages, tool failures, rate limits, malformed outputs, retrieval failures.
- **Timeouts, circuit breakers and queues:** plus retries, fallbacks and rate limits.
- **Guardrails at system boundaries:** don't rely on the model alone.
- **Graceful degradation:** simpler model, reduced functionality, cached answer, human escalation, or explicit inability.
- **Human approval for high-risk actions:** especially irreversible ones.
- **Service-level expectations:** availability, latency and quality targets.
- **Outcome:** a system that stays useful when individual AI components fail.

## M7.7 (M9.7): AI Pipeline Orchestration

- **The end-to-end flow:** retrieval, models, tools, agents, guardrails, caches and external systems.
- **Deterministic vs model-delegated decisions.**
- **Sync, async, queued or long-running:** choose the execution style.
- **Avoiding unnecessary agentic complexity:** use a deterministic pipeline when sufficient.
- **Component boundaries:** parts can be changed, evaluated and scaled independently.
- **Outcome:** individual AI components assembled into a coherent production system.

## M7.8 (M9.8): Monitoring, Feedback and Continuous Improvement

- **Wiring evals and feedback into production:** evals, traces, operational metrics, user feedback.
- **What to monitor:** quality, latency, cost, errors, retrieval behavior, tool usage, model/provider performance.
- **Explicit and implicit feedback:** ratings plus behavioral signals.
- **Failures become evaluation cases.**
- **Versioning for traceable regressions:** models, prompts, retrieval configs, app changes.
- **The loop:** deploy → observe → collect feedback → evaluate → improve → redeploy.
- **Outcome:** products that improve from real-world use instead of staying static.

## M7.9 (M9.9): AI System Design Case Studies

Whiteboard each system end-to-end and defend the choices. For each: requirements → baseline → architecture → bottlenecks → failure modes → scaling → trade-offs.

### Case 1: Customer-Support Copilot

- RAG and fresh knowledge
- Citations
- Model routing
- Caching
- Escalation to humans
- Latency and cost constraints

### Case 2: Coding Assistant

- Large-codebase context
- Retrieval
- Tool and code execution
- Model selection
- Sandboxing
- Long-running tasks

### Case 3: Deep Research Agent

- Search and retrieval
- Planning
- Parallel tool calls
- Context management
- Long-running execution
- Citations
- Cost budgets and failure recovery

**Outcome:** defend architectural decisions rather than just drawing boxes and arrows.

## M7.10 (M9.10): Capstone: AI System Design Document

Choose a substantial AI product and produce a complete design covering:

- Requirements and success criteria
- Expected traffic and constraints
- High-level architecture
- Model choice and routing
- Context / RAG architecture
- Tools and agents where required
- Caching strategy
- Latency and cost budget
- Reliability and fallback strategy
- Security and guardrails
- Evaluation and monitoring
- User-feedback loop
- Scaling strategy
- An architecture diagram, plus explicit justification of major trade-offs and rejected alternatives

**Standard:** a design you could defend in an AI system-design interview or an engineering architecture review.

**Further reading (as listed by the source):** _AI Engineering_ by Chip Huyen.

---

# Appendix: Content Totals

| Module                      | Days      | Units        |
| --------------------------- | --------- | ------------ |
| 1. LLM Fundamentals         | 1–6       | 6            |
| 2. Building with LLM APIs   | 7–17      | 9            |
| 3. Prompt Engineering       | 18–24     | 8            |
| 4. RAG                      | 25–56     | 26           |
| 5. Agents                   | 57–83     | 22           |
| 6. Shipping to Production   | 84–100    | 15           |
| **Total (100-day track)**   | **100**   | **86**       |
| 7. AI System Design (added) | extension | 9 + capstone |
