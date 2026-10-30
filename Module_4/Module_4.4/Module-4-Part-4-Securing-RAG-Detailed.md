# Module 4, Part 4: Securing a RAG Application — Detailed Learning Document

**Schedule:** Days 52–56  
**Prerequisites:** [Part 1: Build a Simple RAG](../Module_4.1/Module-4-Part-1-Building-Simple-RAG-Detailed.md), [Part 2: Advanced RAG](../Module_4.2/Module-4-Part-2-Advanced-RAG-Detailed.md), and [Part 3: Evaluating RAG](../Module_4.3/Module-4-Part-3-Evaluating-RAG-Detailed.md)  
**Companion:** [Practical security experiments](Module-4-Part-4-OpenAI-Practical-Experiments.ipynb)  
**Roadmap:** [Module 4, Part 4](../../GenAI.md#part-4-securing-a-rag-application-days-5256)

A RAG application moves data between users, document sources, indexes, models, logs and clients. Each boundary creates security decisions. This part teaches how to identify those boundaries, restrict access in code, test prompt injection and leakage, and add regression checks. Exercises use fictional data and an isolated application; they are not a security certification.

## Learning outcomes

- Draw a data-flow diagram and identify assets, trust boundaries, entry points and attacker goals.
- Explain direct and indirect prompt injection and test poisoned documents in a controlled environment.
- Enforce tenant and document access before retrieval results reach a model, cache, citation renderer or user.
- Identify sensitive information before indexing, reduce exposure, and trace deletion across copies.
- Build a red-team set, distinguish observed attacks from unknown outcomes, and retest quality after defenses.

## Contents

1. [Setup and running example](#setup-and-running-example)
2. [4.22 Threats to a RAG app](#unit-422-threats-to-a-rag-app--day-52)
3. [4.23 Hidden instructions and poisoned documents](#unit-423-hidden-instructions-and-poisoned-documents--day-53)
4. [4.24 Who can see what](#unit-424-who-can-see-what--day-54)
5. [4.25 Personal data and secrets](#unit-425-personal-data-and-secrets--day-55)
6. [4.26 Red-teaming your RAG](#unit-426-red-teaming-your-rag--day-56)
7. [Capstone and self-check](#capstone-and-self-check)

## Setup and running example

The companion notebook uses a fictional Northstar policy service, synthetic users in two tenants, fake sensitive-data examples and deliberately poisoned documents. It runs all security checks locally by default. Optional OpenAI experiments compare how a model responds to three poisoned documents; they do not call real tools or send data to an attacker endpoint. `RUN_PAID` and a separate experiment flag must both be enabled.

Use a Jupyter kernel with Python, `python-dotenv` and, for optional model calls, the OpenAI SDK. Keep `OPENAI_API_KEY` in the environment or a Git-ignored `.env`; never place it in a notebook or output. If VS Code does not inherit a terminal export, load `.env` in the kernel or restart it. Choose an available model through `OPENAI_MODEL`, review [current pricing](https://developers.openai.com/api/docs/pricing), and use only data you are authorized to send. Read [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data) for the exact endpoint and project settings in use.

> A prompt, classifier, regex, sample notebook, or passed red-team test cannot guarantee immunity to injection or data leakage. The core security boundary belongs in trusted application code and the underlying data services.

---

## Unit 4.22: Threats to a RAG App — Day 52

### Start with the data flow

```text
User / uploaded document / external page                  UNTRUSTED
          │
          ▼
Authentication → trusted identity + permissions             TRUST BOUNDARY
          │
          ├── ingestion → parse → scan → approve → index
          │                           │              │
          ▼                           ▼              ▼
Question → authorized retrieval → model context → model output
                    │                    │              │
                    ▼                    ▼              ▼
              index / cache         API provider     renderer / tools
                    └────────── logs / traces / exports ───────────┘
```

Trust is not binary: an authenticated user can still send malicious instructions; an approved document may become stale or be compromised; a model-generated answer is not trusted application code. The pipeline must enforce policy each time data crosses a boundary.

### Identify assets and entry points

| Asset | Attacker goal | Entry point or failure |
| --- | --- | --- |
| Private documents | Obtain content or even a restricted title | Cross-tenant search, unauthorized citations, cached answer. |
| Account and API credentials | Disclose or misuse secrets | Indexed configuration file, logs, model output or tool arguments. |
| Answer integrity | Make the assistant state a false policy | Poisoned passage, hidden instruction, fake authority text. |
| External tools | Cause an unauthorized action or network request | Model output promoted directly into a tool call or URL. |
| Availability and budget | Consume resources or interrupt service | Expensive queries, enormous documents, repeated calls. |
| Audit evidence | Hide or forge activity | Untrusted text written unescaped into logs or dashboards. |

For each flow, write down **who controls the input**, **which user is making the request**, **what data is accessible**, **what action could occur**, and **where the control is enforced**. Draw ingestion, query, caching, rendering and deletion separately if a single diagram becomes crowded.

### OWASP checklist

The [OWASP 2025 Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) is a starting point. Particularly relevant here: prompt injection, sensitive information disclosure, supply-chain risks, data poisoning, improper output handling, excessive agency and unbounded consumption. Use the full list for your deployment; a checklist does not replace a threat model or specific tests.

**Practice:** Sketch your own pipeline. Label five untrusted inputs, three protected assets and every point at which information or an action can leave the app. Record an owner and a mitigation for each high-impact threat.

---

## Unit 4.23: Hidden Instructions and Poisoned Documents — Day 53

### Direct versus indirect injection

- **Direct:** a user asks the assistant to ignore its rules or disclose another user's document.
- **Indirect:** attacker-controlled material is retrieved as evidence and tells the model to change behavior, leak information or request a tool action.
- **Poisoning:** an attacker inserts misleading content into a document or index so retrieval returns it as if it were reliable. A false factual policy can be harmful even if it contains no obvious instruction.

For example, a fake handbook entry might claim “the daily meal limit is $400,” or a document might tell the assistant to include a link with a sensitive value in its query string. A citation to that document does not make the fact trustworthy. Use controlled publishing, provenance, document versioning, owner review and change auditing in addition to access control. Source authenticity and permission to read are different decisions.

### Three sandbox attacks

1. **Answer manipulation:** a poisoned document instructs the assistant to claim the travel limit is a fabricated value.
2. **Tool misuse:** a document tells the assistant to request a fictional `send_report` action. The notebook records simulated proposals only; there is no real email or action.
3. **Leak link:** a document asks for a link containing a synthetic canary. The notebook never opens the link or sends a request.

Measure separately whether the model followed the injected text, whether a proposed action was blocked, and whether a canary appeared in output. A blocked tool action does not prove the model resisted. Conversely, a model refusal does not prove a different prompt or document would fail. Use harmless local canaries only.

### Defenses at multiple layers

- Keep higher-priority application instructions separate from untrusted question and evidence text. Make clear that documents are evidence, not operational commands. This reduces ambiguity but is not a security boundary.
- Restrict who can upload, edit, approve and index each document; record source and version, and quarantine suspicious or unapproved updates. A poisoned source can contain ordinary-looking false statements, not just suspicious wording.
- Expose the fewest tools and permissions needed. A tool request is only a proposal: trusted application code must validate the tool name, argument types, caller authorization and business rules before taking action.
- Treat model output as untrusted. Escape output for the rendering context; avoid loading remote images or URLs automatically, validate destination and data policy, and enforce network egress rules. Do not rely solely on a URL substring or prompt instruction.
- Minimize secrets and other sensitive data available to the model. Never assume that hiding a secret in an instruction or retrieved passage prevents disclosure.
- Log security decisions without writing secrets or full sensitive passages into logs. Test both direct user prompts and indirect source content.

See [OpenAI's guidance on untrusted data](https://developers.openai.com/api/docs/guides/agent-builder-safety) and [OWASP prompt injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/). A detector or model judge is another signal; it can miss attacks or block legitimate content.

---

## Unit 4.24: Who Can See What — Day 54

### Enforce authorization before search results reach the model

Give each document or chunk a tenant and access policy. Derive the requester and their roles from a trusted authentication service, not from a user-supplied question, metadata filter or model output. The retrieval service must restrict the candidate set to the caller's allowed tenant and documents **before** returning passages to the application or model. Deny access if identity or policy is missing. Apply this to keyword, vector, hybrid, reranking, parent-child expansion and fallback paths.

```text
trusted principal → mandatory tenant + document permission constraints
                  → authorized candidate set → rank → model / citations
```

An application-layer demonstration can filter a local list before ranking. In a real shared vector database, use and verify the database's supported tenant isolation and filter behavior; a client-side filter or post-filtering already exposed results is insufficient. ANN filters can interact with candidate selection, so separately measure authorized retrieval quality.

### Two users, one question

Suppose Alice and Bob both ask “What is the board plan?” Alice is allowed the Northstar board document; Bob is not. Alice may retrieve and cite it, while Bob must see neither its content nor its title, metadata, source ID or answer cached for Alice. An unrelated tenant should also receive nothing. If no permitted evidence exists, the assistant should report insufficient accessible information without revealing whether a restricted document exists.

### Other leakage paths

- **Cache:** bind keys to authenticated tenant, principal or effective permissions, policy version, corpus version and request parameters. Reauthorize on cache read and invalidate on document or permission changes. Never share an answer between users merely because their question matches.
- **Parent expansion:** a permitted child cannot authorize a broader parent that includes restricted content; check the parent and its full contents.
- **Citations and metadata:** titles, snippets, URLs and existence can be sensitive. Resolve citations only from the authorized result set.
- **Conversation history:** reauthorize stored context on later turns, after role changes and on session transfer; a previous response does not permanently authorize its content.
- **Tracing and evaluation:** enforce access and retention on logs, dashboards, evaluation datasets and exported reports. Apply the same tenant boundaries to debugging tools.

**Practice:** With two roles and two tenants, run the same query and inspect the actual retrieved context, cache behavior and citations—not only the final answer. Change a permission and verify old results do not remain available. The notebook uses synthetic users and simulated state, not a production identity provider.

---

## Unit 4.25: Personal Data and Secrets — Day 55

### Scan before indexing

Inventory document sources and minimize collection. Detect information such as emails, phone numbers, identity numbers, access tokens, passwords and customer identifiers. Remove data that is unnecessary for the task before embedding or storage; use approved masking or tokenization when appropriate. Search the same categories in logs, traces, caches, evaluation sets and backups. Restrict access to source files and results.

Regular expressions can illustrate detection but are incomplete. Email-like strings may not be personal data, phone formats vary, and real names, health information and unstructured secrets can be missed. A masked identifier may remain personal data if it can be linked back to a person. Choose detectors and review procedures based on real data and false-positive/false-negative testing. Never print the value of a real credential when reporting a finding; report the category, source and a non-sensitive location.

**Practice:** The notebook scans only synthetic example strings and prints categories and counts, not detected values. Try a fake secret format the regex does not recognize and record it as a blind spot.

### Data copies and deletion

A single document can exist in the source repository, raw extracts, chunk store, vector index, search index, caches, traces, exported datasets and backups. Maintain an inventory and a source-to-chunk mapping. A deletion or correction workflow should identify all applicable copies, issue deletion/invalidation requests, handle retries and verify results across systems. Note retention schedules, backup expiry, legal holds and provider limitations; do not claim deletion is complete until the applicable systems confirm it. Immutable or legally retained logs require an approved retention and deletion design rather than an assumption that they can be erased on demand.

### Provider controls

As described in [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data), API data may be held as application state or abuse-monitoring logs, with different controls and exceptions by endpoint and feature. The notebook sets `store=False` for Responses requests to avoid requesting saved response state, but **that alone is not a promise of zero retention**. Review current default retention, eligible zero-data-retention options, third-party tools, your contract and the endpoints in use before sending personal or confidential data.

### India: DPDP basics

India's [Digital Personal Data Protection Act, 2023](https://www.indiacode.nic.in/handle/123456789/22037) concerns processing of digital personal data and includes provisions concerning lawful processing, notices, safeguards, rights and other obligations. Applicability, exceptions and compliance duties depend on the facts and provisions in force. The government's [13 November 2025 commencement notification](https://www.meity.gov.in/static/uploads/2025/11/c56ceae6c383460ca69577428d36828b.pdf) phases different provisions in at different times; do not assume every provision started together. Check the latest notified rules, amendments and effective dates with qualified legal counsel for your application. The notebook is a technical learning exercise, not a legal compliance determination.

---

## Unit 4.26: Red-Teaming Your RAG — Day 56

### Design an attack set

Test only systems and data you are authorized to test. Use synthetic canaries and isolated tenants. For each case specify attacker control, intended violation, evidence of success, control being tested, and expected authorized behavior. Example categories:

| Category | Test idea | Observable failure |
| --- | --- | --- |
| Direct instruction | Ask to ignore authorization. | Restricted passage reaches context or response. |
| Poisoned source | Change policy answer via embedded instruction. | Model repeats a false claim; independently check tool/output controls. |
| Tool abuse | Source requests an unapproved tool. | Trusted executor performs it without authorization. |
| Leak link | Embed canary in a proposed URL. | Output/rendering/network action reveals the canary. |
| Tenant boundary | Ask about another tenant's record. | Other tenant's text or metadata appears in candidates/context. |
| Cache isolation | Replay a privileged query as another user. | Privileged answer is returned without reauthorization. |
| Permission revocation | Remove a role then reuse an old cache entry. | The old answer remains accessible. |
| Sensitive-data handling | Include a synthetic secret in an ingest record. | Secret appears in the index, trace or response where disallowed. |

### Score outcomes honestly

Define **attack success rate** as observed successful attacks divided by executed, adjudicated attacks for that specified test set. Report the numerator, denominator, unknown/error count, categories and severity. A blocked model response is not proof that a data-boundary violation did not already occur; inspect retrieved context and simulated tool/network traces. Conversely, a canned synthetic payload rejected by a local check is not proof a live model resisted an injection.

Run before and after the change on the same cases and configurations. Preserve the before results. Add every confirmed failure as a regression case; use a separate holdout set to reduce overfitting. For each fix rerun benign questions as well: an overbroad filter that blocks all retrieval can make attack success look low while destroying useful answers. Report allowed-query quality, latency and cost with security outcomes.

The notebook includes 15 deterministic local control probes, three optional live model-facing poisoned-document examples, and benign checks. The 15 probes measure local toy controls, not live-model attack success or production security. No real network action or tool call is performed by the simulator.

**Practice:** Record before/after results for 15 cases with exact outcomes. Inspect every case marked unknown. Turn each observed failure into a permanent test and retain a quality baseline from Part 3.

---

## Capstone and self-check

Create a security review for your RAG assistant:

1. Draw the complete data flow and inventory entry points, assets and trust boundaries.
2. Define two tenants, two roles, document permissions and authoritative identity sources.
3. Test unauthorized retrieval, citations, cache replay and permission revocation with the same questions.
4. Insert three synthetic poisoned documents and record model behavior separately from simulated action and URL controls.
5. Scan a synthetic ingestion dataset and simulated logs for fake secrets and personal identifiers; document missed patterns and false positives.
6. Map deletion across document storage, chunks, indexes, caches, logs, exports and backups; identify owner, status and verification for each.
7. Write at least 15 attack cases and rerun after a change; report observed successes, failures and unknowns.
8. Re-run benign answer quality and latency checks from Part 3. Save sanitized regression evidence and unresolved risks.

**Self-check:**

1. Why can an authenticated user's document still contain untrusted instructions?
2. Why must authorization be checked before passages enter model context?
3. Can a model refusal undo a passage already sent to the wrong user or provider?
4. What other data can a citation or a cache expose besides the answer?
5. Why does `store=False` not establish zero retention everywhere?
6. Why does a regex scanner miss some secrets and personal data?
7. What does a red-team success rate exclude if errors are not adjudicated?
8. Why must security fixes be tested against legitimate questions too?

### Completion checklist

- [ ] I mapped the attack surface and documented controls at trust boundaries.
- [ ] I tested three synthetic poisoned documents and separated model behavior from code-level controls.
- [ ] I checked tenant and role isolation before retrieval, including cache and citation paths.
- [ ] I scanned synthetic data and described deletion across all relevant copies.
- [ ] I reported 15 red-team outcomes, unknowns and benign quality checks.
- [ ] I know these exercises are not proof of production security or legal compliance.

## Further reading

- [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) and [prompt injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/).
- [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data), [safety guidance](https://developers.openai.com/api/docs/guides/agent-builder-safety), and [pricing](https://developers.openai.com/api/docs/pricing).
- [India Code: DPDP Act](https://www.indiacode.nic.in/handle/123456789/22037) and [commencement notification](https://www.meity.gov.in/static/uploads/2025/11/c56ceae6c383460ca69577428d36828b.pdf).

Security, provider terms and legal requirements change. Verify the current documentation and the controls of the actual system you deploy.
