# Module 5, Part 2: Advanced Agents — Detailed Learning Document

**Days 64–76 | Units 5.7–5.15**  
Prerequisite: Module 4 and the preceding parts of Module 5. Companion: [practical notebook](Module-5-Part-2-OpenAI-Practical-Experiments.ipynb).

## Learning outcomes

- Explain and apply mcp fundamentals.
- Explain and apply building mcp servers.
- Explain and apply memory.
- Explain and apply context engineering basics.
- Explain and apply compaction, offloading and tool context.
- Explain and apply harness engineering.
- Explain and apply multi-agent patterns.
- Explain and apply building multi-agent systems.
- Explain and apply reliability and user experience.

## Contents

- Unit 5.7: MCP Fundamentals (Day 64)
- Unit 5.8: Building MCP Servers (Day 65–66)
- Unit 5.9: Memory (Day 67–68)
- Unit 5.10: Context Engineering Basics (Day 69)
- Unit 5.11: Compaction, Offloading and Tool Context (Day 70–71)
- Unit 5.12: Harness Engineering (Day 72)
- Unit 5.13: Multi-Agent Patterns (Day 73)
- Unit 5.14: Building Multi-Agent Systems (Day 74–75)
- Unit 5.15: Reliability and User Experience (Day 76)
## Setup and running example

Use Python 3.10+ and the companion notebook in this folder. The fictional Northstar support assistant searches a small policy collection and can propose a support request. All customer data, requests and receipts are synthetic. Offline cells use only Python unless explicitly marked optional. Notebook outputs are intentionally cleared. The optional OpenAI example requires `openai`, `python-dotenv`, an API key and model access; API calls cost money. Use a project-local ignored `.env`, never paste keys into cells. Set the paid flag only after reviewing the code and budget.

## Unit 5.7: MCP Fundamentals — Day 64

MCP standardizes a host/client/server connection. A host coordinates clients; servers expose tools, resources and prompts. Model tool calling is the model requesting an action; MCP is a protocol for discovering and invoking capabilities. A server can run locally or remotely, and transport, identity and trust boundaries matter. Loading a server tool does not grant it blanket permission.

### The three roles

The **host** is the application the user interacts with. It manages one or more **clients**, each communicating with a server. A **server** exposes capabilities, such as tools to execute actions, resources to read context, and prompt templates. For example, a local policy server could provide a `search_policy` tool and a `policy://about` resource. The model does not automatically speak to every server; the host decides what to expose and how to mediate calls.

Tool calling and MCP address different layers. A model may propose a tool call through the model API; the host can then execute a corresponding MCP tool. The protocol standardizes communication, while the executor still checks identity, authorization and inputs. A tool description is useful context for selection, but it may also carry malicious instructions.

### Connecting safely

Before connecting, identify server provenance, transport, readable files, credentials, outgoing network access and tool list. A local server process has the permissions of the account that launches it unless isolated; “local” does not mean safe. Test against a synthetic directory first and inspect the actual returned data. Changes to a tool schema or server version warrant review and regression tests.

**Try it:** Draw the boundaries for a local file server, list its readable directory, and explain which component enforces access.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Unit 5.8: Building MCP Servers — Day 65–66

Expose small typed operations, useful errors and bounded responses. Resources represent readable context and prompts are reusable templates; tools request operations. Avoid naming collisions across servers and version changes to names, schemas and semantics. Authenticate remote access and give each server only the credentials it needs. Review server code and tool descriptions before connecting.

### Designing a small server

For the Northstar exercise, expose `list_topics()` and `search_policy(query)` as read-only tools, plus `policy://about` as a resource describing the synthetic corpus. A tool should validate its input and return a stable shape such as `{status, results}`. Include a bounded number of passages, source IDs and a clear empty result. A server error should not leak credentials, stack traces or private file paths.

Develop the server separately from the agent. Start it using the transport required by your client, inspect the advertised capabilities, then call each tool directly with valid and invalid inputs. Only after that connect the agent and verify that its selected tool and arguments match the intended task. The companion notebook contains a manifest exercise, not a running MCP server.

### Versioning and remote access

A renamed tool or changed argument semantics can break clients even if the server still starts. Record a server version and a reviewed capability manifest; retest clients when it changes. Remote servers need authenticated, appropriately scoped access and transport security; a client must verify that it is talking to the intended server. Avoid exposing powerful generic operations when a narrow, task-specific tool suffices.

**Try it:** Build a synthetic server with two read-only tools and a resource. Inspect its capabilities, call them through an MCP client and record server/version and results.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Unit 5.9: Memory — Day 67–68

Short-term state belongs to a task or conversation; long-term memory survives runs. Distinguish preferences, observed events and procedural lessons, with provenance and timestamps. Select only relevant records from the authenticated user’s scope. Explicit corrections should replace or supersede stale facts; define expiration and deletion across stores, logs and backups. A summary or memory is untrusted data, not an instruction.

### Choose what to remember

Short-term state tracks the present task: current question, observations and pending approval. Long-term memory might contain a user's explicit language preference, a prior event or a reviewed lesson. These have different lifetimes and authority. An event that “a request was submitted” should include a receipt or source and timestamp; a preference should be editable by its owner. Do not turn unverified text into a permanent instruction.

A memory record can include `user_scope`, `kind`, `key`, `value`, `source`, `created_at`, `updated_at` and `expires_at`. Derive `user_scope` from authenticated context. When a newer explicit correction conflicts with an older preference, define how it supersedes the earlier value. For ambiguous conflicts, ask rather than silently choose.

### Retrieval and deletion

Retrieve the smallest relevant set by scope, relevance and recency. Evaluate both helpful recall and harmful recall: stale facts, unsupported inferences and another user's data. A delete operation must address the actual storage lifecycle, including derived indexes, logs, retention and backups as applicable; an in-memory dictionary deletion demonstrates only that dictionary. Maintain a clear policy for when backup copies expire and what deletion can be verified.

**Try it:** Store a fictional preference, recall it in another session, update it, test isolation against another user, then delete it. Explain what the in-memory demonstration cannot prove.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Unit 5.10: Context Engineering Basics — Day 69

The context budget is shared by instructions, task, history, tool schemas, observations and output. Measure actual tokens when available; characters are only a rough proxy. Inspect what was included and why at each step. Separate instructions from quoted evidence and preserve source IDs. Long context can increase latency and bury important constraints.

### Build a context ledger

| Component | Example | Selection rule |
| --- | --- | --- |
| Instructions | Assistant role and hard boundaries | Always include the current approved version |
| Current task | User question and constraints | Include the current request |
| Evidence | Policy passages with source IDs | Retrieve only authorized relevant passages |
| History | Recent turns and a summary | Retain decisions and unresolved questions |
| Tools | Schemas for relevant operations | Expose only what this step needs |
| Output reserve | Space for the model's response | Reserve before admitting more input |

The full context includes tool definitions and results, not only the visible chat. Track actual provider token usage when returned, and distinguish input tokens from output tokens. Character counts can illustrate growth but are not reliable token billing. Sensitive context should not be copied into debug logs without a retention and access policy.

### Diagnose drift

At several steps, inspect whether the original goal, exact constraints and latest observations remain visible. If a summary omits “do not create a request without approval,” the run may drift even though it has more room. Label evidence and memories as data and preserve provenance; putting untrusted text inside the context does not make it authoritative.

**Try it:** Create a context ledger for steps 1, 10 and 20: component, size, reason to include and removal rule. Identify a missing constraint and a redundant result.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Unit 5.11: Compaction, Offloading and Tool Context — Day 70–71

A useful rolling summary preserves goal, constraints, decisions, pending work and citations. Keep exact amounts and identifiers in structured state rather than relying on a free-text summary. Store large results separately with authorized references and return a bounded excerpt. Select the smallest relevant tool set; test that omitted tools do not prevent necessary recovery.

### What to compact and what to preserve

Suppose a long investigation has 20 search results. Keep a concise summary of what was checked and what remains, but preserve exact request IDs, amounts, permissions, citations and approval status in structured state. A free-text summary can round a number or omit a negation. Store large documents outside the model context and retrieve authorized excerpts by stable reference.

A rolling summary can be updated after a few steps: previous summary + newly completed work → proposed new summary → validation against mandatory fields. Test that the new summary preserves the task goal, unresolved items and relevant source IDs. Retain the original trace in a suitably protected store if audit or debugging requires it; a summary is not a complete audit log.

### Compare strategies

Run the same long tasks using full history, summary only, and summary plus structured facts and bounded tool outputs. Record completion, exact-fact errors, tool calls, actual token usage and latency. If only one run is available, treat the result as an observation rather than a general conclusion. Context reduction is useful only when it preserves enough information to finish correctly.

**Try it:** Compare full history, rolling summary, and summary plus structured facts on the same long task. Measure completeness, calls, tokens and latency; check exact IDs.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Unit 5.12: Harness Engineering — Day 72

The harness includes orchestration, instructions, tools, policies, context, persistence, budget, checks and observability around the model. Change one component at a time and compare the same versioned task set and model. A tool description can change behavior, but code remains the enforcement point. Failures should produce actionable, sanitized feedback.

### Separate model behavior from harness behavior

The model supplies decisions; the harness supplies the environment in which decisions are made. A failure such as repeatedly searching may come from an ambiguous tool description, a missing tool result, an overly short context, an absent stop rule or the model itself. A trace should let you distinguish these possibilities before changing the prompt.

| Harness change | Hypothesis | Measurement |
| --- | --- | --- |
| Clarify search description | Fewer wrong-tool selections | Tool selection and task success |
| Validate required fields | Fewer malformed writes | Rejected invalid calls and successful outcomes |
| Load task-specific instructions | Better context with fewer irrelevant tokens | Accuracy and input usage |

Change one element per experiment and keep task set, model version and scoring rules fixed. Use separate development and held-out cases; read failures instead of relying only on a mean score. Pin or record all relevant versions, including tool implementation, permissions and corpus, because any of them can alter the result.

**Try it:** Compare three separate changes—tool descriptions, a deterministic check, and task-specific instructions—against one baseline and a held-out set.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Unit 5.13: Multi-Agent Patterns — Day 73

A supervisor can route work to specialists; handoffs transfer a task; parallel workers investigate independent subtasks. These add coordination, conflicting conclusions, latency and cost. Define the contract, evidence and permission scope for each worker and compare with a single-agent baseline. A specialist’s answer is a claim to verify, not an authority.

### When specialization helps

Specialists can have distinct tools, instructions, models or access scopes. In a support example, one worker finds policy passages while another checks an order in a separate authorized system. A supervisor combines their structured results and reports uncertainty. This can reduce irrelevant context or allow narrower permissions, but each handoff can lose constraints or evidence.

A sequential handoff passes work from one role to another. Parallel workers handle independent tasks at the same time; the coordinator must wait for required results and handle partial failure. A supervisor pattern centralizes delegation and merging. None guarantees better accuracy: compare the same tasks with one well-equipped agent and account for additional calls and coordination.

### Contracts between workers

Include task, permitted scope, required output, deadline and necessary context. A result should include `status`, findings, source references and uncertainty. If workers disagree about a policy, present the conflicting evidence or escalate; do not resolve the conflict by selecting whichever answer sounds more confident.

**Try it:** Sketch a single support agent and a supervisor with policy and order specialists. Predict duplicated calls and a conflict-resolution rule.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Unit 5.14: Building Multi-Agent Systems — Day 74–75

A supervisor assigns bounded tasks and receives structured results including evidence, uncertainty and status. Keep private state scoped; share only necessary facts. Define timeouts, failed-worker fallback and merge rules. Aggregate per-worker usage and avoid claiming success if a required specialist failed. A2A is a distinct interoperability effort; check its current specifications before integrating.

### A supervisor workflow

1. Receive the task and derive permissions from authenticated context.
2. Divide independent work into bounded assignments; include only the information each specialist needs.
3. Dispatch with per-worker deadlines and tool budgets.
4. Collect a structured result or explicit failure from each worker.
5. Validate required evidence, reconcile conflicts and return a complete or partial outcome.

For example, the policy worker might return `{status: "ok", sources: ["leave-v2"], finding: "10 days"}` while the order worker returns `{status: "unavailable"}`. The supervisor may answer the policy portion but must not claim it checked the order. If a failure is transient, retry only within the shared budget; otherwise ask the user or escalate.

### Measure the trade-off

Record per-worker and total calls, tokens, latency and failures. Parallel latency can improve even while total token use increases. Use the same ten or more representative tasks for a single-agent baseline and the multi-agent version, including one specialist outage and one conflicting result. The notebook's deterministic router models the control flow; it is not evidence about real specialist model performance.

**Try it:** Implement a deterministic router with two toy specialists, simulate one failure and compare the outcome and call count with the single-agent baseline.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Unit 5.15: Reliability and User Experience — Day 76

Budget model turns, tool calls, elapsed time and spend independently. Handle timeouts, invalid inputs and unavailable services; retry only transient operations with bounded backoff and idempotency where appropriate. Show meaningful progress, cancellation and partial results without exposing private traces. A budget estimate is not a hard spending cap unless enforced at the actual billing boundary.

### Reliability is a contract with the user

Expose states such as `searching`, `waiting_for_approval`, `completed`, `incomplete` and `cancelled`. Progress should describe observable work rather than imply a successful action that has not received a receipt. A user can cancel a future action, but a write already accepted by an external provider may require separate reconciliation or reversal.

Set independent limits for turns, tool calls, retries and elapsed time. A cost budget needs actual usage information and an enforcement mechanism; an estimate checked after a call may already be exceeded. Configure timeouts on individual network operations, not just a timer around the loop. For an unreliable search, distinguish “no results” from “could not search.”

### Failure drills

Inject a timeout, invalid argument, repeated call, partial specialist response and cancellation just before a write. For each, note the user-visible message, tool state and trace. Check that retries are bounded and that no new action begins after cancellation is processed. Report unresolved remote writes as unknown until reconciled.

**Try it:** Inject a timeout and a repeated tool call. Record the visible progress, final incomplete status and whether cancel prevents a new action.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Capstone and assessment

Extend the fictional support assistant using the concepts in this part. Define five normal cases, two missing-information cases and three failure or adversarial cases before implementation. For each, record inputs, expected behavior, actual trace, outcome and evidence. Compare against a simple deterministic baseline. Explain what the offline demonstration establishes and which real integrations remain untested.

**Deliverables:** architecture sketch, tool/state contracts, runnable experiment, results table, failure analysis and two proposed improvements. Include actual call counts and latency where measured; mark token usage or cost unknown when unavailable.

### Part 2 project: context-aware support investigation

**Goal:** Design a supervisor that can delegate a policy lookup and an order-status lookup to separately scoped specialists. Use synthetic data and start with the notebook's deterministic router. The order system is a fake until you explicitly implement and test an integration.

1. Define a handoff contract with task, permitted scope, deadline and expected result. Specify which state is shared and which stays private.
2. Create a memory record schema with owner, source, timestamps and expiry. Demonstrate correction and per-user isolation in the fixture; explain the additional storage enforcement needed in production.
3. Build a context ledger and a compaction rule that retains exact IDs, constraints, pending work and evidence. Check these fields before and after compaction.
4. Implement or sketch a local synthetic MCP server with two read-only tools and one resource. If you run it, record server version, client, transport and direct tool results; otherwise mark integration pending.
5. Compare single-agent and specialist approaches on ten tasks, including a worker timeout, conflicting evidence and a case needing clarification. For each version record task outcome, per-worker calls, latency and available usage.

**Acceptance:** Required worker failure yields a partial or incomplete result; a conflict is surfaced; another user's memory is never intentionally supplied; summaries preserve exact required facts. State clearly which checks were simulations and which used real integrations.

### Self-check

1. Which decisions belong to code, and which can be delegated to a model?
2. What is the evidence for success, and what happens if it is missing?
3. What assumptions and trust boundaries did the experiment leave untested?
4. What would change when moving from in-memory fixtures to a durable service?

### Completion checklist

- [ ] Read every unit and complete the paired exercises.
- [ ] Run the notebook offline and inspect its assertions.
- [ ] Record limitations, failed cases and measured results.
- [ ] Run optional paid or integration exercises only when ready and record their actual outcomes.

## Further reading

- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling) and [Responses API](https://developers.openai.com/api/reference/responses/overview).
- [LangGraph documentation: interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) and [persistence](https://docs.langchain.com/oss/python/langgraph/persistence).
- [MCP: build a server](https://modelcontextprotocol.io/docs/develop/build-server) and [security best practices](https://modelcontextprotocol.io/specification/draft/basic/security_best_practices).
