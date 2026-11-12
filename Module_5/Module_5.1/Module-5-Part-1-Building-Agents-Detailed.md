# Module 5, Part 1: Build an Agent — Detailed Learning Document

**Days 57–63 | Units 5.1–5.6**  
Prerequisite: Module 4. Companion: [practical notebook](Module-5-Part-1-OpenAI-Practical-Experiments.ipynb).

## Learning outcomes

- Explain when an agent is useful and when a fixed workflow is preferable.
- Explain and apply the agent loop.
- Build and inspect a bounded graph using LangGraph.
- Design and validate safe tool contracts.
- Explain and apply planning and reflection.
- Explain checkpoints, approval, idempotency and recovery.

## Contents

- Unit 5.1: What Is an Agent? (Day 57)
- Unit 5.2: The Agent Loop (Day 58)
- Unit 5.3: Building Agents with LangGraph (Day 59–60)
- Unit 5.4: Tools for Agents (Day 61)
- Unit 5.5: Planning and Reflection (Day 62)
- Unit 5.6: Saving Progress and Asking for Approval (Day 63)
## Setup and running example

Use Python 3.10+ and the companion notebook in this folder. The fictional Northstar support assistant searches a small policy collection and can propose a support request. All customer data, requests and receipts are synthetic. Offline cells use only Python unless explicitly marked optional. Notebook outputs are intentionally cleared. The optional OpenAI example requires `openai`, `python-dotenv`, an API key and model access; API calls cost money. Use a project-local ignored `.env`, never paste keys into cells. Set the paid flag only after reviewing the code and budget.

## Unit 5.1: What Is an Agent? — Day 57

An agent is a bounded decision loop: it receives a goal, chooses an action, observes the result and decides what to do next. A model proposes actions; application code owns execution, authorization and termination. State includes the task, observations, budget and outcome. A chain has predetermined steps; an agent selects among possible steps at runtime.

Use a fixed workflow for a predictable FAQ lookup or a known form validation sequence. Consider an agent for an investigation where the next lookup depends on earlier results. For each candidate write the expected benefit, failure cost, latency budget and simpler baseline. A model choosing a tool does not make the tool safe.

### What changes when a model chooses the next step?

Consider: “Find the leave policy and, if it does not answer my situation, prepare a support request.” The required second step depends on the first result. An agent may decide to search, inspect a passage, ask a clarifying question or propose a request. Its freedom is limited to the tools, inputs and budget the application permits. The goal is the desired result; the policy is the set of boundaries; neither is a substitute for the other.

| Component | Northstar example | Owner |
| --- | --- | --- |
| Goal | Answer a leave question using policy evidence | User/application |
| Model | Suggest next action or answer | Provider/model |
| Tools | Search policy; prepare a support request | Application |
| State | Question, observations, pending decision, budget | Application |
| Stop rules | Final answer, refusal, error, or budget reached | Application |

### Chain, workflow or agent?

A fixed chain is appropriate when the path is known: validate input → calculate a fee → format output. A conditional workflow adds known branches: if no order exists, show an error; otherwise fetch its status. An agent is useful when a bounded set of possible steps exists but the sequence depends on observations. These are design choices, not levels of sophistication. A workflow can invoke a model for one step and still be deterministic in its routing.

Start with the simplest design that meets the task. Compare completion rate, unsupported claims, tool calls, latency and cost on the same cases. If the agent produces no meaningful improvement, its added failure modes and complexity may not be justified.

**Try it:** Classify these: calculate tax from a known formula; find why an order is delayed using several systems; summarize one document; investigate a failed deployment; send a fixed reminder. Justify each and name one alternative.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Unit 5.2: The Agent Loop — Day 58

A loop sends the goal and available tools, receives zero or more tool requests, validates and executes allowed requests, and sends each result back with its call ID. A final answer is a separate terminal state. Observation is what the tool actually returned, not what the model hoped happened.

Set separate limits on model turns, total tool calls, elapsed time and repeated identical actions. If a limit is reached return an explicit incomplete outcome, not a fabricated success. A deadline checked between calls does not cancel a hanging call: set client/tool timeouts too. A plan-first agent maintains a revisable checklist; an incremental agent decides one step at a time. Compare both on the same tasks rather than assuming planning always helps.

### A complete turn

1. Give the model the task, allowed tool definitions and relevant state.
2. The model returns a final message or one or more proposed tool calls with names, arguments and call IDs.
3. Your executor checks each name, argument, permission and remaining budget. It runs only allowed calls.
4. It returns each result or sanitized error against the corresponding call ID. The model observes the results and decides again.
5. The application records a final answer or an explicit incomplete, rejected or failed outcome.

Example trace: user asks about leave → `search_policy({"query":"planned leave"})` → `[leave] Request 10 days ahead` → answer with `[leave]`. If search fails, the observed fact is “search unavailable”; it is not evidence that no policy exists. Tell the user the lookup failed or try an allowed fallback.

### Limits and repeated actions

A model-turn limit bounds requests to the model; a tool-call limit bounds actions even when one response contains several calls. A wall-clock deadline and per-call timeout address slow services. For repeated calls compare normalized `(tool, arguments)` plus recent results; after a small threshold, stop or ask for help. A repeated call might occasionally be legitimate after data changes, so make the rule task-specific and record why it fired.

Use a structured termination reason such as `completed`, `needs_information`, `tool_unavailable`, `budget_exhausted` or `cancelled`. Cancellation requires preventing additional actions and attempting to cancel in-flight work where supported; it cannot guarantee that a remote action already accepted by a provider was undone.

**Try it:** Trace an agent calling search, order lookup and a calculator. Then simulate an unavailable tool and a repeated request; identify the exact stopping condition.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Unit 5.3: Building Agents with LangGraph — Day 59–60

A graph makes control flow explicit. State is the typed data passed between nodes; a node returns updates; a conditional edge selects the next node. A support graph can route a simple question to an answer node and a lookup request to a search node. Inspect intermediate states to see which branch ran.

A graph is not automatically an agent: a graph with fixed branches can be a deterministic workflow. For a tool-using graph, include a model node, a guarded tool node and a conditional edge back to the model, with a budget edge to END. State reducers and concurrent updates need deliberate handling. See the optional runnable graph in the notebook.

### State, nodes and edges in a concrete graph

A useful state design includes `question`, `messages`, `tool_results`, `step_count`, `status` and `answer`. Decide which fields a node can update and which are trusted. The authenticated identity and permissions should come from application context, not a model-generated state field. A node reads state and returns updates; an edge directs control; a conditional edge chooses among routes.

```text
START → model → [final?] → END
                 ├─ tool requested → validate → tool → model
                 └─ budget exhausted → incomplete → END
```

This is a design sketch. The companion notebook first demonstrates a smaller deterministic branch so you can see state and routing without a model or paid call. To turn it into a tool-using loop, implement the model node and validated tool node, then route back only while the budget remains. Test final answers, invalid calls, unavailable tools and exhaustion independently.

### State updates and inspection

When multiple nodes update state, define how values are merged. Replacing a message list can accidentally discard observations; appending blindly can duplicate them on a replay. Reducers and checkpoint behavior must match the data and execution pattern. Inspect the state and node path after each run, and avoid putting secrets or unlimited tool output in checkpoints. Use a unique thread ID per run or conversation and an appropriate persistent checkpointer when durability is required.

**Try it:** Draw the graph with START, route, answer, search and END. Add a maximum-step branch and explain which state is trusted.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Unit 5.4: Tools for Agents — Day 61

A tool contract specifies a unique name, when it is useful, a typed input schema, output shape and error semantics. Search should return short records with source IDs; a request-creation tool is a separate, state-changing operation. Schema-valid inputs may still be unauthorized, out of scope or factually wrong.

Wrap the Module 4 retriever behind a narrow search interface: derive the user and document access from trusted server context, filter before returning results and preserve citations. Never accept a model-supplied tenant as authorization. Return an explicit empty result or a sanitized error, and cap returned text. A toy keyword search in the notebook is not production RAG.

### Design a usable contract

A search tool needs a clear description: “Search approved Northstar policy excerpts for a question; returns source IDs and short passages.” Its input can be `query: string`; its output can contain `status`, `results` and `source_id`. Define whether empty results mean a successful search with no matches, and distinguish that from a timeout or permission denial. A malformed request should not be silently interpreted as a different operation.

A write tool has additional requirements: authenticated actor, target, validated content, approval where needed, operation ID and a receipt. Keep authorization context outside model-controlled arguments. For example, a generated `tenant_id` is a request, not proof that the user may access that tenant.

| Scenario | Tool response | Agent behavior |
| --- | --- | --- |
| Search succeeds, no matches | `ok` with empty results | Say evidence was not found; ask or escalate |
| Search times out | `unavailable` | Bounded retry or report failure |
| Access denied | `forbidden` | Do not retry as another user |
| Write accepted | Receipt with operation ID | Report only the confirmed action |
| Write outcome unknown | `unknown` | Reconcile before any retry |

### Connect Module 4 RAG

Use the Module 4 retriever as an implementation behind `search_policy`. Apply access filters before passages reach the model, retain source/version identifiers and limit result size. Evaluate retrieval separately from the agent's decision to search. The notebook's substring search is only a small fixture: it does not establish retrieval quality, tenant isolation or citation support.

**Try it:** Write contracts for search_policy(query) and create_request(subject). Test empty search, invalid arguments, forbidden access and duplicate requests.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Unit 5.5: Planning and Reflection — Day 62

A plan is a short sequence of observable subgoals, not proof that each step occurred. After an observation, mark completed work using receipts and revise the remaining steps when assumptions fail. A review step can compare the proposed answer against sources or a schema, but self-review by the same model can repeat the original mistake.

Prefer deterministic validation for IDs, totals and required fields; use human review or a calibrated judge for subjective quality. Measure the extra calls, latency and token use of planning and reflection. Avoid logging hidden reasoning; record concise decisions, actions and evidence instead.

### Plans as state, not promises

For a request involving a policy question and a possible support request, a plan might be: (1) find the policy, (2) decide whether evidence answers the question, (3) ask for missing details if needed, (4) propose a support request for approval. Store a short status for each step and cite the observation or receipt that completed it. If no passage is found, revise the plan rather than treating step 1 as successful merely because a search was attempted.

### Reflection with an external check

A useful review asks a concrete question: “Does every policy claim have a retrieved source?” A simple program can check whether cited IDs occur in the retrieved set, but cannot establish that a passage actually supports the claim. Reviewers or a calibrated judge must assess semantic support. A second model call is not independent ground truth and can confidently approve the same error.

For a fair experiment use the same tasks and model settings for a no-plan baseline and a plan-and-check candidate. Record `task_id`, variant, correct outcome, supported claims, model calls, tool calls, latency, token usage and known cost. Repeat where behavior varies, inspect failures, and reserve unseen tasks for a final comparison. Report missing usage rather than inventing a cost.

**Try it:** Run five multi-part cases with and without an explicit plan. Record success, unsupported claims, calls, time and tokens; include a case with missing evidence.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Unit 5.6: Saving Progress and Asking for Approval — Day 63

Persist the task, versioned state, next step, request ID and approval status. A checkpoint lets a run resume, but does not by itself make an external write exactly once. A crash after a remote write and before recording its receipt is ambiguous; reconcile with the provider using an operation ID before retrying.

For consequential actions show the exact destination and contents, obtain an authenticated approval bound to that action and expiry, then revalidate authorization before execution. A changed action requires a fresh approval. Rejection is a terminal decision, not an implicit approval. LangGraph interrupts resume nodes from their beginning, so avoid non-idempotent side effects before the interrupt. In-memory checkpoints are demonstrations only and disappear on process exit.

### Checkpoint versus action receipt

A checkpoint says what the workflow saved at a point in time. A receipt says what the external system accepted or completed. These are different facts. Consider a crash after the provider creates request `R-42` but before the application stores the receipt. Restarting from the last checkpoint may repeat the action. Use a stable operation ID, provider-supported idempotency where available, and a lookup/reconciliation procedure for ambiguous outcomes.

A practical state machine is `draft → awaiting_approval → approved → submitting → confirmed`, with separate `rejected`, `expired` and `outcome_unknown` states. Do not tell the user a request was created in `submitting` or `outcome_unknown`. Resolve an unknown outcome against the provider before deciding whether it is safe to retry.

### What a person approves

Display the intended action, destination, exact content and meaningful consequences. Bind the decision to an authenticated actor and a digest or version of that action; set an expiry. If the content or target changes, request new approval. On resume, re-check current permissions and approval validity. A demonstration approval flag is useful for learning but does not authenticate a person or provide durable evidence.

LangGraph interrupts can rerun the interrupted node when resumed. Put non-idempotent side effects after the interrupt, preferably in a separate node, and design for replay. A persistent checkpointer and durable operation records are still required for a real deployment.

**Try it:** Simulate approve, reject, replay and a crash after the write. Show why a durable idempotency key and receipt reconciliation are necessary.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Capstone and assessment

Extend the fictional support assistant using the concepts in this part. Define five normal cases, two missing-information cases and three failure or adversarial cases before implementation. For each, record inputs, expected behavior, actual trace, outcome and evidence. Compare against a simple deterministic baseline. Explain what the offline demonstration establishes and which real integrations remain untested.

**Deliverables:** architecture sketch, tool/state contracts, runnable experiment, results table, failure analysis and two proposed improvements. Include actual call counts and latency where measured; mark token usage or cost unknown when unavailable.

### Part 1 project: policy assistant with an approval boundary

**Goal:** Build a small assistant that answers questions from fictional policy passages and can prepare—but only after approval, submit—a synthetic support request. Begin with a deterministic search-and-answer baseline. Use the notebook's search fixture first; add a real model only after the offline behavior is understood.

**Build in stages:**

1. Write ten test cases: an answerable policy question, an unanswerable question, a no-tool greeting, a request needing clarification, an unavailable search, an invalid tool argument, a repeated call, an approval rejection, an approval acceptance and a duplicate operation.
2. Define the state, tool schemas and stop reasons. Trace one search → observation → answer path by hand. Specify maximum model turns, tool calls and timeouts.
3. Add a model-driven tool loop or a LangGraph implementation. Validate each proposed call outside the model, preserve call IDs, and make a missing or failed search explicit.
4. Implement the approval and receipt flow against an in-memory fake. Simulate a crash after acceptance and explain what must change for a real provider with durable state.
5. Compare baseline and agent on the same cases. Record supported answer, action outcome, number of calls, elapsed time, usage and any unknown values.

**Acceptance:** All ten cases have an observed outcome; no rejected request is created; a duplicate operation does not create another synthetic receipt; exhaustion is reported as incomplete; citations point to returned sources. Manually inspect whether each cited source actually supports its claim. The in-memory test does not establish durable recovery or real provider behavior.

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
