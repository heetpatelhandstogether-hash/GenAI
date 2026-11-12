# Module 5, Part 3: Evaluating Agents — Detailed Learning Document

**Days 77–79 | Units 5.16–5.18**  
Prerequisite: Module 4 and the preceding parts of Module 5. Companion: [practical notebook](Module-5-Part-3-OpenAI-Practical-Experiments.ipynb).

## Learning outcomes

- Explain and apply why agent evals are different.
- Explain and apply evaluating tool use and task success.
- Explain and apply testing recovery, memory and efficiency.

## Contents

- Unit 5.16: Why Agent Evals Are Different (Day 77)
- Unit 5.17: Evaluating Tool Use and Task Success (Day 78)
- Unit 5.18: Testing Recovery, Memory and Efficiency (Day 79)
## Setup and running example

Use Python 3.10+ and the companion notebook in this folder. The fictional Northstar support assistant searches a small policy collection and can propose a support request. All customer data, requests and receipts are synthetic. Offline cells use only Python unless explicitly marked optional. Notebook outputs are intentionally cleared. The optional OpenAI example requires `openai`, `python-dotenv`, an API key and model access; API calls cost money. Use a project-local ignored `.env`, never paste keys into cells. Set the paid flag only after reviewing the code and budget.

## Unit 5.16: Why Agent Evals Are Different — Day 77

A correct final answer can follow an unsafe or wasteful trajectory. Capture a structured trace with task ID, version, action, sanitized arguments, outcome, timing and termination reason. Review traces for wrong tools, repeated calls, early stops and missing evidence. Do not log secrets or sensitive raw data unnecessarily.

### Why the trajectory matters

Suppose an agent answers “10 days” correctly but first attempts to read another employee's private record. A final-answer-only score would miss the attempted boundary violation. Conversely, an agent might select the right tool and handle its outage safely but be unable to complete the task. Score the outcome and the path separately.

A useful trace captures a task ID, model and prompt version, tool name, sanitized arguments, timestamps, result status, approval events and termination reason. Preserve enough detail to reproduce a failure while limiting sensitive data. Provider traces and local application traces may show different parts of the run; connect them using a correlation ID when permitted.

### Read before automating

Review 20 traces if available and label each failure: wrong selection, malformed arguments, unnecessary calls, unsafe attempt, poor recovery, lost context or premature completion. One trace may have multiple labels. Record whether data is synthetic, a local model run or a production event; do not present one category as evidence for another. Use the observed failures to design evaluation cases.

**Try it:** Review 20 real or explicitly synthetic traces. Label failure categories and distinguish attempted from completed actions.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Unit 5.17: Evaluating Tool Use and Task Success — Day 78

Create a versioned set of tasks with expected outcomes, allowed tools, constraints and acceptable alternative paths. Score tool selection, argument validity, action order, outcome and forbidden actions separately. Task success rate = successful eligible tasks / eligible tasks; report exclusions and missing runs. Exact path matching may penalize legitimate alternatives.

### Build a case with an oracle

A case needs an ID, initial state, user task, allowed operations, test environment, success criteria and evidence needed to judge completion. Some tasks allow several correct paths; specify required and forbidden outcomes rather than one exact sequence unless order itself is essential. Include no-tool questions, lookup tasks, missing information, invalid input and forbidden actions.

| Dimension | Example rule | Limitation |
| --- | --- | --- |
| Tool choice | Search is allowed; delete is forbidden | Allowed does not mean necessary |
| Arguments | Query is a nonempty string within length limit | Valid shape does not prove authorization |
| Outcome | Answer is supported by a specified policy | May need human semantic review |
| Safety | No unauthorized write completed | Also count attempts |
| Efficiency | Calls and elapsed time within budget | Low cost alone is not success |

### Calculate and compare

`task success rate = successful eligible cases / eligible cases`. Report total cases, exclusions, missing runs and the eligibility rule. `unsafe action rate = cases with a defined unsafe action / valid security cases`; separately report attempts and completed actions. Do not drop missing or failed cases to improve a score. A deterministic check works well for IDs and schemas; a model judge for open-ended quality needs a rubric, human calibration and disagreement review.

**Try it:** Build 15 cases including no-tool, multi-step, missing information and forbidden action. Independently score five traces and compare disagreements.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Unit 5.18: Testing Recovery, Memory and Efficiency — Day 79

Inject failures at named points: timeout, explicit error, empty results and malformed data. Score whether the agent recognized the failure, recovered safely or reported uncertainty; a retry alone is not recovery. Test memory updates, deletion and cross-user isolation. Record calls, steps, latency and tokens; calculate cost only from known usage and applicable prices.

### Fault injection matrix

| Injection | What to observe | Safe outcome |
| --- | --- | --- |
| Search timeout | Error recognized and retry count bounded | Fallback or explicit unavailable status |
| Empty search | Empty result distinguished from outage | No invented citation; clarify or abstain |
| Malformed result | Input validation at boundary | Sanitized error and no unsafe action |
| Stale memory | Newer evidence considered | Correct or ask about conflict |
| Cross-user request | Server derives authenticated scope | Other user's memory is inaccessible |
| Provider write timeout | Outcome may be ambiguous | Reconcile before retry |

Score recovery by final behavior and evidence, not simply the presence of a retry. Count calls, retry delays, time to outcome and any duplicate actions. Run the same injected faults on both baseline and candidate so that differences in environment do not masquerade as improvements.

### Interpreting usage

Collect input and output tokens from each response when the provider reports them. For a task with several agents or retries, sum all attributable calls. Estimate cost only using the applicable model, rates, units and date; cached or other usage categories may have different rates. Missing usage means cost is unknown. Report median and a tail percentile for latency only when the sample is large enough to make them meaningful, along with sample size.

**Try it:** Run three search failures, two empty searches and three memory cases. Compare baseline and candidate on identical cases and report missing or unknown measurements.

**Check:** Explain your decision and give one failure case, one measurable result and the evidence supporting it.

## Capstone and assessment

Extend the fictional support assistant using the concepts in this part. Define five normal cases, two missing-information cases and three failure or adversarial cases before implementation. For each, record inputs, expected behavior, actual trace, outcome and evidence. Compare against a simple deterministic baseline. Explain what the offline demonstration establishes and which real integrations remain untested.

**Deliverables:** architecture sketch, tool/state contracts, runnable experiment, results table, failure analysis and two proposed improvements. Include actual call counts and latency where measured; mark token usage or cost unknown when unavailable.

### Part 3 project: agent evaluation report

**Goal:** Evaluate two versions against a versioned set of at least 15 tasks with normal, edge, recovery and security cases. The candidate may be a change in tool descriptions, routing or a validation rule; change one variable at a time.

1. Define each case's initial state, success criteria, allowed and forbidden actions, and acceptable alternatives before running either version. Assign stable IDs.
2. Inject three unavailable-search cases, two empty-result cases and three memory cases, including a stale fact and cross-user isolation. Record whether each fault actually occurred.
3. Capture sanitized traces and score task outcome, tool choice, argument correctness, unsafe attempts, completed unsafe actions, recovery and efficiency separately. Independently review a sample and resolve scoring disagreements.
4. Reject or explicitly report missing, duplicate and unexpected runs. Report numerator and denominator for every rate, plus excluded cases and reasons.
5. Compare matched task results, inspect regressions, and preserve failing cases for the next run. Record model, prompts, tool, corpus and evaluator versions.

**Acceptance:** The report includes individual results and evidence, not just an average. It labels synthetic traces, actual model runs and external integrations separately. Unknown token use or cost remains unknown. The notebook's simple scorer is a starting point, not a comprehensive evaluator.

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
