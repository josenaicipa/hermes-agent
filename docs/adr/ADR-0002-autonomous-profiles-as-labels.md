# ADR-0002: Autonomous profiles are labels; cron agentic runs are uncapped

- **Status:** Accepted
- **Date:** 2026-07-31
- **Scope:** Hermes profile `vpsclone` (Hermes V2.9)
- **Owners:** Jose / Hermes architecture
- **Supersedes:** [ADR-0001](ADR-0001-autonomous-execution-limit-contract.md)
- **Decision type:** Execution model for scheduled autonomous work

## Context

ADR-0001 gave every autonomous cron run a fixed positive envelope (hard
timeout, max turns, processed-token budget, metered USD budget) plus an
inactivity watchdog, and converted any threshold hit into a failed run with a
canonical `limit_reason`.

In practice the envelopes did not bound *risk*, they bounded *completion*.
The scheduler's own clock, turn counter and token/spend totals are not
evidence that a run stopped making progress, and killing on them truncated
legitimate long work: an `implementation` job doing real multi-file edits hit
`max-turns` mid-change, and the "fallback summary is not a success" rule then
recorded a failure for work that was progressing normally. The inactivity
watchdog had the same defect — a slow provider or a long-running tool call is
indistinguishable from a hang at the scheduler layer. The one-shot re-scope
guard compounded it by permanently stopping a task after a second such hit.

Meanwhile the limits that actually protect the system are not the scheduler's:
provider-native rate limits, real model context limits, compression,
admission/permission checks, and the operator's ability to cancel a run.

## Decision

`autonomous_profile` remains a **fail-closed validated routing/audit label**
and nothing more. Allowed values:

`light`, `standard`, `implementation`, `large`, `high-risk-review`, `retry`,
`experimental`

Resolution precedence is unchanged: `job['autonomous_profile']` →
`config.cron.autonomous_limits.default_profile` → `standard`. Unknown, empty,
numeric, and `0`/`unlimited`-style sentinel values are rejected *before* an
agent is constructed, so a typo can never silently pick a default. Every label
persisted under ADR-0001 is still valid, so existing jobs need no migration.

A label carries **no** resource envelope. `AutonomousProfile` has exactly one
field (`name`); a cap attribute there would reintroduce what this ADR removes.

Consequently, an agentic cron run is **not** stopped by:

- scheduler wall-clock duration;
- API-call / turn count;
- processed or cache token totals;
- estimated USD spend;
- inactivity.

The wall-clock/turns/token/USD evaluator, the inactivity watchdog and the
one-shot re-scope guard are removed. `cron.autonomous_limits` now supervises
*plumbing only* (`run_supervised_agentic_run`): a worker Future for teardown,
the caller's ContextVars, `task_id` propagation, and the `run_claim`
heartbeat. It never interrupts the agent.

### True-unlimited iterations

Cron selects its iteration cap explicitly via
`cron.autonomous_limits.resolve_cron_iteration_cap()`, which returns
`agent.iteration_budget.UNLIMITED_ITERATIONS`. Cron does **not** inherit
`agent.max_turns` and does **not** fall back to the 500 default.

Unlimited is a dedicated sentinel object — deliberately **not** `0` (which
historically meant "exhausted") and **not** a large integer (still a ceiling,
and it silently truncates long runs). It orders above every real iteration
count and is closed under `+`/`-`, so the existing bounded comparisons keep
their meaning. `IterationBudget`, the conversation loop and the turn finalizer
branch on it through `is_unlimited` / `cap_reached` / `iterations_available` /
`iterations_exhausted`; persisted and logged copies go through
`iteration_cap_repr`, which renders it as `"unlimited"` so JSON stays valid.

### Preserved unchanged

Manual cancellation and client disconnect, the worker Future, ContextVars and
`task_id`, the `run_claim` heartbeat, singleflight/claim semantics, teardown
and cleanup ordering, permissions/admission, provider-native rate limits, real
model context limits, compression, and script-only `no_agent` behavior
(which never spawns an agent and is unaffected).

`HERMES_CRON_TIMEOUT` no longer bounds any run. It survives only as the
operator-tunable quiet-period hint used to derive the one-shot run-claim
dead-owner TTL; a live run is protected by the heartbeat regardless.

## Consequences

### Positive

- Long, legitimately slow work completes instead of being truncated and
  misreported as a failure.
- One less class of false failure: no more "fallback summary after max-turns".
- The label keeps its useful half — routing, reporting and the
  agentic-efficiency ledger still classify runs by intent.
- Fail-closed label validation before agent spawn is retained in full.

### Trade-offs

- A genuinely wedged agentic cron run is no longer auto-killed by the
  scheduler; it must be cancelled by the operator (or ended by the provider).
  This is the deliberate trade: the scheduler could not tell "wedged" from
  "slow", and guessing wrong destroyed real work.
- Cost is now bounded by provider-side limits and job design, not by a local
  USD ceiling. Spend attribution stays observable through the existing
  agentic-efficiency ledger.
- `AutonomousLimitError` and its canonical reasons remain defined for
  administrative/manual stops, but no resource threshold raises them.

## Alternatives rejected

1. **Keep the caps, raise the numbers:** rejected — it moves the truncation
   point without removing the failure mode, and every raise is a new guess.
2. **Keep inactivity only:** rejected — inactivity is precisely the signal the
   scheduler cannot interpret (slow provider vs. hang), and it caused the same
   truncation.
3. **Represent unlimited as `0` or a huge integer:** rejected — `0` already
   means "exhausted" on this code path, and a huge integer is still a ceiling
   that fails silently and late.
4. **Drop `autonomous_profile` entirely:** rejected — the label still has real
   value for routing, reporting and ledger classification, and dropping it
   would force a migration of every existing job.

## Verification

- `tests/cron/test_autonomous_profile_labels.py` — label set (including
  `experimental`), precedence, fail-closed validation, "label exposes no cap
  field", non-stopping behavior for wall-clock/turns/tokens/USD/inactivity,
  preserved plumbing (worker Future, ContextVars, `task_id`, heartbeat), and
  the scheduler selecting explicit unlimited iterations. Supersedes the
  removed `tests/cron/test_autonomous_limits.py`.
- `tests/agent/test_unlimited_iterations.py` — sentinel identity/ordering
  /arithmetic, JSON-safe representation, unlimited and bounded
  `IterationBudget` semantics, and the loop/finalizer predicates.
