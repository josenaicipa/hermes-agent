# ADR-0001: Positive limits for autonomous model execution

- **Status:** Accepted
- **Date:** 2026-07-23
- **Scope:** Hermes profile `vpsclone`
- **Owners:** Jose / Hermes architecture
- **Decision type:** Operational safety and cost containment

## Context

Autonomous crons, controllers, workers, loops, and retries can continue consuming time, turns, tokens, or metered spend after useful progress has stopped. Provider-side limits are inconsistent and do not replace local process control. An unlimited `0/0` sentinel is especially unsafe when no operator is present. Previous routes also allowed a limit-triggered split to become another unbounded retry loop.

The contract must cover every model-capable autonomous entry point while preserving deterministic `no_agent` jobs. It must fail closed before model spawn when the execution profile is missing or invalid, kill the entire child process group, preserve a canonical terminal reason, and produce enough evidence to distinguish timeout, turn exhaustion, and budget exhaustion.

## Decision

Every autonomous model execution MUST use one fixed positive profile and pass through an approved supervisor/wrapper. Unlimited `0/0` is prohibited for autonomous runs and is allowed only in an interactive session with Jose present.

| Profile | Hard timeout | Max turns | Processed-token budget | Metered budget |
|---|---:|---:|---:|---:|
| `light` | 300s | 6 | 200,000 | USD 0.50 |
| `standard` | 600s | 16 | 1,200,000 | USD 2 |
| `implementation` | 1,200s | 30 | 2,000,000 | USD 5 |
| `large` | 1,800s | 45 | 4,000,000 | USD 10 |
| `high-risk-review` | 900s | 24 | 1,500,000 | USD 8 |
| `retry` | 600s | 16 | 1,000,000 | USD 2 |

Subscription providers are bounded by processed tokens; metered providers are bounded by USD. Both retain positive timeout and max-turns limits. Durable launchers MUST have an outer deadline at least 60 seconds above the selected inner timeout. Scheduler `flock`/`fire_claim` and per-worktree locks prevent overlapping execution.

Canonical terminal reasons are:

- `limit_reason=hard-timeout`
- `limit_reason=max-turns`
- `limit_reason=budget`

A threshold hit interrupts the agent, terminates the whole process group, records telemetry, and fails the run. A fallback summary produced after max-turn exhaustion is not a successful completion.

### Re-scope rule

A stable task key may receive at most one automatic split/re-scope after its first limit hit. That run uses the bounded `retry` profile. If the re-scoped run reaches any limit again, the supervisor records:

```text
limit_count=2
action=stop-and-alert
```

The task then stops permanently until manual intervention and emits a Discord alert. There is no third automatic attempt. Only a verified successful completion clears the counter; failed or incomplete result dictionaries do not.

### Coverage boundary

- Agentic cron jobs run through `cron.autonomous_limits.run_with_autonomous_limits` and carry an explicit positive profile.
- Script-only jobs that call a model use `hermes_autonomous_cli_wrapper.py` in `hermes-db` or `child` accounting mode.
- Deterministic `no_agent` scripts that never call a model are in scope for inventory but do not require an LLM wrapper.
- Passive provider/listener services accept requests but do not originate autonomous work, so they are not autonomous entry points.

## Consequences

### Positive

- Deterministic upper bounds on runtime, model calls, tokens, and spend.
- Process-tree cleanup is testable through `residual_processes=0`.
- One canonical contract covers cron agents, Hermes CLI controllers, and direct API workers.
- Limit-driven task splitting cannot become a retry loop.
- The profiles provide capacity for roughly 10× current workload volume without reopening the safety decision; larger work is decomposed or explicitly assigned `large`, never made unlimited.

### Trade-offs

- A useful partial result may be classified as failure when a limit is reached.
- Conservative preflight cost estimates may stop a direct API worker before its exact provider bill would exceed the cap.
- Workloads that legitimately need more capacity require an explicit profile decision rather than implicit continuation.

## Alternatives rejected

1. **Provider limits only:** rejected because they do not guarantee local process-tree cleanup or consistent turn accounting.
2. **Global unlimited runs with watchdogs:** rejected because a watchdog is detection, not a deterministic resource contract.
3. **Unlimited automatic splitting:** rejected because each split can become a new runaway loop.
4. **One profile for every task:** rejected because reminders, standard research, implementation, large-context work, and high-risk review need different positive envelopes.

## Verification and acceptance

Acceptance requires all of the following:

- 100% inventory coverage with zero pending model-capable autonomous paths.
- Cheap fixtures proving `limit_reason=max-turns` and `limit_reason=budget`.
- `residual_processes=0` for each fixture.
- Regression tests covering hard timeout, max turns, token/USD budget, first re-scope, second-limit stop-and-alert, and success-only counter reset.

Evidence for the accepted implementation:

- Coverage: 60 enabled crons plus 5 launcher surfaces; 65 covered, 0 pending.
- Model-capable workloads: 32 migrated, 0 pending.
- Scheduler regressions: 283 passed, 0 failed.
- Core implementation commit: `8e91f55314`.
- Machine-readable report: `file:///home/jose-naicipa/.hermes/profiles/vpsclone/home/.hermes/runs/autonomous-entrypoint-coverage-20260724/final-coverage-report.json`.

## Rollback

Rollback is permitted only to the previous bounded implementation, never to autonomous `0/0`. Revert the scheduler integration commit, disable affected autonomous jobs, and retain deterministic `no_agent` jobs while restoring a previously verified positive-limit wrapper. Any replacement must pass the same fixtures and coverage gate before jobs are re-enabled.
