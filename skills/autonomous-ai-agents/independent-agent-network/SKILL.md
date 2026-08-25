---
name: independent-agent-network
description: Route work to isolated agents via the network broker.
version: 1.0.0
author: Jose Naicipa (@josenaicipa)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Multi-Agent, Profiles, Linear, 1Password, Dispatch]
    category: autonomous-ai-agents
    related_skills: [hermes-agent]
---

# Independent Agent Network Skill

Dispatch work to the canonical Naicipa fleet. Each agent is an isolated
Hermes profile with a pinned model. Routing is by alias or lane name.
Every job must cite a Linear issue. Credentials come from the 1Password
broker — never from prompts, memory, or receipts.

This skill does not spawn Discord bots or create extra platform tokens.

## When to Use

- Provision or inspect the canonical roster.
- Send a task to Oscar, Ada, or any other named agent.
- Request a credential for a profile without printing the value.

## Prerequisites

- Hermes CLI on PATH (`hermes network --help`).
- Isolated profiles created with `hermes network provision`.
- A Linear issue id such as `NAI-68` for every dispatch.
- 1Password CLI (`op`) authenticated in the **broker** process, not in
  the agent prompt.

## How to Run

Use `` `terminal` ``. Do not add a core tool.

```
hermes network roster --json
hermes network provision --core
hermes network route Oscar
hermes network dispatch Ada --linear NAI-68 --goal "Review the landing copy"
hermes network credentials request --profile oscar --name OPENAI_API_KEY
```

## Quick Reference

| Agent | Lane | Model |
|---|---|---|
| Oscar | producto | grok-4.6 |
| Ada | critico | claude-opus-5 |
| Sebastian | visual | claude-sonnet-5 |
| Juan | growth | grok-4.6 |
| Frank | infra | grok-4.6 |
| Nerd | research | gpt-5.6-terra |

Department agents: CRM, Revenue, Commerce, Edu, Content, Mat. Full
roster: `hermes network roster`.

## Procedure

1. Resolve the target with `hermes network route <alias>`.
2. Confirm a Linear issue exists. Dispatch without `--linear` is refused.
3. Dispatch asynchronously. The command returns a job id and exits.
4. If the child needs a provider key, request it through
   `hermes network credentials`. The receipt has `granted` and
   `reference` only — never the secret.
5. Do not copy `.env` files between profiles.

## Pitfalls

- Unknown aliases fail closed. There is no default-profile fallback.
- Do not paste `op://` resolved values, tokens, or `.env` contents into
  chat, memory, or the job goal.
- Do not run `systemctl` against `hermes-gateway-*` or
  `memory-fabric.service`.
- Do not create Discord bots or extra platform tokens from this skill.

## Verification

```
hermes network status --json
hermes network jobs --json
```

A provisioned core agent has `provisioned: true` and `pinned_model`
equal to the roster model. A credentials receipt must omit any secret
value.
