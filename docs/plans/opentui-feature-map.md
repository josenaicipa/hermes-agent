# OpenTUI Feature Map & Porting Roadmap

**What this is:** the complete inventory of every Hermes TUI feature — slash commands, overlays,
modals, prompts, chrome/HUD, and agent-level surfaces — with Ink (source of truth) vs the new
native OpenTUI engine (`ui-tui-opentui/`) port status. This is the master checklist that scopes
all remaining phases. Compiled 2026-06-08 from 3 parallel file:line-grounded recon passes.

**Source of truth = Ink TUI** (`ui-tui/src/`) + Python registry (`hermes_cli/commands.py`).
**Target = `ui-tui-opentui/`** (native OpenTUI on Bun; Ink stays default & untouched).
**Companion docs:** `opentui-native-rewrite-spec.md` (the spec), `opentui-migration-spec.md`
§11–14 (launcher/distribution review).

Legend: ✅ done in OpenTUI · ⚠️ partial · ❌ missing · 🔴 blocking (unhandled = agent deadlock).

---

## 0. Current OpenTUI engine state (what exists today)

Renders: static header line, transcript scrollbox w/ role gutters, markdown→spans, tool-result
bordered box, streaming `▍` cursor, single-line `<input>` composer, basic status text.
`src/gateway/eventAdapter.ts` handles: `gateway.ready`, `message.start/delta/complete`,
`thinking/reasoning.delta` (stored on `Msg.thinking` but **not rendered**), `tool.start` (label
only), `tool.complete`, `status.update`, `error`, `gateway.stderr/start_timeout/protocol_error`,
**and (Phase 4 ✅) the 4 interactive `*.request` events** (clarify/approval/sudo/secret) via a
native prompt overlay + `*.respond` RPCs — the deadlock is fixed.
**Explicitly drops** (eventAdapter `default:` branch): `notification.*`, `voice.*`,
`browser.progress`, `background.complete`, `subagent.*`, `tool.progress/generating`,
`reasoning.available`.

---

## 1. SLASH COMMANDS

**Canonical registry:** `hermes_cli/commands.py:64` `COMMAND_REGISTRY` — **70 `CommandDef` entries**,
5 categories. The OpenTUI engine should consume the `commands.catalog` RPC (not hardcode), exactly
like Ink. **Status: ❌ the OpenTUI app has NO slash command handling yet.**

### Dispatch ladder to reproduce (`ui-tui/src/app/createSlashHandler.ts:10`)
1. Parse (`domain/slash.ts:6`) → 2. client-local handler (`app/slash/registry.ts:20`, aggregates
`commands/{core,session,ops,setup,debug}.ts`) → 3. catalog alias/prefix (`catalog.canon`) →
4. `gw.request('slash.exec', …)` (Python `_SlashWorker` subprocess) → 5. `command.dispatch`
fallback (quick_commands / plugins / **skills** / pending-input).
- Forced to `command.dispatch` (slash.exec rejects): `_PENDING_INPUT_COMMANDS` =
  {retry, queue, q, steer, goal, undo} (`tui_gateway/server.py:6461`); `_WORKER_BLOCKED` =
  {snapshot, snap} (`:6473`).
- Skills (not in registry) route via `command.dispatch` → `{type:"skill", message}` → submitted as
  a user turn. `{type:"alias"}` re-dispatches; `{type:"prefill"}` fills the composer.
- TUI catalog hides `_TUI_HIDDEN`={sethome,commands,approve,deny} and adds `_TUI_EXTRA`=
  {compact,details,logs,mouse} (`server.py:6437,6447`).

### Commands that OPEN a UI surface (port priority — need a component)
| Command(s) | Opens | Ink component |
|---|---|---|
| `/model` (bare) | model picker | `modelPicker.tsx` (`appOverlays.tsx:161`) |
| `/sessions` `/resume` `/switch` `/session` | session switcher | `activeSessionSwitcher.tsx` (`appOverlays.tsx:145`) |
| `/skills` (bare) | skills hub | `skillsHub.tsx` (`appOverlays.tsx:173`) |
| `/agents` `/tasks` `/replay` `/replay-diff` | agents dashboard | `agentsOverlay.tsx` (`appLayout.tsx:409`) |
| `/new` `/clear` | confirm dialog | `prompts.tsx` ConfirmPrompt (`appOverlays.tsx:49`) |
| `/status` `/usage` `/history` `/logs` `/tools` `rollback diff`, long `/skills` | pager | `FloatBox` pager (`appOverlays.tsx:177`) |
| `/help` | inline panel (not overlay) | `transcript.panel()` (`core.ts:108`) |

### TUI-only client commands (13, NOT in COMMAND_REGISTRY — must reimplement)
`mouse/scroll`, `redraw`, `compact`, `details`, `fortune`, `terminal-setup`, `logs`(→pager),
`sessions`(→switcher), `replay`/`replay-diff`(→agents), `setup`(suspend+shell), `heapdump`, `mem`.
Defined in `ui-tui/src/app/slash/commands/{core,session,ops,setup,debug}.ts`.

### Full 70-command registry (abridged — full table in recon, all in `commands.py:64-225`)
- **Session (29):** start, new/reset, topic, clear, redraw, history, save, retry, undo, title,
  handoff, branch/fork, compress, rollback, snapshot/snap, stop, approve, deny, background/bg/btw,
  agents/tasks, queue/q, steer, goal, subgoal, status, sethome, resume, sessions, restart.
- **Configuration (15):** config, model, codex-runtime, personality, statusbar/sb, verbose, footer,
  yolo, reasoning, fast, skin, indicator, voice, busy.
- **Tools & Skills (12):** tools, toolsets, skills, bundles, cron, curator, kanban, reload,
  reload-mcp, reload-skills, browser, plugins.
- **Info (13):** whoami, profile, gquota, commands, help, usage, insights, platforms/gateway,
  platform, copy, paste, image, update, version/v, debug.
- **Exit (1):** quit/exit.
Subcommand completion declared for: footer, reasoning, fast, voice, busy, indicator, skills, cron,
curator, kanban.

### Autocomplete (`ui-tui/src/hooks/useCompletion.ts:41`)
`looksLikeSlashCommand` → `complete.slash` RPC (server builds `SlashCommandCompleter` from registry
+ skills/bundles + TUI extras, caps 30); else `complete.path`. `/model …` returns null → uses the
picker instead. Dropdown rendered in `FloatingOverlays` (`appOverlays.tsx:203`).

---

## 2. OVERLAYS / MODALS / POPUPS / PROMPTS

State: single atom `$overlayState` (`ui-tui/src/app/overlayStore.ts:19`, 11 slots); computed
`$isBlocked` hides the composer when any slot is set (`appLayout.tsx:273`). Two render zones:
**PromptZone** (inline blocking prompts, priority approval→confirm→clarify→sudo→secret) and
**FloatingOverlays** (dropdowns above composer). Agents overlay replaces the transcript pane.
Lifecycle: `resetFlowOverlays()` clears prompts/pager at turn-end but **preserves** user overlays
(agents/modelPicker/sessions/skillsHub).

### 2a. ✅ BLOCKING gateway prompts — DONE (Phase 4; was 🔴 unhandled = deadlock)
Dispatched in `createGatewayEventHandler.ts:722-747` (Ink); in the OpenTUI engine handled by
`src/gateway/eventAdapter.ts` → prompt channel → `src/components/prompts/promptOverlay.tsx`,
replied via the `*.respond` RPCs. Verified by `bun src/demo.prompts.tsx` (45/45 green).

| Event | Payload | Component | Responds | RPC reply | Port |
|---|---|---|---|---|---|
| `clarify.request` | `{choices[]\|null, question, request_id}` | `prompts/clarifyPrompt.tsx` (`<select>`+Other→free-text) | ↑↓/1-N/Enter, "Other"→free-text, Esc | `clarify.respond {answer, request_id}` | ✅ |
| `approval.request` | `{command, description}` | `prompts/approvalPrompt.tsx` (`<select>`) | ↑↓/1-4 once/session/always/deny, Esc/Ctrl+C→deny | `approval.respond {choice, session_id}` | ✅ |
| `sudo.request` | `{request_id}` | `prompts/maskedPrompt.tsx` 🔐 | masked pw, Enter, Esc/Ctrl+C→'' | `sudo.respond {password, request_id}` | ✅ |
| `secret.request` | `{env_var, prompt, request_id}` | `prompts/maskedPrompt.tsx` 🔑 | masked input, Enter, Esc/Ctrl+C→'' | `secret.respond {value, request_id}` | ✅ |

Cancel paths (Ctrl+C/Esc) send the deny/cancel RPC so the agent unblocks. **`confirm`** is a local
(non-gateway) blocking dialog (`prompts/confirmPrompt.tsx`, Y/N/Esc) driven by a local callback
(`gw.onLocalConfirm`), not an RPC — ✅ included.

### 2b. Floating overlays / pickers
| Name | Trigger | Component | Port |
|---|---|---|---|
| Model picker | `/model`, embedded in switcher | `modelPicker.tsx` | hard (multi-stage + fuzzy + key entry) |
| Session switcher | `/resume`, **Ctrl+X**, click count | `activeSessionSwitcher.tsx` | hard (merged list + embeds model picker + close/delete RPCs) |
| Skills hub | `/skills` | `skillsHub.tsx` | hard (3-stage + install) |
| Agents dashboard | `/agents`, `/replay*` | `agentsOverlay.tsx` | hard (tree + Gantt + accordions + draggable scrollbar; largest single port) |
| Pager | `transcript.page()` — many `/cmd`s | `appOverlays.tsx:177` | moderate (porting it unlocks `/status /logs /history /tools` at once) |
| Completions dropdown | typing `/` or path | `appOverlays.tsx:203` | moderate |

### 2c. Passive / inline (not overlay slots, don't block)
Help hint (`?` card, `helpHint.tsx`), queued-messages strip (`queuedMessages.tsx`), todo panel
(`todoPanel.tsx`), thinking/reasoning + subagent tree (`thinking.tsx`, inline transcript), `/help`
panel (`transcript.panel()`), FPS overlay (`fpsOverlay.tsx`).

---

## 3. CHROME (persistent UI) + AGENT FEATURES — the gap list

Ink chrome composed in `appLayout.tsx`; the **status rule** (`appChrome.tsx:390`) is one
progressively-disclosed line. Live turn state in `turnStore.ts` (`TurnState`); UI state in
`uiStore.ts`. (Note: there is **no `turnController.ts` file** — `turnController.*` is an object
invoked from `createGatewayEventHandler.ts`.)

### Chrome gaps
| Feature | Ink | OpenTUI | Port | Drives from |
|---|---|---|---|---|
| Model in header | `appChrome.tsx:547` | ❌ | trivial | `SessionInfo.model/reasoning_effort/fast` |
| Session id | `branding.tsx:296` | ❌ | trivial | session.info |
| cwd / branch label | `appChrome.tsx:614` | ❌ | trivial | `SessionInfo.cwd` |
| Context % + token bar | `appChrome.tsx:551` | ❌ | moderate | `Usage.context_*` |
| Cost read-out | `appChrome.tsx:596` | ❌ | moderate | `Usage.cost_usd` |
| Compressions/duration/dev-credits | `appChrome.tsx:564-607` | ❌ | moderate | Usage/session |
| Update-available banner | `branding.tsx:397` | ❌ | trivial | `SessionInfo.update_behind/command` |
| Profile in prompt | `appLayout.tsx:179` | ❌ | trivial | `SessionInfo.profile_name` |
| MCP servers panel | `branding.tsx:246` | ❌ | moderate | `SessionInfo.mcp_servers[]` |
| Banner / SessionPanel intro | `branding.tsx:85/160` | ❌ | moderate | theme/session |
| Response separator `───` | `appLayout.tsx:108` | ❌ | trivial | history roles |
| Draggable scrollbar | `appChrome.tsx:653` | ⚠️ (auto only) | moderate | scroll state |
| Sticky-prompt line | `appLayout.tsx:245` | ❌ | moderate | viewport scroll |
| FPS overlay / help hint / GoodVibesHeart | various | ❌ | trivial | cosmetic |
| Busy face/verb/elapsed ticker | `appChrome.tsx:119` | ⚠️ (text only) | moderate | turn timing |
| Queued messages | `queuedMessages.tsx` | ❌ | moderate | composer queue |
| Multiline input / paste / history | `textInput.tsx` | ❌ | moderate | replaces `<input>` |

### Agent-feature gaps (each: gateway event → turn field → renderer)
| Feature | Ink renderer | OpenTUI | Port |
|---|---|---|---|
| Reasoning/thinking display | `thinking.tsx:621` (`reasoning.delta/available`) | ❌ (data captured on `Msg.thinking`, never rendered) | moderate |
| Tool trail (live spinner+args+timing+collapse) | `thinking.tsx:689` (`tool.start/generating/progress`) | ⚠️ flat labels only | moderate |
| Tool result (inline diffs) | inline-diff path `cgeh:698` | ⚠️ plain box | moderate |
| Subagents/delegation tree | `thinking.tsx:281` + `agentsOverlay` (`subagent.*`) | ❌ (dropped) | **hard** (biggest) |
| Delegation HUD (SpawnHud) | `appChrome.tsx:270` (`$delegationState`) | ❌ | hard |
| Todos panel | `todoPanel.tsx` (`payload.todos`) | ❌ | moderate |
| Activity feed | `thinking.tsx:878` (status/stderr) | ❌ | hard (coupled to `/details` section visibility) |
| Notifications sticky/ttl | `appChrome.tsx:533` (`notification.show/clear`) | ❌ | moderate |
| Voice listening/transcribing | `appChrome.tsx:578` (`voice.status/transcript`) | ❌ | moderate |
| Browser progress | system line (`browser.progress`) | ❌ | trivial |
| Background-task completion + count | `cgeh:752`, count `:590` (`background.complete`) | ❌ | trivial |

---

## 4. RECOMMENDED PORT ORDER (consolidated)

1. **Phase 4 — 🔴 blocking prompts + confirm** (§2a). Deadlock-critical; sudo/secret trivial,
   clarify/approval moderate. Makes any non-trivial session actually usable. **DO THIS FIRST.**
2. **Wire `session.info` + `Usage` into the adapter** → unlocks most trivial chrome (model, cwd,
   context%, cost, update banner, profile) in one stroke.
3. **Reasoning render + tool trail** (data largely already captured) + todos panel.
4. **Pager + completions dropdown** → unlocks many `/commands` + slash autocomplete.
5. **Slash command system** (catalog RPC + dispatch ladder + the 13 TUI-only cmds).
6. **Pickers:** model → session switcher → skills hub.
7. **Subagents tree + agents dashboard + SpawnHud** (hardest; last).
8. **Polish:** banner/SessionPanel/MCP panel, sticky-prompt, draggable scrollbar, queued msgs,
   multiline input, notifications, voice, FPS/help-hint.

This map IS the backlog. Each row is an independently portable unit with its Ink reference.
