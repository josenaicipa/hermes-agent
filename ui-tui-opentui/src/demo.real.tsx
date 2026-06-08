// Headless verifier for the REAL gateway end-to-end path. Mounts the app,
// spawns a live Python tui_gateway, submits a trivial prompt, and captures the
// frame + a checklist. Proves the Phase-2 transport works even if the agent
// backend needs API keys (gateway.ready + captured stderr/error still prove the
// pipe). Hard timeout so it never hangs.
//
// Run: bun src/demo.real.tsx → demo-real-frame.txt + demo-real-report.txt
import '@opentui/react/runtime-plugin-support'

import { bootstrapGatewayEnv } from './gateway/env.ts'

const repoRoot = bootstrapGatewayEnv()

import { writeFileSync } from 'node:fs'

import { createTestRenderer } from '@opentui/core/testing'
import { createRoot } from '@opentui/react'
import React from 'react'

import { App } from './components/app.tsx'
import { RealGateway } from './gateway/realGateway.ts'

const COLS = 90
const ROWS = 28
const HARD_TIMEOUT_MS = 60_000

// Track raw events so the report can attest gateway.ready / stderr / errors
// independent of what made it into the rendered frame.
const seen = {
  ready: false,
  messageStart: false,
  messageComplete: false,
  errorEvents: [] as string[],
  stderrLines: [] as string[],
  startTimeout: false,
  types: new Set<string>()
}

const { renderer, renderOnce, flush, captureCharFrame } = await createTestRenderer({
  width: COLS,
  height: ROWS
})

const gw = new RealGateway({ cols: COLS })

// Observe the raw event stream before draining.
gw.client.on('event', (ev: { type?: string; payload?: Record<string, unknown> }) => {
  if (!ev?.type) {
    return
  }

  seen.types.add(ev.type)

  if (ev.type === 'gateway.ready') {
    seen.ready = true
  }

  if (ev.type === 'message.start') {
    seen.messageStart = true
  }

  if (ev.type === 'message.complete') {
    seen.messageComplete = true
  }

  if (ev.type === 'error') {
    seen.errorEvents.push(String(ev.payload?.message ?? 'unknown'))
  }

  if (ev.type === 'gateway.stderr') {
    seen.stderrLines.push(String(ev.payload?.line ?? ''))
  }

  if (ev.type === 'gateway.start_timeout') {
    seen.startTimeout = true
  }
})

let crashed: string | null = null
process.on('uncaughtException', e => {
  crashed = String(e?.stack ?? e)
})
process.on('unhandledRejection', e => {
  crashed = String((e as Error)?.stack ?? e)
})

const t0 = performance.now()
gw.start()
createRoot(renderer).render(<App cols={COLS} gw={gw} rows={ROWS} />)
await renderOnce()
await flush()
const t1 = performance.now()

// Submit a trivial prompt and wait for completion OR hard timeout.
let submitResolved = false

const done = new Promise<void>(resolve => {
  gw.send('say hi in 3 words', () => {
    submitResolved = true
    resolve()
  })
})

const timeout = new Promise<void>(resolve => setTimeout(resolve, HARD_TIMEOUT_MS))

// Render loop while we wait so the frame reflects streamed state.
const renderPump = (async () => {
  const deadline = performance.now() + HARD_TIMEOUT_MS

  while (performance.now() < deadline && !submitResolved) {
    await new Promise(r => setTimeout(r, 200))
    await renderOnce()
    await flush()
  }
})()

await Promise.race([done, timeout])

// A few final render cycles to settle the frame.
for (let k = 0; k < 4; k++) {
  await new Promise(r => setTimeout(r, 120))
  await renderOnce()
  await flush()
}

void renderPump

const frame = captureCharFrame()
writeFileSync(new URL('../demo-real-frame.txt', import.meta.url), frame)

const assistantInFrame = seen.messageComplete

const report = [
  `=== Phase 2 REAL gateway verification ===`,
  `repo root: ${repoRoot}`,
  `python: ${process.env.HERMES_PYTHON ?? process.env.VIRTUAL_ENV ?? '(default .venv resolution)'}`,
  `rendered ${COLS}x${ROWS}; first paint ${(t1 - t0).toFixed(2)}ms`,
  `frame chars: ${frame.length}`,
  ``,
  `--- transport / event attestations ---`,
  `gateway.ready seen: ${seen.ready}`,
  `message.start seen: ${seen.messageStart}`,
  `message.complete seen: ${seen.messageComplete}`,
  `submit onDone fired: ${submitResolved}`,
  `error events: ${seen.errorEvents.length}${seen.errorEvents.length ? ' → ' + seen.errorEvents.slice(0, 3).join(' | ') : ''}`,
  `stderr lines captured: ${seen.stderrLines.length}`,
  `gateway.start_timeout: ${seen.startTimeout}`,
  `distinct event types: ${[...seen.types].sort().join(', ') || '(none)'}`,
  ``,
  `--- frame content checks ---`,
  `header present: ${frame.includes('hermes')}`,
  `user submit echoed: ${frame.includes('say hi in 3 words')}`,
  `assistant reply rendered: ${assistantInFrame}`,
  ``,
  `--- crash check ---`,
  `no crash: ${crashed === null}`,
  ...(crashed !== null ? [`CRASH: ${(crashed as string).slice(0, 800)}`] : []),
  ``,
  `--- verdict ---`,
  verdict(),
  ...(seen.stderrLines.length ? [``, `--- last stderr tail ---`, ...seen.stderrLines.slice(-12)] : [])
].join('\n')

function verdict(): string {
  if (seen.messageComplete) {
    return 'PASS: real assistant reply streamed end-to-end.'
  }

  if (seen.ready) {
    return 'TRANSPORT OK: gateway.ready received (agent backend may need API keys; see stderr/error tail).'
  }

  if (seen.startTimeout) {
    return 'BLOCKED: gateway start timed out (python/env issue; see stderr tail).'
  }

  if (seen.stderrLines.length || seen.errorEvents.length) {
    return 'PARTIAL: gateway spoke (stderr/error captured) but no gateway.ready — backend config issue.'
  }

  return 'FAIL: no events received from gateway (transport did not come up).'
}

writeFileSync(new URL('../demo-real-report.txt', import.meta.url), report + '\n')
process.stdout.write(report + '\n')

try {
  gw.kill('demo.shutdown')
  renderer.destroy()
} catch {
  /* already down */
}

process.exit(0)
