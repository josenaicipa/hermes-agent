// FakeGateway — Phase 0 stand-in for the real gatewayClient/rpc transport.
// Emits a representative transcript and can stream an assistant reply so the
// native view can be exercised without a Python tui_gateway behind it.
//
// Phase 4: also implements the prompt channel + a spy `respond` so the headless
// prompts verifier (src/demo.prompts.tsx) can drive synthetic clarify/approval/
// sudo/secret/confirm requests and assert the correct *.respond RPC fires.
import type { Msg, PromptState } from './model.ts'

export type Listener = (msgs: Msg[]) => void
export type PromptListener = (prompt: PromptState | null) => void

/** A recorded *.respond RPC call, captured by the FakeGateway spy. */
export interface RespondCall {
  method: string
  params: Record<string, unknown>
}

const SEED: Msg[] = [
  { role: 'user', text: 'how do I switch the TUI to opentui?' },
  {
    role: 'assistant',
    text:
      'You build a **native** OpenTUI view layer and keep the renderer-agnostic logic. ' +
      'The Ink engine stays as the *default*; OpenTUI runs behind `HERMES_TUI_ENGINE=opentui`.\n\n' +
      'Key points:\n' +
      '- `domain/`, `protocol/`, stores and `gatewayClient` are **reused**.\n' +
      '- The ~10k LOC of `.tsx` is **rewritten** against `@opentui/react`.\n' +
      '- `wrap-trim` and ANSI parsing become native span emitters.'
  },
  { role: 'tool', text: '$ bun src/entry.opentui.tsx\nrenderer 120x32 — frame ok (3.2ms)' },
  { role: 'system', text: 'engine=opentui · runtime=bun · phase=0 skeleton' }
]

export class FakeGateway {
  private msgs: Msg[] = [...SEED]
  private listeners = new Set<Listener>()

  // ── Prompt channel + RPC spy (Phase 4) ────────────────────────────────
  private prompt: PromptState | null = null
  private promptListeners = new Set<PromptListener>()
  /** Every respond() call, in order. The verifier asserts against this. */
  readonly respondCalls: RespondCall[] = []
  /** Local confirm resolutions (ok=true on confirm, false on cancel). */
  readonly confirmResults: boolean[] = []

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn)
    fn(this.msgs)

    return () => this.listeners.delete(fn)
  }

  subscribePrompt(fn: PromptListener): () => void {
    this.promptListeners.add(fn)
    fn(this.prompt)

    return () => this.promptListeners.delete(fn)
  }

  getPrompt(): PromptState | null {
    return this.prompt
  }

  setPrompt(p: PromptState | null): void {
    this.prompt = p

    for (const fn of this.promptListeners) {
      fn(this.prompt)
    }
  }

  /** Spy for the *.respond RPCs — records the call and resolves immediately. */
  respond<T = unknown>(method: string, params: Record<string, unknown>): Promise<T> {
    this.respondCalls.push({ method, params })

    return Promise.resolve(undefined as T)
  }

  /** Synthetic session id (approval.respond carries one). */
  sessionId(): string | null {
    return 'fake-session'
  }

  /** Called by the app when a LOCAL confirm dialog resolves (no RPC). */
  onLocalConfirm(ok: boolean): void {
    this.confirmResults.push(ok)
  }

  private emit() {
    const snapshot = [...this.msgs]

    for (const fn of this.listeners) {
      fn(snapshot)
    }
  }

  /** Simulate a user submit + streamed assistant reply. onDone fires at end. */
  send(text: string, onDone?: () => void): void {
    this.msgs = [...this.msgs, { role: 'user', text }]
    this.emit()

    const reply =
      'Native OpenTUI reply to: *' + text + '*. ' + 'This text streams token-by-token to exercise incremental render.'

    const words = reply.split(' ')
    let i = 0
    const idx = this.msgs.length
    this.msgs = [...this.msgs, { role: 'assistant', text: '', streaming: true }]
    this.emit()

    const timer = setInterval(() => {
      i++
      const partial = words.slice(0, i).join(' ')
      const next = [...this.msgs]
      next[idx] = { role: 'assistant', text: partial, streaming: i < words.length }
      this.msgs = next
      this.emit()

      if (i >= words.length) {
        clearInterval(timer)
        onDone?.()
      }
    }, 60)
  }
}
