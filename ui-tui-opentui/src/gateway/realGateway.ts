// RealGateway — Phase 2 transport.
//
// Path-imports the REAL renderer-agnostic GatewayClient from the sibling Ink
// package (zero drift: no copy) and adapts its request()-based RPC surface +
// EventEmitter stream into the SAME { subscribe, send } shape the OpenTUI app
// already consumes from FakeGateway.
//
// Wiring order (per the client's contract):
//   new GatewayClient() → start() → on('event')/on('exit') → drain()
// drain() flushes events buffered before subscription.
//
// Submit RPC discovered in ../../ui-tui/src/app/useSubmission.ts:110 —
//   gw.request<PromptSubmitResponse>('prompt.submit', { session_id, text })
// A session is created first via session.create (useSessionLifecycle.ts:170 —
//   rpc<SessionCreateResponse>('session.create', { cols })).
import { GatewayClient } from '../../../ui-tui/src/gatewayClient.ts'
import type { PromptState } from '../model.ts'

import { EventAdapter, type Listener, type PromptListener } from './eventAdapter.ts'

export { GatewayClient }

interface SessionCreateResponse {
  session_id: string
  info?: unknown
}

/**
 * Adapts the real GatewayClient to the app's { subscribe, send } interface.
 *
 * NOTE: the caller (entry / demo) MUST set process.env.HERMES_PYTHON_SRC_ROOT
 * to the hermes repo root before constructing this — the GatewayClient resolves
 * the python spawn root relative to its OWN file location otherwise, which is
 * wrong when imported across directories.
 */
export class RealGateway {
  readonly client: GatewayClient
  readonly adapter: EventAdapter
  private sid: string | null = null
  private sessionPromise: Promise<string> | null = null
  private cols: number

  constructor(opts: { cols?: number } = {}) {
    this.cols = opts.cols ?? 80
    this.client = new GatewayClient()
    this.adapter = new EventAdapter(this.client)
  }

  /** Spawn the gateway and begin draining buffered events. Idempotent-ish. */
  start(): void {
    this.client.start()
    this.adapter.attach()
    this.client.drain()
  }

  /** The app subscribes here, exactly as it does for FakeGateway. */
  subscribe(fn: Listener): () => void {
    return this.adapter.subscribe(fn)
  }

  /** Subscribe to the prompt channel (blocking interactive requests). */
  subscribePrompt(fn: PromptListener): () => void {
    return this.adapter.subscribePrompt(fn)
  }

  /** Set or clear the active prompt (used to clear after answering, or to
   * drive a local confirm dialog). */
  setPrompt(p: PromptState | null): void {
    this.adapter.setPrompt(p)
  }

  getPrompt(): PromptState | null {
    return this.adapter.getPrompt()
  }

  /**
   * Local confirm resolution hook. The /new, /clear confirm dialogs are not
   * yet wired to real gateway actions in the OpenTUI engine (later phase), so
   * this is a no-op today — the prompt is simply cleared by the app. Present
   * for interface parity with FakeGateway.
   */
  onLocalConfirm(_ok: boolean): void {
    void _ok
  }

  /** The current session id (null before session.create resolves). Needed for
   * approval.respond's { session_id } reply param. */
  sessionId(): string | null {
    return this.sid
  }

  /**
   * Thin wrapper over the GatewayClient RPC for the *.respond replies
   * (clarify/approval/sudo/secret.respond) so callers don't reach into
   * `.client`. Returns the RPC promise.
   */
  respond<T = unknown>(method: string, params: Record<string, unknown>): Promise<T> {
    return this.client.request<T>(method, params)
  }

  /** Expose the adapter status (ready flag + last status text). */
  getStatus() {
    return this.adapter.getStatus()
  }

  private async ensureSession(): Promise<string> {
    if (this.sid) {
      return this.sid
    }

    if (this.sessionPromise) {
      return this.sessionPromise
    }

    this.sessionPromise = this.client.request<SessionCreateResponse>('session.create', { cols: this.cols }).then(r => {
      this.sid = r.session_id

      return r.session_id
    })

    return this.sessionPromise
  }

  /**
   * Submit a user prompt. Mirrors FakeGateway.send(text, onDone): echoes the
   * user message locally (the gateway only streams assistant/tool events), then
   * fires prompt.submit. onDone resolves when message.complete lands OR on error.
   */
  send(text: string, onDone?: () => void): void {
    // Echo the user's message into the transcript immediately.
    this.adapter.pushUser(text)

    const finish = (() => {
      let called = false

      return () => {
        if (called) {
          return
        }

        called = true
        onDone?.()
      }
    })()

    // Resolve onDone when the turn completes.
    const onEvent = (ev: { type?: string }) => {
      if (ev?.type === 'message.complete' || ev?.type === 'error') {
        this.client.off('event', onEvent)
        finish()
      }
    }

    this.client.on('event', onEvent)

    void this.ensureSession()
      .then(sid => this.client.request('prompt.submit', { session_id: sid, text }))
      .catch((e: Error) => {
        this.client.off('event', onEvent)
        finish()
        // The 'error' gateway event (if any) is surfaced by the adapter; this
        // catch just prevents an unhandled rejection on transport failure.
        void e
      })
  }

  kill(reason = 'shutdown'): void {
    this.adapter.detach()

    try {
      this.client.kill(reason)
    } catch {
      /* already gone */
    }
  }
}
