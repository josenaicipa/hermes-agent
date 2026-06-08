// Native OpenTUI prompt overlay dispatcher (Phase 4).
//
// Renders the active interactive prompt (clarify/approval/sudo/secret/confirm)
// and wires each answer/cancel to the correct *.respond RPC so the blocked
// Python agent unblocks. This is the deadlock fix: every exit path — answer,
// pick, Enter, Esc, Ctrl+C — calls a respond RPC (or the local confirm hook)
// and then clears the prompt.
//
// Reply RPC contract (verified against ui-tui/src/app/useMainApp.ts &
// gatewayTypes.ts):
//   clarify.respond  { answer, request_id }
//   approval.respond { choice, session_id }
//   sudo.respond     { password, request_id }
//   secret.respond   { value, request_id }
import React, { useCallback } from 'react'

import type { PromptState } from '../../model.ts'
import type { Theme } from '../../theme.ts'

import { type ApprovalChoice, ApprovalPromptView } from './approvalPrompt.tsx'
import { ClarifyPromptView } from './clarifyPrompt.tsx'
import { ConfirmPromptView } from './confirmPrompt.tsx'
import { MaskedPrompt } from './maskedPrompt.tsx'

/** The subset of the Gateway the prompt overlay needs. */
export interface PromptGateway {
  respond(method: string, params: Record<string, unknown>): Promise<unknown>
  setPrompt(p: PromptState | null): void
  sessionId(): string | null
  onLocalConfirm(ok: boolean): void
}

export function PromptOverlay({ gw, prompt, t }: { gw: PromptGateway; prompt: PromptState; t: Theme }) {
  const clear = useCallback(() => gw.setPrompt(null), [gw])

  // clarify.respond { answer, request_id }. Empty answer = cancel.
  const onClarify = useCallback(
    (answer: string, requestId: string) => {
      void gw.respond('clarify.respond', { answer, request_id: requestId })
      clear()
    },
    [gw, clear]
  )

  // approval.respond { choice, session_id }
  const onApproval = useCallback(
    (choice: ApprovalChoice) => {
      void gw.respond('approval.respond', { choice, session_id: gw.sessionId() })
      clear()
    },
    [gw, clear]
  )

  // sudo.respond { password, request_id }. Empty password = cancel.
  const onSudo = useCallback(
    (password: string, requestId: string) => {
      void gw.respond('sudo.respond', { password, request_id: requestId })
      clear()
    },
    [gw, clear]
  )

  // secret.respond { value, request_id }. Empty value = cancel.
  const onSecret = useCallback(
    (value: string, requestId: string) => {
      void gw.respond('secret.respond', { request_id: requestId, value })
      clear()
    },
    [gw, clear]
  )

  // Local confirm — no RPC; just the local hook + clear.
  const onConfirm = useCallback(
    (ok: boolean) => {
      gw.onLocalConfirm(ok)
      clear()
    },
    [gw, clear]
  )

  switch (prompt.kind) {
    case 'clarify':
      return (
        <ClarifyPromptView
          focused
          onAnswer={a => onClarify(a, prompt.requestId)}
          onCancel={() => onClarify('', prompt.requestId)}
          req={prompt}
          t={t}
        />
      )

    case 'approval':
      return <ApprovalPromptView focused onChoice={onApproval} req={prompt} t={t} />

    case 'sudo':
      return (
        <MaskedPrompt
          focused
          icon="🔐"
          label="sudo password required"
          onSubmit={pw => onSudo(pw, prompt.requestId)}
          t={t}
        />
      )

    case 'secret':
      return (
        <MaskedPrompt
          focused
          icon="🔑"
          label={prompt.prompt || 'secret required'}
          onSubmit={v => onSecret(v, prompt.requestId)}
          sub={`for ${prompt.envVar}`}
          t={t}
        />
      )

    case 'confirm':
      return (
        <ConfirmPromptView
          focused
          onCancel={() => onConfirm(false)}
          onConfirm={() => onConfirm(true)}
          req={prompt}
          t={t}
        />
      )
  }
}
