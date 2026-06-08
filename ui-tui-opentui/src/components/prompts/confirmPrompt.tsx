// Native OpenTUI confirm dialog — local (non-gateway) yes/no.
// Maps FROM ui-tui/src/components/prompts.tsx ConfirmPrompt (:195).
// Used by /new, /clear. Driven by a LOCAL callback, NOT an RPC.
//
// Y/N quick keys, ↑↓ + Enter via <select>, Esc cancels.
import { useKeyboard } from '@opentui/react'
import React from 'react'

import type { ConfirmPrompt as ConfirmPromptState } from '../../model.ts'
import type { Theme } from '../../theme.ts'

export function ConfirmPromptView({
  focused,
  onCancel,
  onConfirm,
  req,
  t
}: {
  focused: boolean
  onCancel: () => void
  onConfirm: () => void
  req: ConfirmPromptState
  t: Theme
}) {
  useKeyboard(key => {
    if (!focused) {
      return
    }

    const ch = (key.sequence ?? '').toLowerCase()

    if (key.name === 'escape' || (key.ctrl && key.name === 'c') || ch === 'n') {
      onCancel()

      return
    }

    if (ch === 'y') {
      onConfirm()
    }
  })

  const accent = req.danger ? t.color.error : t.color.label

  // index 0 = cancel (No), index 1 = confirm (Yes) — matches Ink ordering.
  const options = [
    { description: '', name: req.cancelLabel ?? 'No', value: 0 },
    { description: '', name: req.confirmLabel ?? 'Yes', value: 1 }
  ]

  return (
    <box
      style={{ borderColor: accent, borderStyle: 'double', flexDirection: 'column', paddingLeft: 1, paddingRight: 1 }}
    >
      <text fg={accent}>
        <b>
          {req.danger ? '⚠' : '?'} {req.title}
        </b>
      </text>

      {req.detail ? (
        <box style={{ paddingLeft: 1 }}>
          <text fg={t.color.text}>{req.detail}</text>
        </box>
      ) : null}

      <select
        focused={focused}
        onSelect={((index: number) => (index === 1 ? onConfirm() : onCancel())) as never}
        options={options}
        showDescription={false}
        style={{ height: 2, marginTop: 1 }}
      />

      <text fg={t.color.muted}>↑/↓ select · Enter confirm · Y/N quick · Esc cancel</text>
    </box>
  )
}
