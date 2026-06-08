// Native OpenTUI masked entry — shared by the sudo (🔐) and secret (🔑) prompts.
// Maps FROM ui-tui/src/components/maskedPrompt.tsx.
//
// We do NOT use OpenTUI's <input> here. <input> has no native mask option
// (verified against @opentui/core InputRenderableOptions — only
// value/placeholder/maxLength), and the "feed it stars via value + read real
// chars via onInput" trick is a feedback loop: onInput reports the masked value
// you set, so the real characters are lost. Instead we own a hidden buffer and
// capture raw keystrokes via useKeyboard, rendering '*' for each char. This is
// the robust path for masked input in OpenTUI.
import { useKeyboard } from '@opentui/react'
import React, { useState } from 'react'

import type { Theme } from '../../theme.ts'

export function MaskedPrompt({
  focused,
  icon,
  label,
  onSubmit,
  sub,
  t
}: {
  focused: boolean
  icon: string
  label: string
  onSubmit: (value: string) => void
  sub?: string
  t: Theme
}) {
  const [value, setValue] = useState('')

  useKeyboard(key => {
    if (!focused) {
      return
    }

    // Esc / Ctrl+C cancel → submit empty so the agent unblocks (deny/cancel).
    if (key.name === 'escape' || (key.ctrl && key.name === 'c')) {
      onSubmit('')

      return
    }

    if (key.name === 'return') {
      onSubmit(value)

      return
    }

    if (key.name === 'backspace') {
      setValue(v => v.slice(0, -1))

      return
    }

    // Printable single character (ignore other control/navigation keys).
    const ch = key.sequence ?? ''

    if (ch.length === 1 && !key.ctrl && !key.meta && ch >= ' ') {
      setValue(v => v + ch)
    }
  })

  const masked = value.length ? '*'.repeat(value.length) : ''
  const cursor = focused ? '▍' : ''

  return (
    <box style={{ flexDirection: 'column', paddingLeft: 1, paddingRight: 1 }}>
      <text fg={t.color.label}>
        <b>
          {icon} {label}
        </b>
      </text>

      {sub ? <text fg={t.color.muted}>{` ${sub}`}</text> : null}

      <box style={{ flexDirection: 'row' }}>
        <text fg={t.color.label}>{'> '}</text>
        <text fg={t.color.text}>{masked}</text>
        <text fg={t.color.accent}>{cursor}</text>
      </box>

      <text fg={t.color.muted}>Enter send · Esc/Ctrl+C cancel · masked input</text>
    </box>
  )
}
