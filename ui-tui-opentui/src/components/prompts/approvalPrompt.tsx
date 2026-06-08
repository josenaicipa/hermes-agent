// Native OpenTUI approval prompt — once / session / always / deny.
// Maps FROM ui-tui/src/components/prompts.tsx ApprovalPrompt (:63).
//
// Esc / Ctrl+C MUST send 'deny' (not just close) or the agent stays blocked.
// Number keys 1-4 quick-pick; ↑↓ + Enter via the native <select>.
import { useKeyboard } from '@opentui/react'
import React from 'react'

import type { ApprovalPrompt as ApprovalPromptState } from '../../model.ts'
import type { Theme } from '../../theme.ts'

export type ApprovalChoice = 'always' | 'deny' | 'once' | 'session'

const CHOICES: ApprovalChoice[] = ['once', 'session', 'always', 'deny']

const LABELS: Record<ApprovalChoice, string> = {
  always: 'Always allow',
  deny: 'Deny',
  once: 'Allow once',
  session: 'Allow this session'
}

const CMD_PREVIEW_LINES = 10

export function ApprovalPromptView({
  focused,
  onChoice,
  req,
  t
}: {
  focused: boolean
  onChoice: (choice: ApprovalChoice) => void
  req: ApprovalPromptState
  t: Theme
}) {
  useKeyboard(key => {
    if (!focused) {
      return
    }

    if (key.name === 'escape' || (key.ctrl && key.name === 'c')) {
      onChoice('deny')

      return
    }

    // Quick-pick by number 1..4.
    const n = parseInt(key.name ?? '', 10)

    if (n >= 1 && n <= CHOICES.length) {
      onChoice(CHOICES[n - 1]!)
    }
  })

  const rawLines = req.command.split('\n')
  const shown = rawLines.slice(0, CMD_PREVIEW_LINES)
  const overflow = rawLines.length - shown.length

  const options = CHOICES.map((c, i) => ({ description: '', name: `${i + 1}. ${LABELS[c]}`, value: c }))

  return (
    <box
      style={{
        borderColor: t.color.error,
        borderStyle: 'double',
        flexDirection: 'column',
        paddingLeft: 1,
        paddingRight: 1
      }}
    >
      <text fg={t.color.error}>
        <b>⚠ approval required · {req.description}</b>
      </text>

      <box style={{ flexDirection: 'column', paddingLeft: 1 }}>
        {shown.map((line, i) => (
          <text fg={t.color.text} key={i}>
            {line || ' '}
          </text>
        ))}
        {overflow > 0 ? <text fg={t.color.muted}>{`… +${overflow} more line${overflow === 1 ? '' : 's'}`}</text> : null}
      </box>

      <select
        focused={focused}
        onSelect={
          ((_index: number, option: { value?: unknown } | null) => {
            const choice = option?.value as ApprovalChoice | undefined

            if (choice) {
              onChoice(choice)
            }
          }) as never
        }
        options={options}
        showDescription={false}
        style={{ height: CHOICES.length, marginTop: 1 }}
      />

      <text fg={t.color.muted}>↑/↓ select · Enter confirm · 1-4 quick pick · Esc/Ctrl+C deny</text>
    </box>
  )
}
