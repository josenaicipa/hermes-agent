// Native OpenTUI clarify prompt.
// Maps FROM ui-tui/src/components/prompts.tsx ClarifyPrompt (:116).
//
// Two modes:
//  - choices present → a <select> of the choices plus a final "Other (type your
//    own)" row; picking it switches to a free-text <input>.
//  - choices null    → straight to the free-text <input>.
//
// Cancel semantics (matches Ink): Esc while typing with choices → back to the
// choice list; Esc otherwise → cancel with an EMPTY answer (so the agent
// unblocks). Ctrl+C always cancels with an empty answer.
import { useKeyboard } from '@opentui/react'
import React, { useState } from 'react'

import type { ClarifyPrompt as ClarifyPromptState } from '../../model.ts'
import type { Theme } from '../../theme.ts'

const OTHER_LABEL = 'Other (type your answer)'

export function ClarifyPromptView({
  focused,
  onAnswer,
  onCancel,
  req,
  t
}: {
  focused: boolean
  onAnswer: (answer: string) => void
  /** Cancel = answer with '' (the gateway treats empty as cancel). */
  onCancel: () => void
  req: ClarifyPromptState
  t: Theme
}) {
  const choices = req.choices ?? []
  const hasChoices = choices.length > 0
  // Start in typing mode when there are no choices to pick from.
  const [typing, setTyping] = useState(!hasChoices)
  const [draft, setDraft] = useState('')

  useKeyboard(key => {
    if (!focused) {
      return
    }

    // Ctrl+C always cancels.
    if (key.ctrl && key.name === 'c') {
      onCancel()

      return
    }

    if (key.name === 'escape') {
      if (typing && hasChoices) {
        setTyping(false)
      } else {
        onCancel()
      }
    }
  })

  const heading = (
    <text>
      <b>
        <span fg={t.color.accent}>ask</span>
        <span fg={t.color.text}> {req.question}</span>
      </b>
    </text>
  )

  // Free-text mode (either no choices, or user picked "Other").
  if (typing) {
    return (
      <box style={{ flexDirection: 'column', paddingLeft: 1, paddingRight: 1 }}>
        {heading}
        <box style={{ flexDirection: 'row' }}>
          <text fg={t.color.label}>{'> '}</text>
          <input
            cursorColor={t.color.accent}
            focused={focused}
            onInput={((v: string) => setDraft(typeof v === 'string' ? v : '')) as never}
            onSubmit={
              ((v: string) => {
                const text = (typeof v === 'string' ? v : draft).trim()
                onAnswer(text)
              }) as never
            }
            placeholder="type your answer…"
            style={{ flexGrow: 1 }}
            textColor={t.color.text}
          />
        </box>
        <text fg={t.color.muted}>{`Enter send · Esc ${hasChoices ? 'back' : 'cancel'} · Ctrl+C cancel`}</text>
      </box>
    )
  }

  // Choice-list mode: choices + an "Other" row at the end.
  const options = [...choices, OTHER_LABEL].map((c, i) => ({ description: '', name: `${i + 1}. ${c}`, value: i }))

  return (
    <box style={{ flexDirection: 'column', paddingLeft: 1, paddingRight: 1 }}>
      {heading}
      <select
        focused={focused}
        onSelect={
          ((index: number) => {
            if (index === choices.length) {
              // "Other" row → switch to free-text.
              setTyping(true)
            } else {
              const picked = choices[index]

              if (typeof picked === 'string') {
                onAnswer(picked)
              }
            }
          }) as never
        }
        options={options}
        showDescription={false}
        style={{ height: options.length, marginTop: 1 }}
      />
      <text
        fg={t.color.muted}
      >{`↑/↓ select · Enter confirm · 1-${choices.length} quick pick · Esc/Ctrl+C cancel`}</text>
    </box>
  )
}
