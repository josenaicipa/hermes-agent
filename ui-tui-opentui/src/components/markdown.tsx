// Native OpenTUI inline rich-text: render a minimal markdown subset as
// OpenTUI <span> children inside a <text>. This is the NATIVE answer to the
// shim's "nested <Text> style cascade" showstopper — we author spans directly.
//
// Supported inline: **bold**, *italic*, `code`. Block: paragraphs split on \n\n,
// "- " bullets. Faithful enough for Phase 0; the full renderer maps from
// ui-tui/src/components/markdown.tsx in Phase 4.
import React from 'react'

import type { Theme } from '../theme.ts'

type Seg = { text: string; bold?: boolean; italic?: boolean; code?: boolean }

const INLINE = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g

function parseInline(line: string): Seg[] {
  const out: Seg[] = []
  let last = 0

  for (const m of line.matchAll(INLINE)) {
    const idx = m.index ?? 0

    if (idx > last) {
      out.push({ text: line.slice(last, idx) })
    }

    const tok = m[0]

    if (tok.startsWith('**')) {
      out.push({ text: tok.slice(2, -2), bold: true })
    } else if (tok.startsWith('*')) {
      out.push({ text: tok.slice(1, -1), italic: true })
    } else if (tok.startsWith('`')) {
      out.push({ text: tok.slice(1, -1), code: true })
    }

    last = idx + tok.length
  }

  if (last < line.length) {
    out.push({ text: line.slice(last) })
  }

  return out
}

function Inline({ segs, t }: { segs: Seg[]; t: Theme }) {
  return (
    <>
      {segs.map((s, i) => {
        if (s.code) {
          return (
            <span bg="#2A2A2A" fg={t.color.accent} key={i}>
              {s.text}
            </span>
          )
        }

        const inner = (
          <span fg={t.color.text} key={i}>
            {s.text}
          </span>
        )

        if (s.bold && s.italic) {
          return (
            <b key={i}>
              <i>{inner}</i>
            </b>
          )
        }

        if (s.bold) {
          return <b key={i}>{inner}</b>
        }

        if (s.italic) {
          return <i key={i}>{inner}</i>
        }

        return inner
      })}
    </>
  )
}

export function Markdown({ text, t, width }: { text: string; t: Theme; width?: number }) {
  const blocks = text.split('\n\n')

  return (
    <box style={{ flexDirection: 'column', width }}>
      {blocks.map((block, bi) => {
        const lines = block.split('\n')
        const isList = lines.every(l => l.trim().startsWith('- ') || l.trim() === '')

        if (isList) {
          return (
            <box key={bi} style={{ flexDirection: 'column' }}>
              {lines
                .filter(l => l.trim())
                .map((l, li) => (
                  <text key={li}>
                    <span fg={t.color.accent}>{'  • '}</span>
                    <Inline segs={parseInline(l.trim().replace(/^- /, ''))} t={t} />
                  </text>
                ))}
            </box>
          )
        }

        return (
          <text key={bi}>
            <Inline segs={parseInline(lines.join(' '))} t={t} />
          </text>
        )
      })}
    </box>
  )
}
