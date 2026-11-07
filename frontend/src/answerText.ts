/** Pure helpers that turn answer text into paragraphs and numbered steps. */
/** "Do this: 1. A 2. B 3. C" on one line becomes separate numbered lines. */
export function splitInlineSteps(line: string): string[] {
  const marks: { idx: number; n: number; len: number }[] = []
  for (const m of line.matchAll(/(?:^|\s)(\d{1,2})[.)]\s+(?=\S)/g)) {
    marks.push({ idx: m.index ?? 0, n: Number(m[1]), len: m[0].length })
  }
  if (marks.length < 2 || marks.some((mk, i) => mk.n !== i + 1)) return [line]
  const parts: string[] = []
  const lead = line.slice(0, marks[0].idx).trim()
  if (lead) parts.push(lead)
  marks.forEach((mk, i) => {
    const end = i + 1 < marks.length ? marks[i + 1].idx : line.length
    parts.push(`${mk.n}. ${line.slice(mk.idx + mk.len, end).trim()}`)
  })
  return parts
}

export type Block = { kind: 'p'; text: string } | { kind: 'ol'; items: string[] }

export function toBlocks(text: string): Block[] {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
    .flatMap(splitInlineSteps)
  const blocks: Block[] = []
  for (const line of lines) {
    const step = line.match(/^(\d+)[.)]\s+(.*)$/)
    const last = blocks[blocks.length - 1]
    if (step) {
      if (last && last.kind === 'ol') last.items.push(step[2])
      else blocks.push({ kind: 'ol', items: [step[2]] })
    } else if (/^\[S\d+\](\[S\d+\])*$/.test(line) && last) {
      // A citation marker alone on its line belongs to the block above it.
      if (last.kind === 'ol') last.items[last.items.length - 1] += ` ${line}`
      else last.text += ` ${line}`
    } else {
      blocks.push({ kind: 'p', text: line })
    }
  }
  return blocks
}
