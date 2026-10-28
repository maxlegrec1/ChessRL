'use client'

import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import PIECES from './chessPieces'
import './charts.css'

/** A game or a puzzle line on a board you can step through. The positions are computed in
 *  advance (docs/blog/make_boards.py in ChessRL), so no chess logic runs here. */

export type Ply = {
  san: string
  fen: string // after the move
  from: string
  to: string
  // Puzzle solver moves: the move's probability for the network alone and its share of the search.
  net?: { alone: number; search: number; favourite?: string; favouriteAlone?: number }
}
export type BoardLine = {
  start: string
  plies: Ply[]
  orientation: 'white' | 'black'
  intro?: string // puzzles: shown before the first move
  sims?: number
  white?: string // games: the players and the result
  black?: string
  result?: string
}

const FILES = 'abcdefgh'
const LIGHT = '#f0d9b5'
const DARK = '#b58863'
const LAST = 'rgba(155, 199, 0, 0.41)'

function board(fen: string): (string | null)[] {
  const out: (string | null)[] = [] // a8, b8, ..., h1
  for (const row of fen.split(' ')[0].split('/'))
    for (const ch of row) {
      if (/\d/.test(ch)) for (let k = 0; k < Number(ch); k++) out.push(null)
      else out.push(ch)
    }
  return out
}

// Move number and side to move, from the FEN before a ply.
function mover(fen: string): { n: number; white: boolean } {
  const f = fen.split(' ')
  return { n: Number(f[5]), white: f[1] === 'w' }
}

const pct = (x: number) => `${Math.round(100 * x)}%`

function Bar({ label, value, strong }: { label: string; value: number; strong?: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-32 shrink-0 text-gray-600 dark:text-gray-400">{label}</span>
      <span className="h-2.5 flex-1 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
        <span className="block h-full rounded-full" style={{ width: pct(value),
          background: strong ? 'var(--cr-s1)' : 'var(--cr-axis)' }} />
      </span>
      <span className="w-10 text-right tabular-nums text-gray-900 dark:text-gray-100">{pct(value)}</span>
    </div>
  )
}

function Icon({ d }: { d: string }) {
  return (
    <svg viewBox="0 0 16 16" className="h-4 w-4" aria-hidden="true">
      <path d={d} fill="currentColor" />
    </svg>
  )
}

export default function ChessBoard({ line }: { line: BoardLine }) {
  const [i, setI] = useState(0) // plies played
  const n = line.plies.length
  const listRef = useRef<HTMLDivElement>(null)
  const fenBefore = (k: number) => (k === 0 ? line.start : line.plies[k - 1].fen)
  const last = i > 0 ? line.plies[i - 1] : null
  const flip = line.orientation === 'black'
  const sq = board(fenBefore(i))
  const go = (k: number) => setI(Math.max(0, Math.min(n, k)))

  useEffect(() => {
    listRef.current?.querySelector('[data-current="true"]')?.scrollIntoView({ block: 'nearest' })
  }, [i])

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const k = { ArrowLeft: i - 1, ArrowRight: i + 1, Home: 0, End: n }[e.key]
    if (k !== undefined) {
      e.preventDefault()
      go(k)
    }
  }
  const place = (idx: number) => {
    const file = idx % 8
    const rank = 7 - Math.floor(idx / 8) // 0 = rank 1
    return { x: flip ? 7 - file : file, y: flip ? rank : 7 - rank, name: `${FILES[file]}${rank + 1}` }
  }

  // Move list rows: [number, white ply index or null, black ply index or null].
  const rows: [number, number | null, number | null][] = []
  line.plies.forEach((_, k) => {
    const m = mover(fenBefore(k))
    if (m.white || rows.length === 0) rows.push([m.n, m.white ? k : null, m.white ? null : k])
    else rows[rows.length - 1][2] = k
  })
  const cell = (k: number | null) =>
    k === null ? (
      <span className="px-2 text-gray-400">…</span>
    ) : (
      <button type="button" onClick={() => go(k + 1)} data-current={k + 1 === i}
        className={`w-full rounded px-2 py-0.5 text-left ${k + 1 === i
          ? 'bg-gray-900 font-semibold text-white dark:bg-gray-100 dark:text-gray-900'
          : 'text-gray-800 hover:bg-gray-100 dark:text-gray-200 dark:hover:bg-gray-800'}`}>
        {line.plies[k].san}
      </button>
    )

  // Puzzle panel: the latest solver move played so far.
  const solved = line.plies.slice(0, i).map((p, k) => ({ p, k })).filter(({ p }) => p.net).pop()
  const btn = 'flex h-9 w-11 items-center justify-center rounded-md border border-gray-300 text-gray-700 ' +
    'hover:bg-gray-100 disabled:opacity-30 dark:border-gray-600 dark:text-gray-200 dark:hover:bg-gray-800'

  return (
    <div className="flex flex-col gap-5 md:flex-row md:items-start">
      <div className="w-full shrink-0 md:w-[26rem]">
        <div tabIndex={0} onKeyDown={onKey}
          className="overflow-hidden rounded-md shadow-sm ring-1 ring-black/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
          aria-label="Chess board; the arrow keys step through the moves">
          <svg viewBox="0 0 8 8" className="block w-full" role="img">
            {sq.map((_, idx) => {
              const { x, y, name } = place(idx)
              const hl = last && (last.from === name || last.to === name)
              return (
                <g key={name}>
                  <rect x={x} y={y} width={1} height={1} fill={(x + y) % 2 ? DARK : LIGHT} />
                  {hl && <rect x={x} y={y} width={1} height={1} fill={LAST} />}
                </g>
              )
            })}
            {[...Array(8)].map((_, k) => (
              <g key={k} style={{ fontSize: 0.2, fontWeight: 600 }}>
                <text x={k + 0.95} y={7.93} textAnchor="end" style={{ fill: (k + 7) % 2 ? LIGHT : DARK, fontSize: 0.2 }}>
                  {FILES[flip ? 7 - k : k]}
                </text>
                <text x={0.05} y={k + 0.22} style={{ fill: k % 2 ? LIGHT : DARK, fontSize: 0.2 }}>
                  {flip ? k + 1 : 8 - k}
                </text>
              </g>
            ))}
            {sq.map((p, idx) => {
              if (!p) return null
              const { x, y } = place(idx)
              return <svg key={idx} x={x} y={y} width={1} height={1} viewBox="0 0 45 45"
                dangerouslySetInnerHTML={{ __html: PIECES[p] }} />
            })}
          </svg>
        </div>
        <div className="mt-3 flex items-center justify-center gap-2">
          <button type="button" className={btn} onClick={() => go(0)} disabled={i === 0} aria-label="first position">
            <Icon d="M3 3h2v10H3zM13 3v10L6 8z" />
          </button>
          <button type="button" className={btn} onClick={() => go(i - 1)} disabled={i === 0} aria-label="previous move">
            <Icon d="M11 3v10L4 8z" />
          </button>
          <button type="button" className={btn} onClick={() => go(i + 1)} disabled={i === n} aria-label="next move">
            <Icon d="M5 3v10l7-5z" />
          </button>
          <button type="button" className={btn} onClick={() => go(n)} disabled={i === n} aria-label="last position">
            <Icon d="M11 3h2v10h-2zM3 3v10l7-5z" />
          </button>
        </div>
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-4 text-sm">
        {line.white && (
          <div className="space-y-1">
            <div className="flex items-center gap-2"><span className="h-3 w-3 rounded-sm border border-gray-400 bg-white" />
              <span className="text-gray-900 dark:text-gray-100">{line.white}</span></div>
            <div className="flex items-center gap-2"><span className="h-3 w-3 rounded-sm bg-gray-900 dark:border dark:border-gray-500" />
              <span className="text-gray-900 dark:text-gray-100">{line.black}</span></div>
          </div>
        )}
        {line.intro && (
          <div className="rounded-md bg-gray-50 p-3 dark:bg-gray-900">
            {!solved ? (
              <p className="text-gray-800 dark:text-gray-200">{line.intro} Press ▶ to see the solution.</p>
            ) : (
              <div className="space-y-2">
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  {(() => {
                    const m = mover(fenBefore(solved.k))
                    return `${m.n}${m.white ? '.' : '…'} ${solved.p.san}`
                  })()}
                </p>
                {line.sims === 0 ? (
                  <>
                    <Bar label="network's probability" value={solved.p.net!.alone} strong />
                    <p className="text-gray-600 dark:text-gray-400">
                      {solved.p.net!.favourite
                        ? `Its first choice was ${solved.p.net!.favourite} (${pct(solved.p.net!.favouriteAlone!)}).`
                        : 'Its first choice, with no search.'}
                    </p>
                  </>
                ) : (
                  <>
                    <Bar label="network alone" value={solved.p.net!.alone} />
                    <Bar label={`after ${line.sims ?? 800} simulations`} value={solved.p.net!.search} strong />
                    {solved.p.net!.favourite && (
                      <p className="text-gray-600 dark:text-gray-400">
                        Without the search, the network would have played {solved.p.net!.favourite} ({pct(solved.p.net!.favouriteAlone!)}).
                      </p>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        )}
        <div ref={listRef} className="max-h-72 overflow-y-auto rounded-md border border-gray-200 dark:border-gray-700">
          <div className="grid grid-cols-[2.5rem_1fr_1fr] gap-y-0.5 p-1 tabular-nums">
            {rows.map(([num, w, b]) => (
              <div key={num} className="contents">
                <span className="px-2 py-0.5 text-right text-gray-400">{num}.</span>
                {cell(w)}
                {cell(b)}
              </div>
            ))}
          </div>
        </div>
        {line.result && i === n && (
          <p className="text-gray-800 dark:text-gray-200">
            Result: {line.result}{line.plies[n - 1].san.endsWith('#') ? ', checkmate' : ''}.
          </p>
        )}
      </div>
    </div>
  )
}
