'use client'

import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import './charts.css'

export type Series = {
  name: string
  slot: number // categorical palette slot, see charts.css
  dim?: boolean // same entity, de-emphasised (e.g. a second seed)
  dots?: boolean // mark every point (sparse measurements)
  points: [number, number][]
}

type Props = {
  series: Series[]
  xLabel: string
  yLabel: string
  xDomain: [number, number]
  yDomain: [number, number]
  xStep: number
  yStep: number
  height?: number
  xFormat?: (x: number) => string
  yFormat?: (y: number) => string
  refLine?: { y: number; label: string }
  xMarks?: { x: number; label: string }[] // dashed vertical lines for events (a fix, a change)
  band?: { y0: number; y1: number; label: string } // shaded reference range
  snap: number // a series' value shows in the tooltip if its point is within this x distance
  endLabels?: boolean
  onPick?: (x: number) => void // click (or Enter) on the plot: the nearest point's x
  marker?: number // a solid vertical line at this x (e.g. the position shown on a linked board)
}

const M = { top: 14, right: 56, bottom: 44, left: 56 }
const fmtInt = (v: number) => Math.round(v).toLocaleString('en-US')

export function Legend({ series }: { series: Series[] }) {
  return (
    <div className="mb-2 flex flex-wrap gap-x-5 gap-y-1 text-sm text-gray-600 dark:text-gray-400">
      {series.map((s) => (
        <span key={s.name} className="inline-flex items-center gap-2">
          <svg width="18" height="4" aria-hidden="true">
            <line x1="1" y1="2" x2="17" y2="2" stroke={`var(--cr-s${s.slot})`} strokeWidth="2.5"
              strokeLinecap="round" opacity={s.dim ? 0.5 : 1} />
          </svg>
          {s.name}
        </span>
      ))}
    </div>
  )
}

function ticks(lo: number, hi: number, step: number) {
  const out: number[] = []
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Math.round(t * 1e6) / 1e6 || 0) // no -0
  return out
}

function nearest(points: [number, number][], x: number, snap: number) {
  let best: [number, number] | null = null
  for (const p of points) if (Math.abs(p[0] - x) <= snap && (!best || Math.abs(p[0] - x) < Math.abs(best[0] - x))) best = p
  return best
}

export default function LineChart({
  series, xLabel, yLabel, xDomain, yDomain, xStep, yStep, height = 320,
  xFormat = (x) => String(x), yFormat = fmtInt, refLine, xMarks = [], band, snap, endLabels = true, onPick, marker,
}: Props) {
  const wrap = useRef<HTMLDivElement>(null)
  const clipId = `cr-clip-${useId().replace(/[^a-zA-Z0-9]/g, '')}`
  const [width, setWidth] = useState(640)
  const [hoverX, setHoverX] = useState<number | null>(null)
  const [pointer, setPointer] = useState<{ x: number; y: number } | null>(null)

  useEffect(() => {
    const el = wrap.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(280, Math.round(e.contentRect.width))))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const narrow = width < 480
  const m = { ...M, right: endLabels && !narrow ? M.right : 16 }
  const pw = width - m.left - m.right
  const ph = height - m.top - m.bottom
  const sx = (x: number) => m.left + ((x - xDomain[0]) / (xDomain[1] - xDomain[0])) * pw
  const sy = (y: number) => m.top + (1 - (y - yDomain[0]) / (yDomain[1] - yDomain[0])) * ph
  const xs = useMemo(
    () => Array.from(new Set(series.flatMap((s) => s.points.map((p) => p[0])))).sort((a, b) => a - b),
    [series]
  )

  // End labels: the value at each line's end; when two would collide, keep the later
  // series' (listed last = the one the chart is about), set its text past the other line's
  // end, and let the legend/tooltip carry the hidden value.
  const labels = useMemo(() => {
    if (!endLabels || narrow) return []
    const placed: { x: number; y: number; tx: number; text: string; slot: number }[] = []
    for (const s of [...series].reverse()) {
      if (s.dim || !s.points.length) continue
      const [x, y] = s.points[s.points.length - 1]
      const px = sx(x), py = sy(y)
      const hit = placed.find((l) => Math.abs(l.y - py) < 14 && Math.abs(l.x - px) < 70)
      if (hit) {
        hit.tx = Math.max(hit.tx, px + 8)
        continue
      }
      placed.push({ x: px, y: py, tx: px + 8, text: yFormat(y), slot: s.slot })
    }
    return placed
  }, [series, width, endLabels, narrow]) // eslint-disable-line react-hooks/exhaustive-deps

  const onMove = (e: PointerEvent<SVGRectElement>) => {
    const r = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
    const px = e.clientX - r.left
    const dataX = xDomain[0] + ((px - m.left) / pw) * (xDomain[1] - xDomain[0])
    let best = xs[0]
    for (const x of xs) if (Math.abs(x - dataX) < Math.abs(best - dataX)) best = x
    setHoverX(best)
    setPointer({ x: px, y: e.clientY - r.top })
  }
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && onPick && hoverX !== null) {
      onPick(hoverX)
      return
    }
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
    e.preventDefault()
    const i = hoverX === null ? xs.length - 1 : xs.indexOf(hoverX)
    const j = Math.min(xs.length - 1, Math.max(0, i + (e.key === 'ArrowRight' ? 1 : -1)))
    setHoverX(xs[j])
    setPointer({ x: sx(xs[j]), y: m.top + 20 })
  }

  const rows = hoverX === null ? [] : series.map((s) => ({ s, p: nearest(s.points, hoverX, snap) }))
  const tipLeft = pointer && pointer.x > width * 0.6

  return (
    <div ref={wrap} className="cr-chart relative w-full">
      <svg width={width} height={height} role="img" aria-label={`${yLabel} by ${xLabel}`}>
        <defs>
          {/* lines leaving the y range are cut at the plot edge (4 px slack keeps edge dots whole) */}
          <clipPath id={clipId}>
            <rect x={m.left - 4} y={m.top - 4} width={pw + 8} height={ph + 8} />
          </clipPath>
        </defs>
        {ticks(yDomain[0], yDomain[1], yStep).map((t) => (
          <g key={`y${t}`}>
            <line x1={m.left} x2={width - m.right} y1={sy(t)} y2={sy(t)} stroke="var(--cr-grid)" />
            <text x={m.left - 8} y={sy(t)} dy="0.32em" textAnchor="end">{yFormat(t)}</text>
          </g>
        ))}
        <line x1={m.left} x2={width - m.right} y1={m.top + ph} y2={m.top + ph} stroke="var(--cr-axis)" />
        {ticks(xDomain[0], xDomain[1], xStep).map((t) => (
          <text key={`x${t}`} x={sx(t)} y={m.top + ph + 18} textAnchor="middle">{t}</text>
        ))}
        <text x={m.left + pw / 2} y={height - 6} textAnchor="middle">{xLabel}</text>
        <text transform={`translate(14 ${m.top + ph / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
        {band && (
          <g>
            <rect x={m.left} y={sy(band.y1)} width={pw} height={sy(band.y0) - sy(band.y1)} fill="var(--cr-text)"
              fillOpacity={0.1} />
            <text x={width - m.right - 6} y={sy(band.y1) + 14} textAnchor="end">{band.label}</text>
          </g>
        )}
        {refLine && (
          <g>
            <line x1={m.left} x2={width - m.right} y1={sy(refLine.y)} y2={sy(refLine.y)} stroke="var(--cr-text)" />
            <text x={m.left + 6} y={sy(refLine.y) - 6}>{refLine.label}</text>
          </g>
        )}
        {xMarks.map((k, i) => (
          <g key={`m${k.x}`}>
            <line x1={sx(k.x)} x2={sx(k.x)} y1={m.top} y2={m.top + ph} stroke="var(--cr-text)" strokeDasharray="4 4" />
            {/* staggered so neighbouring labels don't collide on narrow charts */}
            <text x={sx(k.x) + 5} y={m.top + 11 + (i % 2) * 15}>{k.label}</text>
          </g>
        ))}
        {series.map((s) => (
          <g key={s.name} opacity={s.dim ? 0.5 : 1} clipPath={`url(#${clipId})`}>
            <path fill="none" stroke={`var(--cr-s${s.slot})`} strokeWidth={2} strokeLinejoin="round"
              strokeLinecap="round"
              d={s.points.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join('')} />
            {s.dots && s.points.map((p) => (
              <circle key={p[0]} cx={sx(p[0])} cy={sy(p[1])} r={3} fill={`var(--cr-s${s.slot})`} />
            ))}
          </g>
        ))}
        {labels.map((l) => (
          <g key={`e${l.slot}${l.text}`}>
            <circle cx={l.x} cy={l.y} r={4} fill={`var(--cr-s${l.slot})`} stroke="var(--cr-surface)" strokeWidth={2} />
            <text x={l.tx} y={l.y} dy="0.32em" className="cr-strong">{l.text}</text>
          </g>
        ))}
        {marker !== undefined && (
          <line x1={sx(marker)} x2={sx(marker)} y1={m.top} y2={m.top + ph} stroke="var(--cr-ink)" strokeWidth={1.5}
            pointerEvents="none" />
        )}
        {hoverX !== null && (
          <g pointerEvents="none">
            <line x1={sx(hoverX)} x2={sx(hoverX)} y1={m.top} y2={m.top + ph} stroke="var(--cr-axis)" />
            {rows.map(({ s, p }) => p && (
              <circle key={s.name} cx={sx(p[0])} cy={sy(p[1])} r={4} fill={`var(--cr-s${s.slot})`}
                stroke="var(--cr-surface)" strokeWidth={2} opacity={s.dim ? 0.6 : 1} />
            ))}
          </g>
        )}
        <rect className="cr-plot" x={m.left} y={m.top} width={pw} height={ph} fill="transparent" tabIndex={0}
          aria-label={onPick ? 'Chart: arrow keys read values, Enter or a click shows that position'
            : 'Chart: use the left and right arrow keys to read values'}
          style={onPick ? { cursor: 'pointer' } : undefined}
          onClick={onPick ? () => { if (hoverX !== null) onPick(hoverX) } : undefined}
          onPointerDown={onMove} onPointerMove={onMove} onPointerLeave={() => { setHoverX(null); setPointer(null) }}
          onFocus={() => { if (hoverX === null) { setHoverX(xs[xs.length - 1]); setPointer({ x: sx(xs[xs.length - 1]), y: m.top + 20 }) } }}
          onBlur={() => { setHoverX(null); setPointer(null) }} onKeyDown={onKey} />
      </svg>
      {hoverX !== null && pointer && (
        <div className="cr-tip" style={{
          top: Math.max(0, pointer.y - 12),
          left: tipLeft ? undefined : pointer.x + 14,
          right: tipLeft ? width - pointer.x + 14 : undefined,
        }}>
          <div className="mb-1">{xFormat(hoverX)}</div>
          {rows.map(({ s, p }) => (
            <div key={s.name} className="flex items-center gap-2 whitespace-nowrap">
              <svg width="14" height="4" aria-hidden="true">
                <line x1="1" y1="2" x2="13" y2="2" stroke={`var(--cr-s${s.slot})`} strokeWidth="2.5"
                  strokeLinecap="round" opacity={s.dim ? 0.5 : 1} />
              </svg>
              <b>{p ? yFormat(p[1]) : '–'}</b>
              <span>{s.name}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/** Collapsible data table: every value reachable without hovering. */
export function DataTable({ series, xHeader, xs, snap, xFormat = String, yFormat = fmtInt }: {
  series: Series[]; xHeader: string; xs: number[]; snap: number
  xFormat?: (x: number) => string; yFormat?: (y: number) => string
}) {
  return (
    <details className="mt-1 text-sm text-gray-500 dark:text-gray-400">
      <summary className="cursor-pointer select-none">Data</summary>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full text-right tabular-nums">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="py-1 pr-3 text-left font-medium">{xHeader}</th>
              {series.map((s) => <th key={s.name} className="px-2 py-1 font-medium">{s.name}</th>)}
            </tr>
          </thead>
          <tbody>
            {xs.map((x) => (
              <tr key={x} className="border-b border-gray-100 dark:border-gray-800">
                <td className="py-1 pr-3 text-left">{xFormat(x)}</td>
                {series.map((s) => {
                  const p = nearest(s.points, x, snap)
                  return <td key={s.name} className="px-2 py-1">{p ? yFormat(p[1]) : '–'}</td>
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  )
}
