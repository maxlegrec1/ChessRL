'use client'

import { useState } from 'react'
import './charts.css'

type Row = { label: string; value: number }

/** Horizontal bars in HTML (labels wrap on small screens), value at each bar's tip. */
export default function BarChart({
  rows,
  max,
  slot,
  axisLabel,
  format = (v: number) => v.toLocaleString('en-US'),
}: {
  rows: Row[]
  max: number
  slot: number
  axisLabel: string
  format?: (v: number) => string
}) {
  const [hover, setHover] = useState<number | null>(null)
  return (
    <div className="cr-chart relative">
      <div className="space-y-3">
        {rows.map((r, i) => (
          <div
            key={r.label}
            onPointerEnter={() => setHover(i)}
            onPointerLeave={() => setHover(null)}
          >
            <div className="mb-1 text-sm text-gray-600 dark:text-gray-400">{r.label}</div>
            <div className="relative flex items-center gap-2">
              <div
                className="h-[18px] rounded-r-[4px] transition-opacity"
                style={{
                  width: `${(100 * r.value) / max}%`,
                  background: `var(--cr-s${slot})`,
                  opacity: hover === null || hover === i ? 1 : 0.55,
                }}
              />
              <span className="text-sm font-semibold text-gray-900 tabular-nums dark:text-gray-100">
                {format(r.value)}
              </span>
              {hover === i && (
                <div
                  className="cr-tip"
                  style={{ left: `min(${(100 * r.value) / max}%, calc(100% - 12rem))`, top: -44 }}
                >
                  <b>{format(r.value)}</b> <span>{axisLabel}</span>
                  <div>{r.label}</div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-3 border-t border-gray-200 pt-1 text-xs text-gray-500 dark:border-gray-700 dark:text-gray-400">
        {axisLabel}
      </div>
    </div>
  )
}
