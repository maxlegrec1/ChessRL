'use client'

import type { ReactNode } from 'react'
import data from './chessrl-next-round.json'
import BarChart from './BarChart'
import LineChart, { DataTable, Legend, type Series } from './LineChart'

/** The four charts of the "ChessRL, round two" post (data: docs/blog/make_charts.py in ChessRL). */

function Figure({ title, caption, children }: { title: string; caption: string; children: ReactNode }) {
  return (
    <figure className="cr-chart not-prose my-8">
      <div className="mb-2 text-base font-semibold text-gray-900 dark:text-gray-100">{title}</div>
      {children}
      <figcaption className="mt-2 text-sm text-gray-500 dark:text-gray-400">{caption}</figcaption>
    </figure>
  )
}

const hours = (x: number) => `${x.toFixed(1)} h`
const range = (lo: number, hi: number, step: number) =>
  Array.from({ length: Math.floor((hi - lo) / step) + 1 }, (_, i) => lo + i * step)

export function SpeedupChart() {
  const series = data.speedup as Series[]
  return (
    <Figure title="Same GPU, four recipes"
      caption="Ladder Elo against calibrated Stockfish anchors (random play = 0) by hours of training, each run on one RTX 4090. Each rating: 96 games at 100 simulations per move, ±40 Elo. The encoder curve stops at 17 h, where its inference engine started to diverge (see below).">
      <Legend series={series} />
      <LineChart series={series} xLabel="hours of training" yLabel="ladder Elo" xDomain={[0, 22]}
        yDomain={[-100, 2100]} xStep={2} yStep={250} height={360} snap={0.35} xFormat={hours} />
      <DataTable series={series} xHeader="hours" xs={range(1, 21, 1)} snap={0.35} xFormat={hours} />
    </Figure>
  )
}

export function SearchBudgetChart() {
  const panels = data.searchBudget as unknown as Record<'hours' | 'records_m', Series[]>
  return (
    <Figure title="400 vs 800 simulations per move"
      caption="Per hour, the cheaper search gets everywhere first; per self-play position it is behind mid-run and level again by about 8M positions. The lighter line is a second 800-simulation seed: the spread between seeds is the noise to compare against.">
      <Legend series={panels.hours} />
      <div className="grid gap-6 md:grid-cols-2">
        <div>
          <div className="mb-1 text-sm text-gray-600 dark:text-gray-400">Per hour of training</div>
          <LineChart series={panels.hours} xLabel="hours of training" yLabel="ladder Elo" xDomain={[0, 8.6]}
            yDomain={[0, 2000]} xStep={2} yStep={500} height={300} snap={0.35} xFormat={hours} endLabels={false} />
        </div>
        <div>
          <div className="mb-1 text-sm text-gray-600 dark:text-gray-400">Per self-play position</div>
          <LineChart series={panels.records_m} xLabel="positions recorded (millions)" yLabel="ladder Elo"
            xDomain={[0, 12]} yDomain={[0, 2000]} xStep={2} yStep={500} height={300} snap={0.4}
            xFormat={(x) => `${x.toFixed(1)}M positions`} endLabels={false} />
        </div>
      </div>
      <DataTable series={panels.hours} xHeader="hours" xs={range(1, 8, 1)} snap={0.35} xFormat={hours} />
    </Figure>
  )
}

export function ActivationChart() {
  const series = data.activation as Series[]
  return (
    <Figure title="Without weight decay the residual stream keeps growing"
      caption="Largest activation entering the encoder's final normalisation, on 512 puzzle positions, by training step. In FP16, x² overflows once |x| passes 256.">
      <Legend series={series} />
      <LineChart series={series} xLabel="training step (thousands)" yLabel="largest activation"
        xDomain={[0, 75]} yDomain={[0, 400]} xStep={10} yStep={100} height={300} snap={2.6}
        xFormat={(x) => `step ${x}k`} yFormat={(y) => y.toFixed(0)}
        refLine={{ y: 256, label: 'x² overflows FP16 above |x| = 256' }} />
      <DataTable series={series} xHeader="step" xs={[10, 20, 30, 40, 45.5, 50, 70]} snap={0.1}
        xFormat={(x) => `${x}k`} yFormat={(y) => y.toFixed(0)} />
    </Figure>
  )
}

export function AblationChart() {
  return (
    <Figure title="Which fix mattered: remove one at a time"
      caption="Ladder Elo after 6M self-play positions. Seed-to-seed noise at this point is about ±100.">
      <BarChart rows={data.ablations} max={1400} slot={3} axisLabel="ladder Elo at 6M positions" />
    </Figure>
  )
}
