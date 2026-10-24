'use client'

import type { ReactNode } from 'react'
import data from './chessrl-next-round.json'
import BarChart from './BarChart'
import LineChart, { DataTable, Legend, type Series } from './LineChart'

/** The charts of the "ChessRL, round two" post (data: docs/blog/make_charts.py in ChessRL). */

export { default as EncoderDiagram } from './EncoderDiagram'

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
      caption="Rating by hours of training, each run on one RTX 4090. Each point is 96 games against the reference opponents, with the network searching 100 simulations per move (±40 Elo). The transformer curve stops at 17 hours, where its TensorRT copy started to give wrong outputs (see the FP16 bug below).">
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
    <Figure title="400 or 800 simulations per move during training"
      caption="The simulations here are the search used to play training games (per saved move); ratings always use 100 per move. Per hour, the cheaper search reaches every level first. Per saved position, it is behind in the middle and level again by about 8M positions. The lighter line is the 800-simulation recipe run again with a different random seed: the gap between the two lines shows how much chance alone changes.">
      <Legend series={panels.hours} />
      <div className="grid gap-6 md:grid-cols-2">
        <div>
          <div className="mb-1 text-sm text-gray-600 dark:text-gray-400">Per hour of training</div>
          <LineChart series={panels.hours} xLabel="hours of training" yLabel="ladder Elo" xDomain={[0, 8.6]}
            yDomain={[0, 2000]} xStep={2} yStep={500} height={300} snap={0.35} xFormat={hours} endLabels={false} />
        </div>
        <div>
          <div className="mb-1 text-sm text-gray-600 dark:text-gray-400">Per self-play position</div>
          <LineChart series={panels.records_m} xLabel="positions saved (millions)" yLabel="ladder Elo"
            xDomain={[0, 12]} yDomain={[0, 2000]} xStep={2} yStep={500} height={300} snap={0.4}
            xFormat={(x) => `${x.toFixed(1)}M positions`} endLabels={false} />
        </div>
      </div>
      <DataTable series={panels.hours} xHeader="hours" xs={range(1, 8, 1)} snap={0.35} xFormat={hours} />
    </Figure>
  )
}

export function StepChart() {
  const series = data.perStep as Series[]
  const xs = Array.from(new Set(series.flatMap((s) => s.points.map((p) => p[0])))).sort((a, b) => a - b)
  const steps = (x: number) => `step ${x}k`
  return (
    <Figure title="Per training step, more search and more layers learn faster"
      caption="Rating by training step, always measured with 100 simulations per move. The three runs use the same batch size and trainer settings, so one step is the same amount of training in each. They differ in the search that played their games (simulations per saved move) and in the number of layers. Per hour the order is reversed: these runs do about 4,000, 2,100 and 640 steps per hour.">
      <Legend series={series} />
      <LineChart series={series} xLabel="training step (thousands)" yLabel="ladder Elo" xDomain={[0, 16]}
        yDomain={[0, 1800]} xStep={2} yStep={300} height={320} snap={0.3} xFormat={steps} />
      <DataTable series={series} xHeader="step" xs={xs} snap={0.01} xFormat={(x) => `${x}k`} />
    </Figure>
  )
}

export function ActivationChart() {
  const series = data.activation as Series[]
  return (
    <Figure title="Without weight decay, the values inside the network keep growing"
      caption="Largest value entering the transformer's last normalisation layer, measured on 512 puzzle positions, by training step. The largest number FP16 can hold is 65,504, so x² overflows once x passes 256.">
      <Legend series={series} />
      <LineChart series={series} xLabel="training step (thousands)" yLabel="largest value"
        xDomain={[0, 75]} yDomain={[0, 400]} xStep={10} yStep={100} height={300} snap={2.6}
        xFormat={(x) => `step ${x}k`} yFormat={(y) => y.toFixed(0)}
        refLine={{ y: 256, label: 'above 256, x² overflows in FP16' }} />
      <DataTable series={series} xHeader="step" xs={[10, 20, 30, 40, 45.5, 50, 70]} snap={0.1}
        xFormat={(x) => `${x}k`} yFormat={(y) => y.toFixed(0)} />
    </Figure>
  )
}

export function AblationChart() {
  return (
    <Figure title="Which change mattered: each one removed in turn"
      caption="Rating after 6M self-play positions. Two runs of the same recipe with different random seeds differ by about ±100 Elo at this point.">
      <BarChart rows={data.ablations} max={1400} slot={3} axisLabel="rating after 6M positions" />
    </Figure>
  )
}
