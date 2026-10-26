'use client'

import data from './chessrl-next-round.json'
import BarChart from './BarChart'
import Figure from './Figure'
import LineChart, { DataTable, Legend, type Series } from './LineChart'

/** The charts of the "ChessRL, round two" post (data: docs/blog/make_charts.py in ChessRL). */

export { default as EncoderDiagram } from './EncoderDiagram'

const hours = (x: number) => `${x.toFixed(1)} h`
const stepK = (x: number) => `${x}k`
// Every rated network is a kept checkpoint (one per 10,000 training steps): tables list them by step.
const allXs = (series: Series[]) =>
  Array.from(new Set(series.flatMap((s) => s.points.map((p) => p[0])))).sort((a, b) => a - b)

export function SpeedupChart() {
  const { hours: series, steps_k: bySteps } = data.speedup as unknown as Record<'hours' | 'steps_k', Series[]>
  return (
    <Figure title="Same GPU, four recipes"
      caption="Rating by hours of training, each run on one RTX 4090. Each point is a network kept during the run (one every 10,000 training steps), rated against the reference opponents from the balanced openings, with the network searching 100 simulations per move (about ±45 Elo). The original code is the first post's code, run again here. The transformer curve stops at 17 hours, where its TensorRT copy started to give wrong outputs (see the FP16 bug below).">
      <Legend series={series} />
      <LineChart series={series} xLabel="hours of training" yLabel="ladder Elo" xDomain={[0, 22]}
        yDomain={[0, 2500]} xStep={2} yStep={500} height={360} snap={1.2} xFormat={hours} />
      <DataTable series={bySteps} xHeader="training step" xs={allXs(bySteps)} snap={0.01} xFormat={stepK} />
    </Figure>
  )
}

export function SearchBudgetChart() {
  const panels = data.searchBudget as unknown as Record<'hours' | 'records_m' | 'steps_k', Series[]>
  return (
    <Figure title="400 or 800 simulations per move during training"
      caption="The simulations here are the search used to play training games (per saved move); ratings always use 100 per move. Per hour, the cheaper search is ahead for about the first 12 hours, then the 800-simulation run passes it. Per saved position, the cheaper search is behind except around 8M positions. The lighter line is the 800-simulation recipe run again with a different random seed: the gap between the two shows how much chance alone changes.">
      <Legend series={panels.hours} />
      <div className="grid gap-6 md:grid-cols-2">
        <div>
          <div className="mb-1 text-sm text-gray-600 dark:text-gray-400">Per hour of training</div>
          <LineChart series={panels.hours} xLabel="hours of training" yLabel="ladder Elo" xDomain={[0, 18]}
            yDomain={[1250, 2400]} xStep={3} yStep={250} height={300} snap={1.2} xFormat={hours} endLabels={false} />
        </div>
        <div>
          <div className="mb-1 text-sm text-gray-600 dark:text-gray-400">Per self-play position</div>
          <LineChart series={panels.records_m} xLabel="positions saved (millions)" yLabel="ladder Elo"
            xDomain={[0, 26]} yDomain={[1250, 2400]} xStep={4} yStep={250} height={300} snap={1.5}
            xFormat={(x) => `${x.toFixed(1)}M positions`} endLabels={false} />
        </div>
      </div>
      <DataTable series={panels.steps_k} xHeader="training step" xs={allXs(panels.steps_k)} snap={0.01} xFormat={stepK} />
    </Figure>
  )
}

export function StepChart() {
  const series = data.perStep as Series[]
  return (
    <Figure title="Per training step, more search stays ahead; more layers help early"
      caption="Rating by training step, always measured with 100 simulations per move. The three runs use the same batch size and trainer settings, so one step is the same amount of training in each. They differ in the search that played their games (simulations per saved move) and in the number of layers. Per hour the order changes: these runs did about 3,700, 2,300 and 1,700 steps per hour.">
      <Legend series={series} />
      <LineChart series={series} xLabel="training step (thousands)" yLabel="ladder Elo" xDomain={[0, 62]}
        yDomain={[1250, 2400]} xStep={10} yStep={250} height={320} snap={2} xFormat={(x) => `step ${x}k`} />
      <DataTable series={series} xHeader="training step" xs={allXs(series)} snap={0.01} xFormat={stepK} />
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
      caption="Rating after 20,000 training steps (8.2M self-play positions), from the balanced openings. Each rating is about ±45 Elo.">
      <BarChart rows={data.ablations} max={1600} slot={3} axisLabel="rating after 20,000 steps" />
    </Figure>
  )
}
