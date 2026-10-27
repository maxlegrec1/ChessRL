'use client'

import data from './chessrl-next-round.json'
import boards from './chessrl-boards.json'
import BarChart from './BarChart'
import ChessBoard, { type BoardLine } from './ChessBoard'
import Figure from './Figure'
import LineChart, { Legend, type Series } from './LineChart'

/** The charts of the "ChessRL: continued improvements" post (data: docs/blog/make_charts.py and
 *  docs/blog/make_boards.py in ChessRL). */

export { default as EncoderDiagram } from './EncoderDiagram'

const hours = (x: number) => `${x.toFixed(1)} h`
const percent = (y: number) => `${Math.round(100 * y)}%`

export function SpeedupChart() {
  const series = data.speedup as Series[]
  return (
    <Figure title="Same GPU, four recipes"
      caption="Each run on one RTX 4090. Each point is a network kept during the run, rated with 100 simulations per move (about ±45 Elo); the first point is the untrained network. The original code is the first post's code, run again here.">
      <Legend series={series} />
      <LineChart series={series} xLabel="hours of training" yLabel="Elo" xDomain={[0, 22]}
        yDomain={[0, 2500]} xStep={2} yStep={500} height={360} snap={1.2} xFormat={hours} />
    </Figure>
  )
}

export function TacticsChart() {
  const series = data.puzzles as Series[]
  return (
    <Figure title="Tactics: share of 1,000 Lichess puzzles solved"
      caption="A puzzle counts as solved only if the network plays every move of the solution, with 100 simulations per move. Measured every 30 minutes during the runs. The first post did not measure puzzles; its code, run again here, is the bottom line.">
      <Legend series={series} />
      <LineChart series={series} xLabel="hours of training" yLabel="puzzles solved" xDomain={[0, 22]}
        yDomain={[0, 0.8]} xStep={2} yStep={0.2} height={320} snap={0.35} xFormat={hours} yFormat={percent} />
    </Figure>
  )
}

export function SearchBudgetChart() {
  const series = data.searchBudget as Series[]
  return (
    <Figure title="400 or 800 simulations per move during training"
      caption="The simulations here are the search used to play training games; ratings always use 100 per move. The lighter line is the 800-simulation recipe run again with a different random seed.">
      <Legend series={series} />
      <LineChart series={series} xLabel="hours of training" yLabel="Elo" xDomain={[0, 18]}
        yDomain={[0, 2500]} xStep={2} yStep={500} height={320} snap={1.2} xFormat={hours} />
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
    </Figure>
  )
}

export function AblationChart() {
  return (
    <Figure title="Which change mattered: each one removed in turn"
      caption="Rating after 20,000 training steps, about ±45 Elo each.">
      <BarChart rows={data.ablations} max={1600} slot={3} axisLabel="Elo after 20,000 steps" />
    </Figure>
  )
}

export function ShowcaseGame() {
  return (
    <Figure title="A game against full-strength Stockfish, limited to 1,000 positions per move"
      caption="The network searches 800 simulations per move. It won all 10 games of this match, 5 with each colour. Pieces by Colin M. L. Burnett.">
      <ChessBoard line={boards.game as unknown as BoardLine} />
    </Figure>
  )
}

export function PuzzleBoard() {
  return (
    <Figure title="A puzzle rated 2,900"
      caption="With 800 simulations per move, the network solved 35 of the 63 puzzles rated 2,500 or more in our set. Pieces by Colin M. L. Burnett.">
      <ChessBoard line={boards.puzzle as unknown as BoardLine} />
    </Figure>
  )
}
