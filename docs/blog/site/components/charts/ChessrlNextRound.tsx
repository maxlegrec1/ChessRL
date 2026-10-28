'use client'

import data from './chessrl-next-round.json'
import boards from './chessrl-boards.json'
import ChessBoard, { type BoardLine } from './ChessBoard'
import Figure from './Figure'
import LineChart, { Legend, type Series } from './LineChart'

/** The charts of the "ChessRL, continued" post (data: docs/blog/make_charts.py and
 *  docs/blog/make_boards.py in ChessRL). */

export { default as EncoderDiagram } from './EncoderDiagram'

const hours = (x: number) => `${x.toFixed(1)} h`
const percent = (y: number) => `${Math.round(100 * y)}%`

export function SpeedupChart() {
  const series = data.speedup as Series[]
  return (
    <Figure title="Same GPU, same 24 hours"
      caption="Each run on one RTX 4090 with 32 CPU cores, rated about every 30 minutes with the network searching 100 simulations per move (about ±45 Elo). Both runs moved to new machines once, after about 5.5 hours; the time that took is not counted.">
      <Legend series={series} />
      <LineChart series={series} xLabel="hours of training" yLabel="Elo" xDomain={[0, 24.5]}
        yDomain={[-100, 2500]} xStep={4} yStep={500} height={360} snap={0.3} xFormat={hours} />
    </Figure>
  )
}

export function TacticsChart() {
  const series = data.puzzles as Series[]
  return (
    <Figure title="Tactics: share of 1,000 Lichess puzzles solved"
      caption="A puzzle counts as solved only if the network plays every move of the solution, with 100 simulations per move. Same two runs, measured with each rating.">
      <Legend series={series} />
      <LineChart series={series} xLabel="hours of training" yLabel="puzzles solved" xDomain={[0, 24.5]}
        yDomain={[0, 0.8]} xStep={4} yStep={0.2} height={300} snap={0.3} xFormat={hours} yFormat={percent} />
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

export function ShowcaseGame() {
  return (
    <Figure title="No search at all: the network against Stockfish limited to 1400 Elo"
      caption="The network plays Black and picks the move it finds most likely, without any search. Stockfish 16 is limited to 1400 Elo, at 100 ms per move. Pieces by Colin M. L. Burnett.">
      <ChessBoard line={boards.game as unknown as BoardLine} />
    </Figure>
  )
}

export function PuzzleBoard({ id }: { id: keyof typeof boards.puzzles }) {
  const line = boards.puzzles[id] as unknown as BoardLine & { meta: { rating: number } }
  return (
    <Figure title={`A puzzle rated ${line.meta.rating.toLocaleString('en-US')}, solved without search`}
      caption={`Lichess puzzle ${id}. Every move of the solution is the network's first choice, with no search. Pieces by Colin M. L. Burnett.`}>
      <ChessBoard line={line} />
    </Figure>
  )
}
