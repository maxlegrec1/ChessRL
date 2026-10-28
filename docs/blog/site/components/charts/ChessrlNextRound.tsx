'use client'

import { useState } from 'react'
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
    <Figure
      title="Same GPU, same 24 hours"
      caption="Each run on one RTX 4090 with 32 CPU cores, rated about every 30 minutes with the network searching 100 simulations per move (about ±45 Elo). Both runs moved to new machines once, after about 5.5 hours; the time that took is not counted."
    >
      <Legend series={series} />
      <LineChart
        series={series}
        xLabel="hours of training"
        yLabel="Elo"
        xDomain={[0, 24.5]}
        yDomain={[-100, 2500]}
        xStep={4}
        yStep={500}
        height={360}
        snap={0.3}
        xFormat={hours}
      />
    </Figure>
  )
}

export function TacticsChart() {
  const series = data.puzzles as Series[]
  return (
    <Figure
      title="Tactics: share of 1,000 Lichess puzzles solved"
      caption="A puzzle counts as solved only if the network plays every move of the solution, with 100 simulations per move. Same two runs, measured with each rating."
    >
      <Legend series={series} />
      <LineChart
        series={series}
        xLabel="hours of training"
        yLabel="puzzles solved"
        xDomain={[0, 24.5]}
        yDomain={[0, 0.8]}
        xStep={4}
        yStep={0.2}
        height={300}
        snap={0.3}
        xFormat={hours}
        yFormat={percent}
      />
    </Figure>
  )
}

export function ActivationChart() {
  const series = data.activation as Series[]
  return (
    <Figure
      title="Without weight decay, the values inside the network keep growing"
      caption="Largest value entering the transformer's last normalisation layer, measured on 512 puzzle positions, by training step. The largest number FP16 can hold is 65,504, so x² overflows once x passes 256."
    >
      <Legend series={series} />
      <LineChart
        series={series}
        xLabel="training step (thousands)"
        yLabel="largest value"
        xDomain={[0, 75]}
        yDomain={[0, 400]}
        xStep={10}
        yStep={100}
        height={300}
        snap={2.6}
        xFormat={(x) => `step ${x}k`}
        yFormat={(y) => y.toFixed(0)}
        refLine={{ y: 256, label: 'above 256, x² overflows in FP16' }}
      />
    </Figure>
  )
}

// Charts under a game: x is the number of moves played (plies / 2); a click there moves the board.
const toPly = (x: number) => Math.round(x * 2)

export function SearchGame() {
  const game = boards.games.sf2000 as unknown as BoardLine
  const [k, setK] = useState(0)
  const score: Series[] = [
    {
      name: "the network's search (800 simulations)",
      slot: 1,
      points: [game.startScore ?? 0.5, ...game.plies.map((p) => p.whiteScore ?? 0.5)].map(
        (s, j) => [j / 2, s]
      ),
    },
    {
      name: 'Stockfish 16 at full strength (depth 20)',
      slot: 2,
      dotted: true,
      points: [game.startSfScore ?? 0.5, ...game.plies.map((p) => p.sfWhiteScore ?? 0.5)].map(
        (s, j) => [j / 2, s]
      ),
    },
  ]
  return (
    <Figure
      title="With search: the network against Stockfish limited to 2000 Elo"
      caption="The network plays White and searches 800 simulations per move; Stockfish 16 is limited to 2000 Elo, at 100 ms per move. The chart shows White's expected score at every position (a win counts 1, a draw half), from the network's own search and, dotted, from Stockfish at full strength. Stockfish sees an even game until 4…Ng4, a weak move by the limited Stockfish, and a won game for White from move 7. The network's search is more optimistic at the start (62% against 52%), and slower to see the win: it passes 90% on move 11. Click the chart to see a position."
    >
      <ChessBoard line={game} ply={k} onPly={setK} />
      <div className="mt-5">
        <Legend series={score} />
        <LineChart
          series={score}
          xLabel="moves played"
          yLabel="White's score"
          xDomain={[0, 40]}
          yDomain={[0.5, 1]}
          xStep={5}
          yStep={0.1}
          height={220}
          snap={0.3}
          xFormat={(x) => `after move ${Math.floor(x)}`}
          yFormat={percent}
          onPick={(x) => setK(toPly(x))}
          marker={k / 2}
        />
      </div>
    </Figure>
  )
}

export function ShowcaseGame() {
  const game = boards.games.sims0 as unknown as BoardLine
  const [k, setK] = useState(0)
  const n = game.plies.length
  const est = [game.startMovesLeft ?? 0, ...game.plies.map((p) => p.movesLeft ?? 0)]
  const movesLeft: Series[] = [
    { name: "network's estimate", slot: 1, points: est.map((e, j) => [j / 2, e]) },
    { name: 'actual', slot: 4, points: est.map((_, j) => [j / 2, (n - j) / 2]) },
  ]
  return (
    <Figure
      title="No search at all: the network against Stockfish limited to 1400 Elo"
      caption="The network plays Black and picks the move it finds most likely, without any search. Stockfish 16 is limited to 1400 Elo, at 100 ms per move. The network also estimates how many moves the game has left (its moves-left output, read for every position): in the opening it guesses about 20 when 35 to 40 are left, and over the last 10 moves it follows the real count closely. Click the chart to see a position."
    >
      <ChessBoard line={game} ply={k} onPly={setK} />
      <div className="mt-5">
        <Legend series={movesLeft} />
        <LineChart
          series={movesLeft}
          xLabel="moves played"
          yLabel="moves left"
          xDomain={[0, 41]}
          yDomain={[0, 45]}
          xStep={5}
          yStep={10}
          height={220}
          snap={0.3}
          xFormat={(x) => `after move ${Math.floor(x)}`}
          yFormat={(y) => y.toFixed(0)}
          onPick={(x) => setK(toPly(x))}
          marker={k / 2}
        />
      </div>
    </Figure>
  )
}

export function PuzzleBoard({ id }: { id: keyof typeof boards.puzzles }) {
  const line = boards.puzzles[id] as unknown as BoardLine & { meta: { rating: number } }
  return (
    <Figure
      title={`A puzzle rated ${line.meta.rating.toLocaleString('en-US')}, solved without search`}
      caption={`Lichess puzzle ${id}. Every move of the solution is the network's first choice, with no search.`}
    >
      <ChessBoard line={line} />
    </Figure>
  )
}
