'use client'

import type { ReactNode } from 'react'
import data from './chessrl-round-one.json'
import Figure from './Figure'
import LineChart, { DataTable, Legend, type Series } from './LineChart'

/** Diagrams and charts of the first ChessRL post (data: docs/blog/make_round1_charts.py in ChessRL). */

const kSteps = (x: number) => `step ${x}k`
const STEP_LABEL = 'training step (thousands)'

function Box({
  title,
  lines,
  dashed = false,
  className = '',
}: {
  title: string
  lines: ReactNode[]
  dashed?: boolean
  className?: string
}) {
  return (
    <div
      className={`rounded-lg border p-3 ${dashed ? 'border-dashed' : ''} border-gray-300 dark:border-gray-700 ${className}`}
    >
      <div className="text-sm font-semibold text-gray-900 dark:text-gray-100">{title}</div>
      <ul className="mt-1 space-y-0.5 text-sm text-gray-600 dark:text-gray-400">
        {lines.map((l, i) => (
          <li key={i}>{l}</li>
        ))}
      </ul>
    </div>
  )
}

/** An arrow with a label. Phones show every arrow pointing down (the diagram becomes a list);
 *  from md up it points in `dir`. */
function Flow({
  dir,
  label,
  up = false,
  className = '',
}: {
  dir: 'right' | 'down' | 'left' | 'up'
  label: string
  up?: boolean
  className?: string
}) {
  const rot = {
    right: 'md:rotate-0',
    down: 'md:rotate-90',
    left: 'md:rotate-180',
    up: 'md:-rotate-90',
  }[dir]
  return (
    <div
      className={`flex flex-col items-center justify-center gap-1 px-1 py-1 text-center text-xs text-gray-500 dark:text-gray-400 ${className}`}
    >
      <svg
        width="20"
        height="20"
        viewBox="0 0 16 16"
        className={`${up ? '-rotate-90' : 'rotate-90'} ${rot}`}
        aria-hidden="true"
      >
        <path
          d="M2 8h11M9 4l4 4-4 4"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span>{label}</span>
    </div>
  )
}

export function LoopDiagram() {
  return (
    <Figure
      title="The self-play loop: two processes and two files"
      caption="The two processes share nothing but a file of positions and a one-line pointer file, so either one can be stopped, restarted or upgraded on its own."
    >
      <div className="grid grid-cols-1 gap-1 md:grid-cols-[minmax(0,1fr)_96px_minmax(0,1fr)_96px_minmax(0,1fr)] md:gap-y-2">
        <Box
          className="md:col-start-1 md:row-start-1"
          title="1. Self-play (C++)"
          lines={[
            '1,024 games at the same time',
            '800 simulations per move',
            'network run with TensorRT',
          ]}
        />
        <Flow className="md:col-start-2 md:row-start-1" dir="right" label="saves finished games" />
        <Box
          className="md:col-start-3 md:row-start-1"
          title="2. Buffer file"
          lines={[
            'the last 2M positions, 832 bytes each',
            'when full, new positions replace random old ones',
          ]}
        />
        <Flow className="md:col-start-4 md:row-start-1" dir="right" label="reads random batches" />
        <Box
          className="md:col-start-5 md:row-start-1"
          title="3. Trainer (PyTorch)"
          lines={['batches of 2,048 positions', 'Adam optimizer, FP16', 'trains the five outputs']}
        />
        <Flow className="md:col-start-5 md:row-start-2" dir="down" label="every 500 steps" />
        <Box
          className="md:col-start-5 md:row-start-3"
          title="4. Engine build"
          lines={['checkpoint → ONNX → TensorRT engine']}
        />
        <Flow
          className="md:col-span-3 md:col-start-2 md:row-start-3"
          dir="left"
          label="names the new engine"
        />
        <Box
          className="md:col-start-1 md:row-start-3"
          title="5. Pointer file"
          lines={[
            <code key="p">engines/.iteration</code>,
            'names the newest engine',
            'rewritten in one atomic step',
          ]}
        />
        <Flow
          className="md:col-start-1 md:row-start-2"
          dir="up"
          up
          label="self-play checks it between simulations and switches to the new engine without restarting"
        />
      </div>
    </Figure>
  )
}

export function ResNetDiagram() {
  return (
    <Figure
      title="The network: 1.29M parameters, five outputs"
      caption="Solid and dashed outputs come in pairs: the solid one learns from what happened in the game, the dashed one from the search's estimate of the same position."
    >
      <div className="grid grid-cols-1 items-center gap-1 md:grid-cols-[minmax(0,0.9fr)_32px_minmax(0,0.9fr)_32px_minmax(0,1.3fr)]">
        <Box
          title="Input"
          lines={[
            '111 planes of 8 × 8',
            'pieces in the last 8 positions, castling, side to move, move counters',
          ]}
        />
        <Flow dir="right" label="" />
        <Box title="Trunk" lines={['3 × 3 convolution to 64 channels', '4 residual blocks']} />
        <Flow dir="right" label="" />
        <div className="grid gap-1.5">
          <Box
            title="Policy"
            lines={[
              'a score for each of 1,929 possible moves',
              'learns: the move the search chose',
            ]}
          />
          <Box title="Value" lines={['win / draw / loss', "learns: the game's real result"]} />
          <Box
            title="Value, search version"
            dashed
            lines={['win / draw / loss', "learns: the search's estimate"]}
          />
          <Box
            title="Moves left"
            lines={['plies until the game ends', 'learns: the real number']}
          />
          <Box
            title="Moves left, search version"
            dashed
            lines={["learns: the search's estimate"]}
          />
        </div>
      </div>
    </Figure>
  )
}

export function SfEloChart() {
  const series = data.sfElo as Series[]
  return (
    <Figure
      title="Desktop run: Elo against Stockfish-1400"
      caption="Each point is a match against Stockfish 16 set to UCI_Elo 1400, 100 ms per move (10 to 100 games; the early matches were short and noisy). Elo is the rating that would produce the same score against a 1400 opponent."
    >
      <LineChart
        series={series}
        xLabel={STEP_LABEL}
        yLabel="Elo"
        xDomain={[0, 145]}
        yDomain={[850, 1550]}
        xStep={20}
        yStep={100}
        height={300}
        snap={2}
        xFormat={kSteps}
        refLine={{ y: 1400, label: 'equal strength with Stockfish-1400' }}
      />
      <DataTable
        series={series}
        xHeader="step"
        xs={series[0].points.map((p) => p[0])}
        snap={0.01}
        xFormat={(x) => `${x}k`}
      />
    </Figure>
  )
}

export function EloChainChart() {
  const series = data.eloChain as Series[]
  return (
    <Figure
      title="Instrumented run: Elo gained against its own earlier checkpoints"
      caption="Every 5,000 steps, the new checkpoint plays the previous one (50 games, 800 simulations per move, 8 random opening moves). Adding up the Elo differences gives the curve, relative to step 5,000. Dashed lines: the two fixes described above."
    >
      <LineChart
        series={series}
        xLabel={STEP_LABEL}
        yLabel="Elo relative to step 5k"
        xDomain={[0, 260]}
        yDomain={[-100, 1400]}
        xStep={50}
        yStep={200}
        height={300}
        snap={2.5}
        xFormat={kSteps}
        endLabels={false}
        xMarks={FIXES}
      />
      <DataTable
        series={series}
        xHeader="step"
        xs={series[0].points.map((p) => p[0])}
        snap={0.01}
        xFormat={(x) => `${x}k`}
      />
    </Figure>
  )
}

const FIXES = [
  { x: 64.5, label: 'mirror-bug fix' },
  { x: 140, label: 'Dirichlet fix' },
]
const tableXs = (series: Series[], every: number) =>
  series[0].points
    .map((p) => p[0])
    .filter((x) => Math.abs(x / every - Math.round(x / every)) < 1e-6)

export function LegalMoveChart() {
  const panels = data.legal as unknown as Record<'mass' | 'top1', Series[]>
  const fmt = (y: number) => y.toFixed(2)
  return (
    <Figure
      title="Instrumented run: the mirror bug's signature"
      caption="Left: share of the policy's probability that falls on legal moves. Right: how often the policy's top move is the move the search chose. Both from W&B, averaged over 500-step bins and smoothed over 2,500 steps; the straight segments near 63k and 139k bridge two short restarts that were not plotted. Dashed lines: the two fixes."
    >
      <div className="grid gap-6 md:grid-cols-2">
        <div>
          <div className="mb-1 text-sm text-gray-600 dark:text-gray-400">
            Probability on legal moves
          </div>
          <LineChart
            series={panels.mass}
            xLabel={STEP_LABEL}
            yLabel="share"
            xDomain={[0, 260]}
            yDomain={[0.3, 1]}
            xStep={50}
            yStep={0.1}
            height={260}
            snap={0.3}
            xFormat={kSteps}
            yFormat={fmt}
            endLabels={false}
            xMarks={FIXES}
          />
        </div>
        <div>
          <div className="mb-1 text-sm text-gray-600 dark:text-gray-400">
            Top move agrees with the search
          </div>
          <LineChart
            series={panels.top1}
            xLabel={STEP_LABEL}
            yLabel="share"
            xDomain={[0, 260]}
            yDomain={[0, 0.35]}
            xStep={50}
            yStep={0.05}
            height={260}
            snap={0.3}
            xFormat={kSteps}
            yFormat={fmt}
            endLabels={false}
            xMarks={FIXES}
          />
        </div>
      </div>
      <DataTable
        series={[...panels.mass, ...panels.top1]}
        xHeader="step"
        xs={tableXs(panels.mass, 10)}
        snap={0.01}
        xFormat={(x) => `${x}k`}
        yFormat={fmt}
      />
    </Figure>
  )
}

export function CastleChart() {
  const series = data.castle as Series[]
  return (
    <Figure
      title="Instrumented run: castling moves in the replay buffer"
      caption="Share of buffer records that are castling moves, from W&B (the metric was added with the Dirichlet fix at step 140k), averaged over 500-step bins and smoothed over 2,500 steps. The shaded band is the usual share of castling in normal chess."
    >
      <LineChart
        series={series}
        xLabel={STEP_LABEL}
        yLabel="castling moves (%)"
        xDomain={[135, 260]}
        yDomain={[0, 1.3]}
        xStep={20}
        yStep={0.2}
        height={280}
        snap={0.3}
        xFormat={kSteps}
        yFormat={(y) => y.toFixed(2)}
        endLabels={false}
        band={{ y0: 0.8, y1: 1.2, label: 'normal chess: 0.8 to 1.2%' }}
        xMarks={[FIXES[1]]}
      />
      <DataTable
        series={series}
        xHeader="step"
        xs={tableXs(series, 10)}
        snap={0.01}
        xFormat={(x) => `${x}k`}
        yFormat={(y) => `${y.toFixed(2)}%`}
      />
    </Figure>
  )
}

export function BrawlChart() {
  const series = data.brawl as Series[]
  return (
    <Figure
      title="Instrumented run: score against Stockfish-1400"
      caption="Each new checkpoint plays 50 games against Stockfish 16 set to UCI_Elo 1400, 100 ms per move. Score: wins count 1, draws ½. Above 50%, the model is the stronger player."
    >
      <Legend series={series} />
      <LineChart
        series={series}
        xLabel={STEP_LABEL}
        yLabel="score (%)"
        xDomain={[155, 260]}
        yDomain={[25, 65]}
        xStep={20}
        yStep={10}
        height={280}
        snap={1}
        xFormat={kSteps}
        yFormat={(y) => `${Math.round(y)}%`}
        endLabels={false}
        refLine={{ y: 50, label: 'equal score' }}
      />
      <DataTable
        series={series}
        xHeader="step"
        xs={series[0].points.map((p) => p[0])}
        snap={0.01}
        xFormat={(x) => `${x}k`}
        yFormat={(y) => `${y.toFixed(0)}%`}
      />
    </Figure>
  )
}

export function KqrkChart() {
  const panels = data.kqrk as unknown as Record<'rate' | 'plies', Series[]>
  const xs = panels.rate[0].points.map((p) => p[0])
  return (
    <Figure
      title="Mating with king, queen and rook against a bare king"
      caption="6 positions; the model plays both sides with MCTS. Left: how many positions end in mate. Right: the average number of plies (half-moves) to mate, counting only the positions that were mated."
    >
      <div className="grid gap-6 md:grid-cols-2">
        <div>
          <div className="mb-1 text-sm text-gray-600 dark:text-gray-400">Positions mated</div>
          <LineChart
            series={panels.rate}
            xLabel={STEP_LABEL}
            yLabel="mated (%)"
            xDomain={[0, 115]}
            yDomain={[0, 100]}
            xStep={20}
            yStep={25}
            height={240}
            snap={1}
            xFormat={kSteps}
            yFormat={(y) => `${Math.round(y)}%`}
            endLabels={false}
          />
        </div>
        <div>
          <div className="mb-1 text-sm text-gray-600 dark:text-gray-400">Plies to mate</div>
          <LineChart
            series={panels.plies}
            xLabel={STEP_LABEL}
            yLabel="plies"
            xDomain={[0, 115]}
            yDomain={[0, 40]}
            xStep={20}
            yStep={10}
            height={240}
            snap={1}
            xFormat={kSteps}
            endLabels={false}
          />
        </div>
      </div>
      <DataTable
        series={[...panels.rate, ...panels.plies]}
        xHeader="step"
        xs={xs}
        snap={0.01}
        xFormat={(x) => `${x}k`}
      />
    </Figure>
  )
}
