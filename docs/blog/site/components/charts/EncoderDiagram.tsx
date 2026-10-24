import type { ReactNode } from 'react'
import './charts.css'

/** The encoder of the "ChessRL, round two" post (src/models/encoder.py in ChessRL): squares
 *  become tokens, four layers mix them, then the policy and value heads read them out. */

const ACCENT = 'var(--cr-s1)'
const INK = 'var(--cr-text)'

function Board({ x, y, sq }: { x: number; y: number; sq: number }) {
  const cells: ReactNode[] = []
  for (let r = 0; r < 8; r++)
    for (let f = 0; f < 8; f++)
      cells.push(
        <rect key={r * 8 + f} x={x + f * sq} y={y + r * sq} width={sq} height={sq}
          fill={(r + f) % 2 ? 'var(--cr-sq-dark)' : 'var(--cr-sq-light)'} />
      )
  return <g>{cells}</g>
}

/** A square picked out on a board: file 0-7 from a, row 0-7 from rank 8. */
function Mark({ x, y, sq, f, r, filled = true }: { x: number; y: number; sq: number; f: number; r: number; filled?: boolean }) {
  return (
    <rect x={x + f * sq + 1} y={y + r * sq + 1} width={sq - 2} height={sq - 2} fill={filled ? ACCENT : 'none'}
      fillOpacity={0.3} stroke={ACCENT} strokeWidth={2} />
  )
}

/** A row of cells standing for a vector. */
function Vec({ x, y, w, n, stroke }: { x: number; y: number; w: number; n: number; stroke: string }) {
  const cw = (w + 2) / n - 2
  return (
    <g>
      {Array.from({ length: n }, (_, i) => (
        <rect key={i} x={x + i * (cw + 2)} y={y} width={cw} height={12} rx={1.5} fill="none" stroke={stroke} />
      ))}
    </g>
  )
}

/** Arrowhead with its tip at (x, y). */
function Tip({ x, y, dir, color = INK }: { x: number; y: number; dir: 'down' | 'right' | 'left'; color?: string }) {
  const pts = {
    down: `${x - 5},${y - 8} ${x + 5},${y - 8} ${x},${y}`,
    right: `${x - 8},${y - 5} ${x - 8},${y + 5} ${x},${y}`,
    left: `${x + 8},${y - 5} ${x + 8},${y + 5} ${x},${y}`,
  }[dir]
  return <polygon points={pts} fill={color} />
}

function Plus({ x, y }: { x: number; y: number }) {
  return (
    <g>
      <circle cx={x} cy={y} r={9} fill="var(--cr-surface)" className="cr-wire" />
      <path d={`M${x - 4.5} ${y}h9M${x} ${y - 4.5}v9`} className="cr-wire" />
    </g>
  )
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="mb-2 text-sm font-medium text-gray-900 dark:text-gray-100">{title}</div>
      {children}
    </div>
  )
}

function Arrow({ down = false }: { down?: boolean }) {
  return (
    <div className="flex justify-center self-center text-gray-400 dark:text-gray-500" aria-hidden="true">
      <svg width="18" height="18" viewBox="0 0 16 16" className={down ? 'rotate-90' : 'rotate-90 md:rotate-0'}>
        <path d="M2 8h11M9 4l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.75"
          strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  )
}

// Every panel is drawn at the same scale (at most 1.32x its viewBox) so text matches across panels.
const svgProps = (w: number, h: number, label: string) => ({
  viewBox: `0 0 ${w} ${h}`,
  role: 'img',
  'aria-label': label,
  className: 'cr-dg mx-auto block h-auto w-full',
  style: { maxWidth: Math.round(w * 1.32) },
})

function Tokens() {
  return (
    <svg {...svgProps(220, 296, 'Each of the 64 squares becomes a token: its 111 input numbers go through one linear layer, plus a learned vector for its position, giving 128 numbers.')}>
      <Board x={46} y={4} sq={16} />
      <Mark x={46} y={4} sq={16} f={4} r={4} />
      <line x1={118} y1={84} x2={118} y2={144} stroke={ACCENT} strokeWidth={1.5} />
      <Tip x={118} y={150} dir="down" color={ACCENT} />
      <Vec x={46} y={154} w={128} n={16} stroke={ACCENT} />
      <text x={110} y={182} textAnchor="middle">111 numbers describe this square</text>
      <line x1={110} y1={190} x2={110} y2={214} className="cr-wire" />
      <Tip x={110} y={220} dir="down" />
      <text x={118} y={202}>linear layer</text>
      <text x={118} y={216}>+ position</text>
      <Vec x={46} y={226} w={128} n={16} stroke="var(--cr-ink)" />
      <text x={110} y={256} textAnchor="middle" className="cr-strong">the square&apos;s token: 128 numbers</text>
      <text x={110} y={272} textAnchor="middle">64 tokens, one per square</text>
    </svg>
  )
}

function Layers() {
  // The main line (residual stream) runs down the left; each block branches off it at a dot,
  // and its output is added back in at a + node further down.
  const attendTo: [number, number][] = [[6, 1], [7, 5], [2, 0], [0, 7], [5, 7]]
  const c = (f: number, r: number): [number, number] => [60 + f * 9 + 4.5, 112 + r * 9 + 4.5]
  const [sx, sy] = c(2, 5)
  return (
    <svg {...svgProps(300, 330, 'Four layers, each: add the input tokens back in, then attention, where every square reads from every other square with a learned bias for each pair of squares, then a small network applied to each square on its own. Each block adds its output back onto the main line.')}>
      <text x={34} y={12}>64 tokens in</text>
      <path d="M24 0V316" className="cr-wire" />
      <Tip x={24} y={324} dir="down" />
      <Plus x={24} y={34} />
      <text x={42} y={38}>add the input tokens back in</text>

      <Tip x={24} y={62} dir="down" />
      <circle cx={24} cy={74} r={3} fill={INK} />
      <path d="M24 74H40" className="cr-wire" />
      <Tip x={48} y={74} dir="right" />
      <rect x={48} y={58} width={248} height={150} rx={6} className="cr-box" />
      <text x={60} y={78} className="cr-strong">attention</text>
      <text x={60} y={96}>normalise, then 8 heads in parallel</text>
      <Board x={60} y={112} sq={9} />
      <Mark x={60} y={112} sq={9} f={2} r={5} />
      {attendTo.map(([f, r]) => {
        const [tx, ty] = c(f, r)
        return (
          <g key={`${f}${r}`}>
            <line x1={sx} y1={sy} x2={tx} y2={ty} stroke={ACCENT} strokeWidth={1.25} />
            <circle cx={tx} cy={ty} r={2.2} fill={ACCENT} />
          </g>
        )
      })}
      <text x={146} y={126}>each square reads</text>
      <text x={146} y={142}>from every other,</text>
      <text x={146} y={158}>plus a learned bias</text>
      <text x={146} y={174}>for each pair of</text>
      <text x={146} y={190}>squares</text>
      <Tip x={24} y={150} dir="down" />
      <path d="M48 196H40" className="cr-wire" />
      <Tip x={33} y={196} dir="left" />
      <Plus x={24} y={196} />

      <Tip x={24} y={226} dir="down" />
      <circle cx={24} cy={236} r={3} fill={INK} />
      <path d="M24 236H40" className="cr-wire" />
      <Tip x={48} y={236} dir="right" />
      <rect x={48} y={222} width={248} height={68} rx={6} className="cr-box" />
      <text x={60} y={242} className="cr-strong">small network, square by square</text>
      <text x={60} y={260}>normalise, then</text>
      <text x={60} y={276}>128 → 256 → 128 numbers</text>
      <Tip x={24} y={264} dir="down" />
      <path d="M48 280H40" className="cr-wire" />
      <Tip x={33} y={280} dir="left" />
      <Plus x={24} y={280} />
      <text x={42} y={318}>repeat 4 times, then normalise</text>
    </svg>
  )
}

function Policy() {
  return (
    <svg {...svgProps(230, 216, 'Policy head: every move gets a score, the dot product of a vector q from its from-square token and a vector k from its to-square token.')}>
      <rect x={4} y={4} width={222} height={208} rx={6} className="cr-box" />
      <text x={16} y={26} className="cr-strong">move scores (policy)</text>
      <text x={16} y={44} className="cr-ink">score = q(from) · k(to)</text>
      <Board x={16} y={56} sq={11} />
      <Mark x={16} y={56} sq={11} f={5} r={7} filled={false} />
      <Mark x={16} y={56} sq={11} f={1} r={3} />
      <line x1={76.5} y1={138.5} x2={37} y2={99} stroke={ACCENT} strokeWidth={1.75} />
      <polygon points="32.5,94.5 40.6,97.2 35.2,102.6" fill={ACCENT} />
      <text x={114} y={70}>q: a vector from</text>
      <text x={114} y={86}>the from-square</text>
      <text x={114} y={106}>k: a vector from</text>
      <text x={114} y={122}>the to-square</text>
      <text x={114} y={142}>q · k: one number</text>
      <text x={16} y={168}>all 64 × 64 square pairs at once;</text>
      <text x={16} y={184}>a promotion adds a learned</text>
      <text x={16} y={198}>vector to q</text>
    </svg>
  )
}

function Value() {
  return (
    <svg {...svgProps(230, 216, 'Value head: the 64 tokens are averaged and a small network predicts win, draw or loss; another predicts how many moves are left.')}>
      <rect x={4} y={4} width={222} height={208} rx={6} className="cr-box" />
      <text x={16} y={26} className="cr-strong">position value</text>
      <g>
        {Array.from({ length: 8 }, (_, i) => (
          <rect key={i} x={16 + i * 14} y={44} width={10} height={10} rx={1.5} fill="none" stroke="var(--cr-ink)" />
        ))}
        <text x={136} y={53}>64 tokens</text>
        <line x1={52} y1={62} x2={52} y2={76} className="cr-wire" />
        <Tip x={52} y={82} dir="down" />
        <text x={64} y={76}>average</text>
        <Vec x={16} y={86} w={72} n={8} stroke="var(--cr-ink)" />
        <line x1={52} y1={104} x2={52} y2={118} className="cr-wire" />
        <Tip x={52} y={124} dir="down" />
        <text x={64} y={118}>small network</text>
      </g>
      {[['win', 0.5], ['draw', 0.3], ['loss', 0.2]].map(([label, p], i) => (
        <g key={label as string}>
          <rect x={16} y={132 + i * 18} width={(p as number) * 120} height={12} rx={2} fill={ACCENT} fillOpacity={0.35}
            stroke={ACCENT} />
          <text x={16 + (p as number) * 120 + 8} y={142 + i * 18}>{label}</text>
        </g>
      ))}
      <text x={16} y={198}>+ a second one: moves left</text>
    </svg>
  )
}

export default function EncoderDiagram() {
  return (
    <figure className="cr-chart not-prose my-8">
      <div className="mb-3 text-base font-semibold text-gray-900 dark:text-gray-100">The 0.81M-parameter encoder</div>
      <div className="grid items-start gap-3 md:grid-cols-[minmax(0,0.85fr)_20px_minmax(0,1.15fr)] md:gap-2">
        <Panel title="1. The board becomes 64 tokens"><Tokens /></Panel>
        <Arrow />
        <Panel title="2. Four layers mix the squares"><Layers /></Panel>
      </div>
      <div className="my-2"><Arrow down /></div>
      <Panel title="3. Two heads read the result">
        <div className="grid gap-3 md:grid-cols-2">
          <Policy />
          <Value />
        </div>
      </Panel>
      <figcaption className="mt-3 text-sm text-gray-500 dark:text-gray-400">
        The 111 numbers for a square say which piece stands on it in the current position and in the 7
        before it, whether each of those positions is a repetition, plus castling rights, side to move,
        en passant and the 50-move counter. &quot;Normalise&quot; rescales each token to a fixed average size
        (a layer called RMSNorm). Attention has 8 heads: 8 separate ways of choosing which squares to read from,
        computed in parallel. The learned bias is one 64 × 64 table per head, shared by all four layers: it is
        how the network learns the shape of the board. The &quot;small network&quot; boxes are two-layer
        networks (MLPs). There are really two value outputs and two moves-left outputs: one of each is trained
        on final game results (the one the search uses), the other on the search&apos;s own estimates.
      </figcaption>
    </figure>
  )
}
