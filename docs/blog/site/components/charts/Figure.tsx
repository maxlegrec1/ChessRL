import type { ReactNode } from 'react'
import './charts.css'

/** Title, chart, caption: the frame every chart and diagram of the ChessRL posts sits in. */
export default function Figure({
  title,
  caption,
  children,
}: {
  title: string
  caption: ReactNode
  children: ReactNode
}) {
  return (
    <figure className="cr-chart not-prose my-8">
      <div className="mb-2 text-base font-semibold text-gray-900 dark:text-gray-100">{title}</div>
      {children}
      <figcaption className="mt-2 text-sm text-gray-500 dark:text-gray-400">{caption}</figcaption>
    </figure>
  )
}
