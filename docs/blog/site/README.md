# Native charts for the "ChessRL, round two" post

The post's four charts are React components drawing SVG (`components/charts/`), so they
use the site's fonts and colours, follow its light/dark theme, and show values on hover
(and with the arrow keys). Each has a collapsible data table. Data:
`chessrl-next-round.json`, written by `docs/blog/make_charts.py` from the synced runs.

Preview without touching the live site: `docs/blog/preview.sh` (port 3003 by default).

## Publishing to mvdbk-site

1. Copy `site/components/charts/` to `mvdbk-site/components/charts/`.
2. Register the components in `mvdbk-site/components/MDXComponents.tsx`:

   ```tsx
   import { SpeedupChart, SearchBudgetChart, ActivationChart, AblationChart } from './charts/ChessrlNextRound'

   export const components: MDXComponents = {
     // ...existing entries
     SpeedupChart,
     SearchBudgetChart,
     ActivationChart,
     AblationChart,
   }
   ```
3. Copy `chessrl-next-round.mdx` to `mvdbk-site/data/blog/`, set `draft: false`.
4. `mvdbk-deploy`.

The PNGs in `docs/blog/img/` are no longer used by the post (kept for sharing).
