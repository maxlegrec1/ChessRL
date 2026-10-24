#!/usr/bin/env bash
# Preview the draft post locally on the real blog design, without touching the live site:
# a throwaway copy of mvdbk-site (its node_modules symlinked) gets the post and its native
# chart components, then `next dev` serves it (dev mode shows draft posts).
#   docs/blog/preview.sh [port]      -> http://localhost:<port>/blog/chessrl-next-round/
set -euo pipefail
PORT="${1:-3003}"
SITE="${MVDBK_SITE:-$HOME/mvdbk-site}"
P="${PREVIEW_DIR:-$HOME/.cache/mvdbk-blog-preview}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$P"
rsync -a --delete --exclude node_modules --exclude .next --exclude .contentlayer "$SITE"/ "$P"/
ln -sfn "$SITE/node_modules" "$P/node_modules"
cp "$SRC/chessrl-next-round.mdx" "$P/data/blog/"
mkdir -p "$P/components/charts"
cp "$SRC"/site/components/charts/* "$P/components/charts/"
# Register the chart components for MDX (the same edit as in site/README.md).
sed -i "s|^import TableWrapper from './TableWrapper'$|import TableWrapper from './TableWrapper'\nimport { SpeedupChart, SearchBudgetChart, ActivationChart, AblationChart } from './charts/ChessrlNextRound'|; s|^  BlogNewsletterForm,$|  BlogNewsletterForm,\n  SpeedupChart,\n  SearchBudgetChart,\n  ActivationChart,\n  AblationChart,|" "$P/components/MDXComponents.tsx"
# The site's mobile menu hydration fix (site/README.md); a no-op once the site has it.
python3 "$SRC/site/fix_mobilenav.py" "$P"
cd "$P"
echo "preview: http://localhost:$PORT/blog/chessrl-next-round/"
INIT_CWD="$P" exec npx next dev -p "$PORT" -H 127.0.0.1
