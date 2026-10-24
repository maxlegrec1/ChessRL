#!/usr/bin/env bash
# Preview both ChessRL posts locally on the real blog design, without touching the live site:
# a throwaway copy of mvdbk-site (its node_modules symlinked) gets the posts and their native
# charts (site/apply.py), then `next dev` serves it (dev mode also shows draft posts).
#   docs/blog/preview.sh [port]      -> http://localhost:<port>/blog/chessrl-next-round/
set -euo pipefail
PORT="${1:-3003}"
SITE="${MVDBK_SITE:-$HOME/mvdbk-site}"
P="${PREVIEW_DIR:-$HOME/.cache/mvdbk-blog-preview}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$P"
rsync -a --delete --exclude node_modules --exclude .next --exclude .contentlayer "$SITE"/ "$P"/
ln -sfn "$SITE/node_modules" "$P/node_modules"
python3 "$SRC/site/apply.py" "$P"
cd "$P"
echo "preview: http://localhost:$PORT/blog/chessrl-next-round/ and /blog/chessrl-alphazero-single-gpu/"
INIT_CWD="$P" exec npx next dev -p "$PORT" -H 127.0.0.1
