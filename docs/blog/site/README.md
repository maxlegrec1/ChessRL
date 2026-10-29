# Native charts for the two ChessRL posts

Both posts on mvdbk.fr draw their charts and diagrams as React components (SVG and HTML in
`components/charts/`), so they use the site's fonts and colours and follow its light/dark theme.
The charts show values on hover (and with the arrow keys) and each has a collapsible data table.

| post | source | components | data |
|---|---|---|---|
| AlphaZero from scratch: training a chess AI… | `docs/blog/chessrl-alphazero-single-gpu.mdx` | `ChessrlRoundOne.tsx` | `chessrl-round-one.json`, from `docs/blog/make_round1_charts.py` |
| AlphaZero from scratch, continued… | `docs/blog/alphazero-from-scratch-rtx-4090.mdx` | `ChessrlNextRound.tsx`, `EncoderDiagram.tsx`, `ChessBoard.tsx` | `chessrl-next-round.json`, from `docs/blog/make_charts.py` |

The first post's run data is partly gone (the experiment files stayed on the rented machine), so
`make_round1_charts.py` holds the match results copied from the labels of the original PNG charts,
and reads the training metrics from `docs/blog/data/round1_wandb.json` (exported from W&B).

## Preview

`docs/blog/preview.sh` (port 3003 by default) serves both posts on a throwaway copy of the site.

## Publishing to mvdbk-site

```bash
python3 docs/blog/site/apply.py ~/mvdbk-site --publish   # charts, component registry, both posts, MobileNav fix
git -C ~/mvdbk-site diff --stat                          # review, then commit and push
mvdbk-deploy
```

`apply.py` is idempotent. The old images in `public/static/images/chessrl/` are no longer used by
either post and can be deleted.

## Mobile menu hydration error (unrelated to the posts, every page)

In dev mode the site shows "Hydration failed … components/MobileNav.tsx": Headless UI 2.x
renders the closed menu dialog (`unmount={false}`) as a hidden `<span>` on the server and a
portal `<div>` in the browser. In production React recovers silently (a console error).
`fix_mobilenav.py` (run by `apply.py`) mounts the dialog after hydration; the menu is closed at
load, so nothing visible changes.
