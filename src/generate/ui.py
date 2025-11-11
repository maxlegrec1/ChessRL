from typing import Dict
import time

from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich import box

def _render(stats: Dict[str, float]) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.expand = True

    metrics = Table(title="Self-Play Stats", box=box.SIMPLE_HEAVY)
    metrics.add_column("Metric")
    metrics.add_column("Value", justify="right")

    metrics.add_row("Elapsed (h)", f"{stats['elapsed_h']:.3f}")
    metrics.add_row("Started games", f"{int(stats['started_games'])}")
    metrics.add_row("Finished games", f"{int(stats['finished_games'])}")
    metrics.add_row("Active games", f"{int(stats['active_games'])}")
    metrics.add_row("Total moves", f"{int(stats['total_moves'])}")
    metrics.add_row("Games/hour", f"{stats['games_per_hour']:.1f}")
    metrics.add_row("Moves/hour", f"{stats['moves_per_hour']:.0f}")

    # Results and win rate by player
    metrics.add_row("Player 1 / Player 2", f"{stats['player_1_name']} / {stats['player_2_name']}")
    metrics.add_row("P1-D-P2", f"{int(stats['player_1_wins'])}-{int(stats['draws'])}-{int(stats['player_2_wins'])}")
    metrics.add_row("Player 1 win rate", f"{stats['player_1_winrate']*100:.1f}%")

    # Calibration / temperature
    if stats.get('calibration_target', 0.0) > 0.0:
        is_calib = stats.get('calibrating', 0.0) > 0.0
        if is_calib:
            # Show calibration progress and buckets while calibrating
            metrics.add_row("Calib games", f"{int(stats.get('calibration_done', 0.0))}/{int(stats.get('calibration_target', 0.0))}")
            metrics.add_row("Leela win rate (overall)", f"{stats.get('player_2_winrate_overall', 0.0)*100:.1f}%")
            buckets = stats.get('calibration_buckets', None)
            if buckets:
                bt = Table(title="Buckets", box=box.SIMPLE)
                bt.add_column("#", justify="right")
                bt.add_column("Temp")
                bt.add_column("Games", justify="right")
                bt.add_column("WR", justify="right")
                temps = buckets.get('temps', [])
                games = buckets.get('games', [])
                wrs = buckets.get('wrs', [])
                for idx, (t, g, w) in enumerate(zip(temps, games, wrs)):
                    bt.add_row(str(idx), f"{t:.3f}", str(int(g)), f"{w*100:.1f}%")
                blo = buckets.get('bracket_lo', None)
                bhi = buckets.get('bracket_hi', None)
                if blo is not None and bhi is not None:
                    bt.caption = f"Bracket: [{blo:.3f}, {bhi:.3f}]  Target per bucket: {int(buckets.get('samples_per_bucket', 0))}"
                table.add_row(metrics)
                table.add_row(bt)
                return Panel(table, title="Arena", border_style="cyan")
        else:
            # Calibration finished: show final Leela temp only
            metrics.add_row("Leela temp", f"{stats.get('player_2_temp', 0.0):.3f}")



    table.add_row(metrics)
    return Panel(table, title="Arena", border_style="cyan")


def run(arena):
    """Run the arena with a live UI. Falls back to simple prints if Rich is missing."""
    with Live(refresh_per_second=8) as live:
        for stats in arena.run():
            live.update(_render(stats))
