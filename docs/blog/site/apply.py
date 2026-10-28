"""Apply the two ChessRL posts and their native charts to a copy of mvdbk-site (idempotent).

    python3 docs/blog/site/apply.py <site dir>             # preview: the follow-up post stays a draft
    python3 docs/blog/site/apply.py <site dir> --publish   # the follow-up post goes live

Copies components/charts/, registers the chart and diagram components in
components/MDXComponents.tsx, copies both posts into data/blog/, and applies the mobile-menu
hydration fix (fix_mobilenav.py).
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLOG = HERE.parent
GROUPS = {
    "ChessrlNextRound": ["SpeedupChart", "TacticsChart", "ActivationChart", "EncoderDiagram", "ShowcaseGame",
                         "PuzzleBoard", "SearchGame"],
    "ChessrlRoundOne": ["LoopDiagram", "ResNetDiagram", "LegalMoveChart", "CastleChart", "SfEloChart",
                        "EloChainChart", "BrawlChart", "KqrkChart"],
}
POSTS = {"chessrl-alphazero-single-gpu.mdx": False, "chessrl-next-round.mdx": True}  # name -> draft in preview


def register(path: Path) -> None:
    s = path.read_text()
    for module, names in GROUPS.items():  # drop an earlier registration, then add the current one
        s = re.sub(rf"^import \{{[^}}]*\}} from './charts/{module}'\n", "", s, flags=re.M)
        for name in names:
            s = re.sub(rf"^  {name},\n", "", s, flags=re.M)
    anchor_import, anchor_entry = "import TableWrapper from './TableWrapper'\n", "  BlogNewsletterForm,\n"
    if s.count(anchor_import) != 1 or s.count(anchor_entry) != 1:
        sys.exit(f"unexpected layout in {path}")
    imports = "".join(f"import {{ {', '.join(names)} }} from './charts/{module}'\n" for module, names in GROUPS.items())
    entries = "".join(f"  {name},\n" for names in GROUPS.values() for name in names)
    path.write_text(s.replace(anchor_import, anchor_import + imports).replace(anchor_entry, anchor_entry + entries))


def main() -> None:
    site = Path(sys.argv[1]).expanduser()
    publish = "--publish" in sys.argv[2:]
    charts = site / "components" / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    for f in (HERE / "components" / "charts").iterdir():
        shutil.copy(f, charts / f.name)
    register(site / "components" / "MDXComponents.tsx")
    for name, draft in POSTS.items():
        text = (BLOG / name).read_text()
        text = re.sub(r"^draft: (true|false)$", f"draft: {'true' if draft and not publish else 'false'}", text,
                      count=1, flags=re.M)
        (site / "data" / "blog" / name).write_text(text)
    subprocess.run([sys.executable, str(HERE / "fix_mobilenav.py"), str(site)], check=True)
    print("applied to", site, "(published)" if publish else "(follow-up post as a draft)")


if __name__ == "__main__":
    main()
