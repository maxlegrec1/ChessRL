"""Silence the <body> hydration mismatch that browser extensions cause in mvdbk-site (app/layout.tsx).

Some extensions add attributes to <body> before React loads (ColorZilla adds
cz-shortcut-listen="true", Grammarly data-gr-ext-installed, ...), so React 19 reports that the
server HTML's attributes don't match. suppressHydrationWarning on <body> ignores attribute
differences on that element only; mismatches in its children are still reported.

    python3 fix_body.py <site dir>      # idempotent
"""

import re
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "app" / "layout.tsx"
s = path.read_text()
tags = re.findall(r"<body\s[^>]*>", s)  # Prettier may have split the tag over several lines
if len(tags) != 1:
    sys.exit(f"unexpected layout.tsx: {len(tags)} <body> tags")
if "suppressHydrationWarning" in tags[0]:
    print("already patched")
    sys.exit(0)
path.write_text(s.replace(tags[0], tags[0][:-1] + " suppressHydrationWarning>"))
print("patched", path)
