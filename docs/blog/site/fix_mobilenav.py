"""Fix the mobile menu's hydration mismatch in mvdbk-site (components/MobileNav.tsx).

Headless UI 2.x renders a closed Dialog with unmount={false} differently on the server (a
hidden <span>) and in the browser (a portal <div>), so React 19 reports "Hydration failed"
on every page. The dialog is closed at load anyway: mount it after hydration.

    python3 fix_mobilenav.py <site dir>      # idempotent
"""

import sys
from pathlib import Path

path = Path(sys.argv[1]) / "components" / "MobileNav.tsx"
s = path.read_text()
if "setMounted" in s:
    print("already patched")
    sys.exit(0)
edits = [
    ("  const navRef = useRef(null)\n",
     "  const navRef = useRef(null)\n"
     "  // Headless UI renders the closed dialog's portal differently on the server and in the\n"
     "  // browser; mount it after hydration (it is closed at load anyway).\n"
     "  const [mounted, setMounted] = useState(false)\n"
     "  useEffect(() => setMounted(true), [])\n"),
    ("      <Transition appear show={navShow} as={Fragment} unmount={false}>\n",
     "      {mounted && (\n      <Transition appear show={navShow} as={Fragment} unmount={false}>\n"),
    ("      </Transition>\n    </>\n", "      </Transition>\n      )}\n    </>\n"),
]
for old, new in edits:
    if s.count(old) != 1:
        sys.exit(f"unexpected MobileNav.tsx layout near: {old.strip()!r}")
    s = s.replace(old, new)
path.write_text(s)
print("patched", path)
