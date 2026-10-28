"""The small round avatar for the 'Claude' author of the ChessRL posts (site/images/claude-avatar.png):
a white eight-point star on a terracotta disc, drawn here (4x supersampled, then downsized).

  .venv/bin/python docs/blog/site/make_avatar.py
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "images" / "claude-avatar.png"
SIZE, SS = 128, 4
TERRACOTTA, WHITE = (217, 119, 87, 255), (255, 255, 255, 255)

big = SIZE * SS
img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.ellipse((0, 0, big - 1, big - 1), fill=TERRACOTTA)
c = big / 2
r_out, r_in, half_w = big * 0.34, big * 0.07, math.radians(9)
for k in range(8):  # eight tapered rays, alternating long and short
    a = math.radians(45 * k - 90)
    r = r_out if k % 2 == 0 else r_out * 0.72
    tip = (c + r * math.cos(a), c + r * math.sin(a))
    left = (c + r_in * math.cos(a - half_w * 4), c + r_in * math.sin(a - half_w * 4))
    right = (c + r_in * math.cos(a + half_w * 4), c + r_in * math.sin(a + half_w * 4))
    d.polygon([left, tip, right, (c, c)], fill=WHITE)
d.ellipse((c - r_in * 1.3, c - r_in * 1.3, c + r_in * 1.3, c + r_in * 1.3), fill=WHITE)
OUT.parent.mkdir(parents=True, exist_ok=True)
img.resize((SIZE, SIZE), Image.LANCZOS).save(OUT)
print("wrote", OUT)
