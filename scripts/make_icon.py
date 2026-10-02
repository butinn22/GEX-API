"""Generate the app icon -> ``trading/static/favicon.ico``.

The same file is used for the browser favicon and the desktop shortcut, so the
identity is consistent everywhere. Run it again after changing the palette:

    .venv/Scripts/python.exe scripts/make_icon.py

Design notes: a dark rounded tile in the app's own palette (``--bg`` #0d1117)
with four candlesticks ascending left-to-right. Everything is drawn at 4x and
downsampled, because PIL does not anti-alias polygons — without supersampling the
wicks look jagged at 16px, which is the size Windows actually shows most often.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "trading" / "static" / "favicon.ico"

# Palette lifted from trading/static/index.html (single source of truth for brand).
BG = (13, 17, 23, 255)  # --bg      #0d1117
BORDER = (48, 54, 61, 255)  # --border  #30363d
GREEN = (63, 185, 80, 255)  # --green   #3fb950
RED = (248, 81, 73, 255)  # --danger  #f85149
ACCENT = (47, 129, 247, 255)  # --accent #2f81f7

#: Master design grid (scaled to whatever size we render).
S = 256
SS = 4  # supersample factor

#: (centre x, body top, body bottom, wick top, wick bottom, colour) — an uptrend.
CANDLES = (
    (72, 150, 196, 132, 212, RED),
    (112, 120, 168, 104, 186, GREEN),
    (152, 132, 176, 116, 198, RED),
    (192, 84, 136, 66, 160, GREEN),
)
BODY_W = 26
WICK_W = 7


def _rounded_tile(size: int) -> Image.Image:
    """Dark rounded square with a hairline border, drawn at supersampled scale."""
    big = size * SS
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(big * 0.02)
    radius = int(big * 0.22)
    d.rounded_rectangle(
        (pad, pad, big - pad - 1, big - pad - 1),
        radius=radius,
        fill=BG,
        outline=BORDER,
        width=max(int(big * 0.012), 1),
    )
    return img


def _draw_candles(img: Image.Image, size: int) -> None:
    big = size * SS
    scale = big / S
    d = ImageDraw.Draw(img)

    for cx, top, bottom, wtop, wbot, colour in CANDLES:
        x = cx * scale
        # wick
        d.rectangle(
            (x - (WICK_W * scale) / 2, wtop * scale, x + (WICK_W * scale) / 2, wbot * scale),
            fill=colour,
        )
        # body
        d.rounded_rectangle(
            (x - (BODY_W * scale) / 2, top * scale, x + (BODY_W * scale) / 2, bottom * scale),
            radius=max(int(5 * scale), 1),
            fill=colour,
        )

    # Faint uptrend guide under the candles — reads as "chart" even at 16px.
    trend = ((40, 232), (108, 186), (176, 158), (216, 118))
    d.line(
        [(px * scale, py * scale) for px, py in trend],
        fill=ACCENT[:3] + (110,),  # override alpha (ACCENT is already RGBA)
        width=max(int(6 * scale), 1),
        joint="curve",
    )


def build(size: int) -> Image.Image:
    img = _rounded_tile(size)
    _draw_candles(img, size)
    return img.resize((size, size), Image.LANCZOS)


def main() -> int:
    sizes = [16, 24, 32, 48, 64, 128, 256]
    # PIL's ICO writer takes ONE source image and downscales it into every
    # requested size, so the source must be the largest frame — passing the 16px
    # frame silently yields a 16x16-only icon that Windows then blurs.
    master = build(max(sizes))
    master.save(OUT, format="ICO", sizes=[(s, s) for s in sizes])
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes, sizes: {sizes})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
