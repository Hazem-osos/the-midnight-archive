"""Channel branding drawn from scratch with Pillow (original artwork, no external assets).

Logo = a silver pocket watch stopped at 3:17 + the channel name in Cinzel.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .core import ASSETS, FONTS

BRAND = ASSETS / "brand"
AMBER = (255, 190, 92)
BONE = (236, 226, 204)
BLOOD = (176, 22, 22)


def font(name: str, size: int, weight: str | None = None):
    path = {"brand": FONTS / "Cinzel.ttf", "subs": FONTS / "Anton-Regular.ttf", "stamp": FONTS / "SpecialElite-Regular.ttf"}[name]
    f = ImageFont.truetype(str(path), size)
    if weight:
        try:
            f.set_variation_by_name(weight)
        except Exception:
            pass
    return f


def pocket_watch(size=600, hour=3, minute=17) -> Image.Image:
    S = size * 2  # supersample
    im = Image.new("RGBA", (S, int(S * 1.25)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy, R = S // 2, int(S * 0.25 + S * 0.5), int(S * 0.42)
    # bow + crown
    d.ellipse([cx - S * 0.11, cy - R - S * 0.24, cx + S * 0.11, cy - R - S * 0.02], outline=(170, 170, 175), width=int(S * 0.025))
    d.rounded_rectangle([cx - S * 0.045, cy - R - S * 0.07, cx + S * 0.045, cy - R + S * 0.01], radius=8, fill=(150, 150, 156))
    # case
    for i, col in enumerate([(70, 70, 76), (190, 190, 196), (120, 120, 126)]):
        r = R - i * S * 0.018
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    # dial
    r = R * 0.86
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(226, 214, 186))
    # aging stain
    stain = Image.new("L", im.size, 0)
    ImageDraw.Draw(stain).ellipse([cx - r * 0.9, cy - r * 0.2, cx + r * 0.3, cy + r * 0.9], fill=60)
    stain = stain.filter(ImageFilter.GaussianBlur(S * 0.06))
    im.paste((120, 95, 60, 255), (0, 0), Image.fromarray((np.array(stain) * 0.6).astype(np.uint8)))
    d = ImageDraw.Draw(im)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(60, 50, 40), width=int(S * 0.006))
    # ticks + numerals
    numerals = ["XII", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
    f = font("brand", int(S * 0.055), "Bold")
    for i in range(60):
        a = math.radians(i * 6 - 90)
        r0 = r * (0.86 if i % 5 == 0 else 0.92)
        d.line([cx + r0 * math.cos(a), cy + r0 * math.sin(a), cx + r * 0.97 * math.cos(a), cy + r * 0.97 * math.sin(a)],
               fill=(40, 34, 28), width=int(S * (0.008 if i % 5 == 0 else 0.003)))
    for i, n in enumerate(numerals):
        a = math.radians(i * 30 - 90)
        x, y = cx + r * 0.70 * math.cos(a), cy + r * 0.70 * math.sin(a)
        d.text((x, y), n, font=f, fill=(35, 30, 25), anchor="mm")
    # hands at 3:17
    def hand(angle_deg, length, width, col):
        a = math.radians(angle_deg - 90)
        d.line([cx - length * 0.15 * math.cos(a), cy - length * 0.15 * math.sin(a), cx + length * math.cos(a), cy + length * math.sin(a)],
               fill=col, width=width)
    hand((hour % 12 + minute / 60) * 30, r * 0.48, int(S * 0.022), (25, 22, 20))
    hand(minute * 6, r * 0.74, int(S * 0.013), (25, 22, 20))
    hand(minute * 6 + 180 + 40, r * 0.6, int(S * 0.004), BLOOD)  # stopped seconds hand
    d.ellipse([cx - S * 0.02, cy - S * 0.02, cx + S * 0.02, cy + S * 0.02], fill=(25, 22, 20))
    # glass crack
    d.line([cx + r * 0.2, cy - r * 0.75, cx + r * 0.05, cy - r * 0.35, cx + r * 0.25, cy - r * 0.05, cx + r * 0.12, cy + r * 0.3],
           fill=(255, 255, 255, 140), width=int(S * 0.004))
    return im.resize((size, int(size * 1.25)), Image.LANCZOS)


def glow(img: Image.Image, radius=30, color=AMBER, strength=0.8) -> Image.Image:
    pad = radius * 3
    big = Image.new("RGBA", (img.width + 2 * pad, img.height + 2 * pad), (0, 0, 0, 0))
    big.alpha_composite(img, (pad, pad))
    img = big
    a = img.split()[-1].filter(ImageFilter.GaussianBlur(radius))
    g = Image.new("RGBA", img.size, color + (0,))
    g.putalpha(a.point(lambda v: int(v * strength)))
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.alpha_composite(g)
    out.alpha_composite(img)
    return out


def build_logo(force=False):
    """Create logo.png (transparent), end_card.png (1080x1920) and avatar.png (800x800)."""
    BRAND.mkdir(parents=True, exist_ok=True)
    if (BRAND / "logo.png").exists() and not force:
        return
    watch = pocket_watch(420)
    W, H = 1000, 900
    logo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g = glow(watch, 40)
    logo.alpha_composite(g, ((W - g.width) // 2, -120))
    d = ImageDraw.Draw(logo)
    d.text((W // 2, 610), "THE MIDNIGHT", font=font("brand", 92, "Bold"), fill=BONE, anchor="mm")
    d.text((W // 2, 715), "ARCHIVE", font=font("brand", 120, "Black"), fill=BONE, anchor="mm")
    d.line([200, 790, 800, 790], fill=AMBER + (200,), width=3)
    d.text((W // 2, 835), "SOME FILES WERE NEVER MEANT TO BE OPENED", font=font("stamp", 30), fill=AMBER, anchor="mm")
    logo.save(BRAND / "logo.png")

    # avatar
    av = Image.new("RGBA", (800, 800), (8, 7, 9, 255))
    w2 = pocket_watch(470)
    g2 = glow(w2, 45)
    av.alpha_composite(g2, ((800 - g2.width) // 2, 20 - 135))
    ImageDraw.Draw(av).text((400, 700), "MIDNIGHT ARCHIVE", font=font("brand", 64, "Black"), fill=BONE, anchor="mm")
    av.convert("RGB").save(BRAND / "avatar.png")
