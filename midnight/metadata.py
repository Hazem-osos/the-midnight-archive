"""YouTube metadata and thumbnails."""
from __future__ import annotations

import cv2
from PIL import Image, ImageDraw

from .branding import AMBER, font
from .core import settings
from .render import grade, text_image


def build_metadata(script, ep_number) -> dict:
    ch = settings()["channel"]
    md = script.get("metadata", {})
    title = md.get("title") or f"{script['title']} | Midnight Archive File {ep_number:03d}"
    tags = list(dict.fromkeys(md.get("tags", []) + ["horror story", "scary stories", "creepy", "Midnight Archive", "Elias Voss"]))
    hashtags = list(dict.fromkeys(md.get("hashtags", []) + ["#shorts", "#horrorstory", "#scarystories"]))[:8]
    desc = (f"{script.get('logline', '')}\n\n"
            f"File {ep_number:03d} of THE MIDNIGHT ARCHIVE, narrated by Elias Voss.\n"
            f"{ch['tagline']}\n\n"
            f"{ch['disclaimer']} Visuals, narration voice and sound design were created with local AI and procedural tools.\n\n"
            + " ".join(hashtags))
    return {"title": title[:100], "description": desc[:4900], "tags": tags[:30], "hashtags": hashtags,
            "category_id": settings()["youtube"]["category_id"], "privacy": settings()["youtube"]["privacy"],
            "made_for_kids": False, "contains_synthetic_media": settings()["youtube"].get("contains_synthetic_media", True)}


def build_thumbnails(ep_dir, script, ep_number):
    key = script.get("thumbnail_scene") or next(iter(script["scenes"]))
    img = cv2.imread(str(ep_dir / "images" / f"{key}.png"))
    img = grade(img)
    base = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out = {}
    for name, (W, H) in {"thumbnail_vertical.png": (1080, 1920), "thumbnail.png": (1280, 720)}.items():
        s = max(W / base.width, H / base.height)
        im = base.resize((int(base.width * s) + 1, int(base.height * s) + 1), Image.LANCZOS)
        im = im.crop(((im.width - W) // 2, (im.height - H) // 2, (im.width - W) // 2 + W, (im.height - H) // 2 + H)).convert("RGBA")
        shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(shade)
        for y in range(H):
            a = int(200 * max(0, (y / H - 0.45) / 0.55) ** 1.3)
            d.line([(0, y), (W, y)], fill=(0, 0, 0, a))
        im.alpha_composite(shade)
        words = (script.get("hook_text") or script["title"]).upper().split()
        mid = (len(words) + 1) // 2
        size = 150 if H > W else 110
        t = text_image([" ".join(words[:mid]), [(" ".join(words[mid:]), (230, 40, 40))]], font("subs", size), (255, 255, 255), stroke=10)
        if t.width > W - 60:
            t = t.resize((W - 60, int(t.height * (W - 60) / t.width)))
        im.alpha_composite(t, ((W - t.width) // 2, int(H * (0.62 if H > W else 0.42))))
        tag = text_image([f"MIDNIGHT ARCHIVE · FILE {ep_number:03d}"], font("stamp", 40 if H > W else 34), AMBER)
        im.alpha_composite(tag, ((W - tag.width) // 2, H - tag.height - (140 if H > W else 20)))
        im.convert("RGB").save(ep_dir / name, quality=92)
        out[name] = str(ep_dir / name)
    return out
