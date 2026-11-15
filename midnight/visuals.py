"""Scene image production: local AI images (preferred) or a procedural fallback.

The narrator scene always reuses the locked Elias Voss reference image.
"""
from __future__ import annotations

import shutil

import numpy as np
import psutil
from PIL import Image, ImageDraw, ImageFilter

from .core import ROOT, character, settings, torch_device

REF_MIN_RAM_GB = 3.5


def ai_available(log) -> tuple[bool, str]:
    cfg = settings()["imagegen"]
    if not cfg.get("enabled", True):
        return False, "disabled in settings"
    if not (ROOT / "models" / "dreamshaper-8" / "model_index.json").exists():
        return False, "image model not downloaded (run: python archive.py setup-models)"
    dev = torch_device()
    if dev == "cuda":
        import torch
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram < 3.8:
            return False, f"only {vram:.1f} GB VRAM"
        return True, f"cuda ({torch.cuda.get_device_name(0)}, {vram:.1f} GB VRAM)"
    avail = psutil.virtual_memory().available / 1e9
    if avail < REF_MIN_RAM_GB:
        return False, f"only {avail:.1f} GB RAM available on {dev} (need {REF_MIN_RAM_GB}); close other apps"
    return True, f"{dev} ({avail:.1f} GB RAM free)"


def reference_path():
    return ROOT / character()["visual_reference"]["image"]


# ---------------------------------------------------------------- fallback art
def _fog(w, h, seed, scale=6):
    r = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    for octave in range(4):
        s = scale * 2 ** octave
        small = r.random((s * h // w + 2, s + 2)).astype(np.float32)
        img = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        acc += np.asarray(img, np.float32) / 255 / (2 ** octave)
    acc /= acc.max()
    return acc


def procedural_scene(prompt: str, out_path, seed: int, w=768, h=1152):
    """Atmospheric abstract plate (fog, light shaft, silhouette door/window motif)."""
    r = np.random.default_rng(seed)
    fog = _fog(w, h, seed)
    yy = np.linspace(0, 1, h)[:, None]
    base = 0.05 + 0.1 * fog + 0.06 * (1 - yy)
    lx = r.uniform(0.25, 0.75)
    xx = np.linspace(0, 1, w)[None, :]
    shaft = np.exp(-((xx - lx - (yy - 0.2) * r.uniform(-0.3, 0.3)) ** 2) / 0.006) * (0.25 + 0.4 * fog) * (1 - yy * 0.6)
    lum = np.clip(base + shaft, 0, 1)
    rgb = np.stack([lum * 0.85, lum * 0.95, lum * 1.1], -1)
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    p = prompt.lower()
    if any(k in p for k in ("door", "wall", "room", "hallway")):
        dx = int(w * r.uniform(0.3, 0.5))
        d.rectangle([dx, int(h * 0.35), dx + int(w * 0.28), int(h * 0.8)], fill=(4, 4, 6))
    if any(k in p for k in ("man", "figure", "person", "officer")):
        cx = int(w * r.uniform(0.35, 0.65))
        d.ellipse([cx - 45, int(h * 0.48), cx + 45, int(h * 0.48) + 110], fill=(3, 3, 4))
        d.polygon([(cx - 130, h), (cx - 90, int(h * 0.6)), (cx + 90, int(h * 0.6)), (cx + 130, h)], fill=(3, 3, 4))
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    img.save(out_path)


def procedural_narrator(out_path, w=768, h=1152):
    """Fallback narrator plate: backlit silhouette at a candle-lit desk (used only without AI images)."""
    img = Image.new("RGB", (w, h), (10, 8, 8))
    d = ImageDraw.Draw(img)
    for i in range(12):  # shelves
        y = int(h * 0.08 + i * h * 0.05)
        d.rectangle([0, y, w, y + 6], fill=(30, 22, 16))
        for x in range(0, w, 22):
            d.rectangle([x, y - 40 + (x * 7 % 13), x + 16, y], fill=(40 + x % 30, 28, 20))
    glow = Image.new("L", (w, h), 0)
    ImageDraw.Draw(glow).ellipse([w * 0.15, h * 0.55, w * 0.85, h * 1.05], fill=200)
    glow = glow.filter(ImageFilter.GaussianBlur(120))
    img = Image.composite(Image.new("RGB", (w, h), (255, 170, 80)), img, glow.point(lambda v: int(v * 0.55)))
    d = ImageDraw.Draw(img)
    cx = w // 2
    d.ellipse([cx - 95, int(h * 0.30), cx + 95, int(h * 0.30) + 240], fill=(6, 5, 5))
    d.polygon([(cx - 300, int(h * 0.82)), (cx - 200, int(h * 0.55)), (cx + 200, int(h * 0.55)), (cx + 300, int(h * 0.82))], fill=(6, 5, 5))
    d.rectangle([0, int(h * 0.80), w, h], fill=(30, 18, 10))
    for x in (0.2, 0.28, 0.75):
        d.rectangle([w * x, h * 0.72, w * x + 18, h * 0.80], fill=(220, 210, 190))
        d.ellipse([w * x + 2, h * 0.70, w * x + 16, h * 0.72], fill=(255, 210, 120))
    img.filter(ImageFilter.GaussianBlur(1.0)).save(out_path)


# ---------------------------------------------------------------- orchestration
def ensure_reference(log, generator=None):
    """Generate the locked Elias Voss reference image once. Returns its path."""
    ref = reference_path()
    if ref.exists():
        return ref
    ref.parent.mkdir(parents=True, exist_ok=True)
    vr = character()["visual_reference"]
    if generator is None:
        raise RuntimeError("reference image missing and no image generator available")
    seed = vr.get("seed") or 317
    log.info("Creating the permanent Elias Voss reference image (one time only)")
    generator.generate(vr["prompt"], ref, seed=seed, negative=vr["negative_prompt"], style=False)
    return ref


def produce_images(ep_dir, script, log, mode="auto", progress_cb=None):
    """Create images/<scene>.png and images/<scene>_depth.png for every scene."""
    img_dir = ep_dir / "images"
    img_dir.mkdir(exist_ok=True)
    scenes = script["scenes"]
    use_ai, why = ai_available(log) if mode in ("auto", "ai") else (False, "procedural mode requested")
    if mode == "ai" and not use_ai:
        raise RuntimeError(f"AI images requested but unavailable: {why}")
    log.info(f"Visuals: {'local AI (' + why + ')' if use_ai else 'PROCEDURAL FALLBACK (' + why + ')'}")
    todo = [k for k in scenes if not (img_dir / f"{k}.png").exists()]
    gen = None
    try:
        if use_ai and (todo or not reference_path().exists()):
            from .imagegen import ImageGenerator
            gen = ImageGenerator(log)
            ensure_reference(log, gen)
        for i, key in enumerate(todo):
            sc = scenes[key]
            out = img_dir / f"{key}.png"
            if sc.get("character") == "elias_voss":
                ref = reference_path()
                if ref.exists():
                    shutil.copy(ref, out)
                else:
                    procedural_narrator(out)
                continue
            seed = int(sc.get("seed", 1000 + i))
            if gen:
                gen.generate(sc["prompt"], out, seed=seed)
            else:
                procedural_scene(sc["prompt"], out, seed)
            log.info(f"  image {i+1}/{len(todo)}: {key}")
            if progress_cb:
                progress_cb((i + 1) / max(1, len(todo)))
    finally:
        if gen:
            gen.close()
    # depth maps
    need_depth = [k for k in scenes if not (img_dir / f"{k}_depth.png").exists()]
    if need_depth and settings()["depth"]["enabled"] and (ROOT / "models" / "depth-anything-v2-small" / "config.json").exists():
        try:
            from .depth import DepthEstimator
            de = DepthEstimator(log)
            for k in need_depth:
                de.estimate(img_dir / f"{k}.png", img_dir / f"{k}_depth.png")
            de.close()
        except Exception as e:  # depth is an enhancement; never fail the episode for it
            log.warning(f"Depth estimation skipped: {e}")
    return "ai" if use_ai else "procedural"
