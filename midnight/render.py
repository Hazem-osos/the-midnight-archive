"""Frame renderer: 2.5D depth parallax camera moves, grading, grain, flicker,
transitions, subtitles and branding. Frames are piped straight into FFmpeg.
"""
from __future__ import annotations

import math
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .branding import AMBER, BONE, BRAND, build_logo, font
from .core import settings, which_ffmpeg

MARGIN = 1.22  # source overscan so the camera can move without showing edges


def ease(p):
    return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, p)))


def smooth_noise(n, seed, rate=6.0):
    r = np.random.default_rng(seed)
    pts = r.standard_normal(int(n / rate) + 3)
    x = np.arange(n) / rate
    return np.interp(x, np.arange(len(pts)), pts)


# ------------------------------------------------------------ image prep
def grade(img_bgr: np.ndarray, warm=False) -> np.ndarray:
    """Dark cinematic grade: crushed blacks, teal shadows, amber highlights, low saturation."""
    f = img_bgr.astype(np.float32) / 255.0
    lum = f.mean(axis=2, keepdims=True)
    f = lum + (f - lum) * 0.72                       # desaturate
    f = np.clip((f - 0.035) / 0.93, 0, 1) ** 1.12    # crush blacks, darken mids
    shadow = (1 - lum) ** 2
    hi = lum ** 2
    teal = np.array([0.045, 0.02, -0.02], np.float32)   # BGR
    amber = np.array([-0.04, 0.015, 0.06], np.float32) if warm else np.array([-0.02, 0.0, 0.02], np.float32)
    f = f + shadow * teal + hi * amber
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


class Plate:
    """A scene image + depth prepared at output resolution with overscan."""

    def __init__(self, image_path, depth_path, W, H, warm=False):
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        sw, sh = int(W * MARGIN), int(H * MARGIN)
        scale = max(sw / img.shape[1], sh / img.shape[0])
        img = cv2.resize(img, (int(img.shape[1] * scale + 0.5), int(img.shape[0] * scale + 0.5)), interpolation=cv2.INTER_LANCZOS4)
        y0, x0 = (img.shape[0] - sh) // 2, (img.shape[1] - sw) // 2
        img = img[y0:y0 + sh, x0:x0 + sw]
        # mild unsharp mask to counter upscale softness
        blur = cv2.GaussianBlur(img, (0, 0), 2.0)
        img = cv2.addWeighted(img, 1.35, blur, -0.35, 0)
        self.img = grade(img, warm)
        if depth_path and depth_path.exists():
            d = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
            d = cv2.resize(d, (int(d.shape[1] * scale + 0.5), int(d.shape[0] * scale + 0.5)))[y0:y0 + sh, x0:x0 + sw]
            d = cv2.GaussianBlur(d, (0, 0), 9).astype(np.float32) / 255.0
        else:
            yy = np.linspace(0, 1, sh, dtype=np.float32)[:, None]
            d = np.repeat(0.3 + 0.5 * yy, sw, axis=1)  # assume ground plane: nearer at bottom
        self.depth = d
        self.sw, self.sh = sw, sh
        # low-res depth for fast map building
        self.small_depth = cv2.resize(d, (W // 4, H // 4), interpolation=cv2.INTER_AREA)


MOVES = {
    #            zoom0  zoom1  panx0  panx1  pany0 pany1 parx  pary  dolly
    "push_in":      (1.00, 1.13, 0.00, 0.00, 0.00, -0.01, 0, 0, 0.07),
    "push_in_slow": (1.00, 1.07, 0.00, 0.00, 0.00, 0.00, 0, 0, 0.05),
    "dolly_out":    (1.16, 1.00, 0.00, 0.00, -0.01, 0.00, 0, 0, -0.07),
    "pan_left":     (1.07, 1.09, 0.06, -0.06, 0.00, 0.00, 34, 0, 0.0),
    "pan_right":    (1.07, 1.09, -0.06, 0.06, 0.00, 0.00, -34, 0, 0.0),
    "tilt_up":      (1.08, 1.10, 0.00, 0.00, 0.06, -0.06, 0, 30, 0.0),
    "slow_orbit":   (1.08, 1.12, 0.00, 0.00, 0.00, 0.00, 0, 0, 0.03),
    "narrator":     (1.02, 1.08, 0.00, 0.00, 0.00, 0.01, 0, 0, 0.03),
}


class Camera:
    def __init__(self, W, H):
        self.W, self.H = W, H
        w4, h4 = W // 4, H // 4
        gx, gy = np.meshgrid(np.arange(w4, dtype=np.float32) * 4 + 1.5, np.arange(h4, dtype=np.float32) * 4 + 1.5)
        self.u = gx - W / 2
        self.v = gy - H / 2

    def render(self, plate: Plate, move: str, p: float, t: float, shake=(0.0, 0.0)) -> np.ndarray:
        z0, z1, px0, px1, py0, py1, parx, pary, dolly = MOVES.get(move, MOVES["push_in_slow"])
        e = ease(p)
        zoom = z0 + (z1 - z0) * e
        if move == "narrator":
            zoom *= 1 + 0.004 * math.sin(2 * math.pi * 0.23 * t)  # breathing
        maxpx = (plate.sw - self.W / zoom) / 2
        maxpy = (plate.sh - self.H / zoom) / 2
        cx = plate.sw / 2 + np.clip((px0 + (px1 - px0) * e) * plate.sw, -maxpx * 0.8, maxpx * 0.8)
        cy = plate.sh / 2 + np.clip((py0 + (py1 - py0) * e) * plate.sh, -maxpy * 0.8, maxpy * 0.8)
        ox = parx * (e - 0.5) * 2
        oy = pary * (e - 0.5) * 2
        if move == "slow_orbit":
            ox, oy = 26 * math.sin(2 * math.pi * 0.08 * t), 14 * math.cos(2 * math.pi * 0.08 * t)
        if move == "narrator":
            ox, oy = 10 * math.sin(2 * math.pi * 0.05 * t), 5 * math.sin(2 * math.pi * 0.037 * t)
        d = plate.small_depth - 0.45
        local = zoom * (1 + dolly * e * (d + 0.45))
        mx = cx + self.u / local - d * ox + shake[0]
        my = cy + self.v / local - d * oy + shake[1]
        mx = cv2.resize(mx, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        my = cv2.resize(my, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return cv2.remap(plate.img, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ------------------------------------------------------------ overlays
class Overlay:
    def __init__(self, rgba: Image.Image, x: int, y: int):
        a = np.asarray(rgba.convert("RGBA"), dtype=np.float32)
        self.rgb = a[..., [2, 1, 0]]
        self.alpha = a[..., 3:4] / 255.0
        self.x, self.y = x, y
        self.h, self.w = a.shape[:2]

    def draw(self, frame: np.ndarray, opacity=1.0):
        H, W = frame.shape[:2]
        x0, y0 = max(0, self.x), max(0, self.y)
        x1, y1 = min(W, self.x + self.w), min(H, self.y + self.h)
        if x1 <= x0 or y1 <= y0 or opacity <= 0:
            return
        sx, sy = x0 - self.x, y0 - self.y
        a = self.alpha[sy:sy + y1 - y0, sx:sx + x1 - x0] * opacity
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = (roi * (1 - a) + self.rgb[sy:sy + y1 - y0, sx:sx + x1 - x0] * a).astype(np.uint8)


def text_image(lines, fnt, fill, stroke=0, stroke_fill=(0, 0, 0), shadow=True, spacing=10, align="center", max_w=None):
    """Render multi-line text (list of [(word, color), ...] or strings) to an RGBA image."""
    if isinstance(lines, str):
        lines = [lines]
    norm_lines = [[(ln, fill)] if isinstance(ln, str) else ln for ln in lines]
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    space = tmp.textlength(" ", font=fnt)
    asc, desc = fnt.getmetrics()
    lh = asc + desc
    widths = [sum(tmp.textlength(w, font=fnt) for w, _ in ln) + space * (len(ln) - 1) for ln in norm_lines]
    pad = stroke + 24
    Wt = int(max(widths) + pad * 2)
    Ht = int(lh * len(norm_lines) + spacing * (len(norm_lines) - 1) + pad * 2)
    img = Image.new("RGBA", (Wt, Ht), (0, 0, 0, 0))
    if shadow:
        sh = Image.new("RGBA", (Wt, Ht), (0, 0, 0, 0))
    for layer in ([sh] if shadow else []) + [img]:
        d = ImageDraw.Draw(layer)
        y = pad
        for ln, lw in zip(norm_lines, widths):
            x = pad + ((Wt - 2 * pad - lw) / 2 if align == "center" else 0)
            for w, col in ln:
                if layer is img:
                    d.text((x, y), w, font=fnt, fill=col, stroke_width=stroke, stroke_fill=stroke_fill)
                else:
                    d.text((x + 4, y + 6), w, font=fnt, fill=(0, 0, 0, 200), stroke_width=stroke + 4, stroke_fill=(0, 0, 0, 200))
                x += d.textlength(w, font=fnt) + space
            y += lh + spacing
    if shadow:
        sh = sh.filter(ImageFilter.GaussianBlur(10))
        sh.alpha_composite(img)
        img = sh
    return img


def subtitle_groups(timeline, max_words=3, max_chars=17):
    """Split sentence word timings into short on-screen groups."""
    groups = []
    for seg in timeline["segments"]:
        for s in seg["sentences"]:
            cur = []
            for w in s["words"]:
                trial = " ".join(x["w"] for x in cur + [w])
                if cur and (len(cur) >= max_words or len(trial) > max_chars or cur[-1]["w"][-1] in ",;:—"):
                    groups.append({"words": cur, "voice": seg["voice"]})
                    cur = []
                cur.append(w)
            if cur:
                groups.append({"words": cur, "voice": seg["voice"]})
    for i, g in enumerate(groups):
        g["start"] = g["words"][0]["start"]
        nxt = groups[i + 1]["words"][0]["start"] if i + 1 < len(groups) else g["words"][-1]["end"] + 0.4
        g["end"] = min(g["words"][-1]["end"] + 0.35, nxt)
    return groups


class Subtitles:
    def __init__(self, groups, W, H):
        self.groups = groups
        self.cache = {}
        self.W, self.H = W, H
        self.fnt = font("subs", 92)
        self.y_center = int(H * 0.655)

    def _overlay(self, gi, active):
        key = (gi, active)
        if key not in self.cache:
            g = self.groups[gi]
            tape = g["voice"] != "elias"
            base = (235, 200, 200) if tape else (255, 255, 255)
            hl = (255, 70, 60) if tape else AMBER
            words = [(w["w"].upper(), hl if i == active else base) for i, w in enumerate(g["words"])]
            img = text_image([words], self.fnt, base, stroke=7)
            if img.width > self.W - 80:
                img = img.resize((self.W - 80, int(img.height * (self.W - 80) / img.width)), Image.LANCZOS)
            self.cache[key] = Overlay(img, (self.W - img.width) // 2, self.y_center - img.height // 2)
        return self.cache[key]

    def draw(self, frame, t):
        for gi, g in enumerate(self.groups):
            if g["start"] <= t < g["end"]:
                active = max([i for i, w in enumerate(g["words"]) if w["start"] <= t] or [0])
                ov = self._overlay(gi, active)
                pop = min(1.0, (t - g["start"]) / 0.08)
                ov.draw(frame, pop)
                return


def write_srt(groups, path):
    def ts(x):
        ms = int(round(x * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    with open(path, "w", encoding="utf-8") as f:
        for i, g in enumerate(groups, 1):
            f.write(f"{i}\n{ts(g['start'])} --> {ts(g['end'])}\n{' '.join(w['w'] for w in g['words'])}\n\n")


# ------------------------------------------------------------ main render
def build_clips(script, timeline, total, xf):
    segs = timeline["segments"]
    clips = []
    for i, s in enumerate(segs):
        start = 0.0 if i == 0 else s["start"]
        end = segs[i + 1]["start"] if i + 1 < len(segs) else timeline["narration_end"] + 0.3
        if clips and clips[-1]["scene"] == s["scene"]:
            clips[-1]["end"] = end
            clips[-1]["segments"].append(i)
            continue
        clips.append({"scene": s["scene"], "start": start, "end": end, "segments": [i], "stamp": script["segments"][i].get("stamp")})
    return clips


def render_video(ep_dir, script, timeline, events, out_path, log, progress_cb=None):
    cfg = settings()["video"]
    W, H, FPS = cfg["width"], cfg["height"], cfg["fps"]
    xf = cfg["crossfade"]
    end_len = cfg["end_card_seconds"]
    end_start = timeline["narration_end"] + 0.1
    total = end_start + end_len
    n_frames = int(round(total * FPS))
    clips = build_clips(script, timeline, total, xf)
    clips[-1]["end"] = end_start + 0.6

    build_logo()
    cam = Camera(W, H)
    plates = {}
    scenes = script["scenes"]
    for c in clips:
        if c["scene"] not in plates:
            sc = scenes[c["scene"]]
            plates[c["scene"]] = Plate(ep_dir / "images" / f"{c['scene']}.png", ep_dir / "images" / f"{c['scene']}_depth.png",
                                       W, H, warm=sc.get("character") is not None)

    # static overlays
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.46) / (H * 0.6)) ** 2)
    vign = np.clip(1.15 - r ** 2.2 * 0.85, 0.18, 1.0)
    vign = (np.dstack([vign] * 3) * 255).astype(np.uint8)
    rng = np.random.default_rng(7)
    grains = []
    for _ in range(10):
        g = rng.normal(0, 9, (H // 2, W // 2)).astype(np.float32)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
        grains.append((np.clip(g, 0, 255).astype(np.uint8), np.clip(-g, 0, 255).astype(np.uint8)))
    flick = smooth_noise(n_frames + 10, 11, rate=3.0)
    flick_slow = smooth_noise(n_frames + 10, 12, rate=14.0)

    ep_no = script.get("episode_number", 1)
    brand = Overlay(text_image(["THE MIDNIGHT ARCHIVE"], font("brand", 38, "Bold"), BONE, shadow=True), 0, 118)
    brand.x = (W - brand.w) // 2
    filetag = Overlay(text_image([f"FILE {ep_no:03d}"], font("stamp", 34), AMBER, shadow=True), 0, 170)
    filetag.x = (W - filetag.w) // 2
    hook_text = script.get("hook_text")
    hook = None
    if hook_text:
        words = hook_text.upper().split()
        mid = (len(words) + 1) // 2
        hook_img = text_image([" ".join(words[:mid]), [(" ".join(words[mid:]), (230, 40, 40))]], font("subs", 130), (255, 255, 255), stroke=9, spacing=6)
        if hook_img.width > W - 60:
            hook_img = hook_img.resize((W - 60, int(hook_img.height * (W - 60) / hook_img.width)), Image.LANCZOS)
        hook = Overlay(hook_img, (W - hook_img.width) // 2, int(H * 0.25))
    stamp_cache = {}
    logo = Image.open(BRAND / "logo.png").convert("RGBA")
    logo = logo.resize((int(W * 0.86), int(logo.height * W * 0.86 / logo.width)), Image.LANCZOS)
    logo_ov = Overlay(logo, (W - logo.width) // 2, int(H * 0.30))
    follow = Overlay(text_image(["FOLLOW FOR THE NEXT FILE"], font("stamp", 44), BONE), 0, int(H * 0.30) + logo.height + 30)
    follow.x = (W - follow.w) // 2

    groups = subtitle_groups(timeline)
    write_srt(groups, ep_dir / "subtitles.srt")
    subs = Subtitles(groups, W, H)

    shakes = [e for e in events if e["name"] in ("boom", "stinger")]
    glitches = [e["t"] for e in events if e["name"] in ("glitch", "stinger")]
    for c in clips:
        if "glitch" in scenes[c["scene"]].get("fx", []):
            glitches.append(c["start"] + 0.05)

    ff, _ = which_ffmpeg()
    raw = ep_dir / "video_noaudio.mp4"
    cmd = [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", cfg["preset"], "-crf", str(cfg["crf"]), "-pix_fmt", "yuv420p", "-r", str(FPS), str(raw)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    dust = rng.uniform(0, 1, (70, 4)).astype(np.float32)

    try:
        for fi in range(n_frames):
            t = fi / FPS
            # camera shake after impacts
            sx = sy = 0.0
            for e in shakes:
                dt = t - e["t"]
                if 0 <= dt < 0.7:
                    amp = 14 * math.exp(-dt * 6)
                    sx += amp * math.sin(dt * 71)
                    sy += amp * math.cos(dt * 53)
            frame = None
            if t < end_start + 0.6:
                active = [c for c in clips if c["start"] - xf / 2 <= t < c["end"] + xf / 2]
                layers = []
                for c in active:
                    dur = c["end"] - c["start"]
                    p = (t - c["start"] + xf / 2) / (dur + xf)
                    move = scenes[c["scene"]].get("move", "push_in_slow")
                    img = cam.render(plates[c["scene"]], move, p, t, (sx, sy))
                    w_in = min(1.0, (t - (c["start"] - xf / 2)) / xf) if c is not clips[0] else 1.0
                    layers.append((img, w_in, c))
                frame = layers[0][0]
                for img, w_in, _c in layers[1:]:
                    frame = cv2.addWeighted(frame, 1 - w_in, img, w_in, 0)
                cur = layers[-1][2]
                warm = scenes[cur["scene"]].get("character") is not None
                b = 1.0 + (0.045 * flick[fi] if warm else 0.02 * flick[fi]) + 0.03 * flick_slow[fi]
                frame = cv2.convertScaleAbs(frame, alpha=b, beta=0)
            # end card
            if t >= end_start:
                k = min(1.0, (t - end_start) / 0.6)
                black = np.zeros((H, W, 3), np.uint8) + np.array([9, 7, 8], np.uint8)
                frame = black if frame is None else cv2.addWeighted(frame, 1 - k, black, k, 0)
            frame = cv2.multiply(frame, vign, scale=1 / 255)
            # floating dust in light
            if t < end_start:
                for (dx, dy, ds, dv) in dust:
                    px = int(((dx + 0.01 * math.sin(t * 0.7 + dv * 9)) % 1) * W)
                    py = int(((dy - t * (0.004 + 0.01 * dv)) % 1) * H)
                    cv2.circle(frame, (px, py), int(1 + ds * 2.5), (150, 175, 195), -1, lineType=cv2.LINE_AA)
            gp, gn = grains[fi % len(grains)]
            frame = cv2.subtract(cv2.add(frame, gp), gn)
            # glitch
            for gt in glitches:
                if 0 <= t - gt < 0.22 and (fi % 2 == 0):
                    sh = int(18 * rng.uniform(0.5, 1))
                    frame[..., 2] = np.roll(frame[..., 2], sh, axis=1)
                    frame[..., 0] = np.roll(frame[..., 0], -sh, axis=1)
                    for _ in range(5):
                        y0 = int(rng.uniform(0, H - 80))
                        hh = int(rng.uniform(10, 80))
                        frame[y0:y0 + hh] = np.roll(frame[y0:y0 + hh], int(rng.uniform(-60, 60)), axis=1)
            # overlays
            if t < end_start:
                brand.draw(frame, 0.85)
                filetag.draw(frame, 0.85)
                cur_clip = next((c for c in clips if c["start"] <= t < c["end"]), clips[-1])
                if cur_clip.get("stamp"):
                    st = cur_clip["stamp"]
                    n_chars = int(min(len(st), max(0, (t - cur_clip["start"] - 0.2) * 22)))
                    if n_chars > 0:
                        key = (st, n_chars)
                        if key not in stamp_cache:
                            stamp_cache[key] = Overlay(text_image([st[:n_chars]], font("stamp", 46), (232, 232, 225), align="left"), 60, 300)
                        fade_out = min(1.0, max(0.0, (cur_clip["end"] - t) / 0.4))
                        stamp_cache[key].draw(frame, 0.9 * fade_out)
                if hook and t < timeline["segments"][0]["end"]:
                    hk = min(1.0, t / 0.25) * min(1.0, (timeline["segments"][0]["end"] - t) / 0.3)
                    hook.draw(frame, hk)
                subs.draw(frame, t)
            else:
                k = min(1.0, (t - end_start - 0.3) / 0.7)
                logo_ov.draw(frame, max(0, k))
                follow.draw(frame, max(0.0, min(1.0, (t - end_start - 1.0) / 0.5)))
            proc.stdin.write(frame.tobytes())
            if progress_cb and fi % 30 == 0:
                progress_cb(fi / n_frames)
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="ignore")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg encode failed: {err[-1500:]}")
    except Exception:
        proc.kill()
        raise
    log.info(f"Rendered {n_frames} frames ({total:.1f}s)")
    return raw, total
