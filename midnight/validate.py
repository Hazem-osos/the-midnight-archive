"""Automatic validation of the final MP4 (specs, streams, A/V sync, decodability)."""
from __future__ import annotations

import json
import subprocess

from .core import run, settings, which_ffmpeg


def probe(path) -> dict:
    _, fp = which_ffmpeg()
    return json.loads(run([fp, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]).stdout)


def validate_video(path, srt_path=None) -> dict:
    cfg = settings()["video"]
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})

    try:
        info = probe(path)
    except Exception as e:  # noqa
        return {"ok": False, "checks": [{"check": "ffprobe readable", "ok": False, "detail": str(e)[:300]}]}
    streams = info["streams"]
    v = next((s for s in streams if s["codec_type"] == "video"), None)
    a = next((s for s in streams if s["codec_type"] == "audio"), None)
    s = next((s for s in streams if s["codec_type"] == "subtitle"), None)
    fmt = info["format"]
    check("container is MP4", "mp4" in fmt.get("format_name", ""), fmt.get("format_name"))
    check("video stream present", v is not None)
    check("audio stream present", a is not None)
    if v:
        check("resolution 1080x1920", (v["width"], v["height"]) == (cfg["width"], cfg["height"]), f"{v['width']}x{v['height']}")
        check("codec H.264", v["codec_name"] == "h264", v["codec_name"])
        num, den = map(int, v["r_frame_rate"].split("/"))
        check("30 fps", abs(num / den - cfg["fps"]) < 0.01, f"{num/den:.3f}")
        check("pixel format yuv420p", v.get("pix_fmt") == "yuv420p", v.get("pix_fmt"))
    if a:
        check("audio codec AAC", a["codec_name"] == "aac", a["codec_name"])
        check("audio 48 kHz", a.get("sample_rate") == "48000", a.get("sample_rate"))
    dur = float(fmt.get("duration", 0))
    check(f"duration {cfg['min_duration']}-{cfg['max_duration']}s", cfg["min_duration"] <= dur <= cfg["max_duration"], f"{dur:.2f}s")
    if v and a:
        vd, ad = float(v.get("duration", dur)), float(a.get("duration", dur))
        check("audio/video length match (<0.25s)", abs(vd - ad) < 0.25, f"video {vd:.2f}s / audio {ad:.2f}s")
        check("streams start together", abs(float(v.get("start_time", 0)) - float(a.get("start_time", 0))) < 0.1,
              f"v {v.get('start_time')} a {a.get('start_time')}")
    check("soft subtitle track (eng)", s is not None and s.get("codec_name") == "mov_text", s.get("codec_name") if s else "none")
    if srt_path:
        try:
            n = open(srt_path, encoding="utf-8").read().count("-->")
            check("burned-in subtitles generated", n > 10, f"{n} cues")
        except OSError:
            check("burned-in subtitles generated", False, "srt missing")
    # full decode test (catches corrupt frames)
    ff, _ = which_ffmpeg()
    r = subprocess.run([ff, "-v", "error", "-i", str(path), "-f", "null", "-"], capture_output=True, text=True)
    check("decodes without errors", r.returncode == 0 and not r.stderr.strip(), r.stderr.strip()[:200] or "clean")
    # loudness / silence sanity
    r = subprocess.run([ff, "-i", str(path), "-af", "volumedetect", "-vn", "-f", "null", "-"], capture_output=True, text=True)
    mean = next((ln.split(":")[1].strip() for ln in r.stderr.splitlines() if "mean_volume" in ln), "?")
    try:
        check("audio not silent", float(mean.split()[0]) > -40, mean)
    except ValueError:
        check("audio not silent", False, mean)
    return {"ok": all(c["ok"] for c in checks), "duration": dur, "checks": checks}
