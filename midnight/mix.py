"""Final sound mix: narration + procedural ambience + SFX cues, ducking and loudness."""
from __future__ import annotations

import numpy as np
import soundfile as sf

from . import sfx
from .core import run, which_ffmpeg

SR = sfx.SR


def resolve_events(script, timeline, end_start):
    """Turn script sfx cues ('at': seconds | 'end' | 'end-1.5') into absolute times."""
    events = []
    for seg, tl in zip(script["segments"], timeline["segments"]):
        for cue in seg.get("sfx", []):
            at = cue.get("at", 0)
            if isinstance(at, str) and at.startswith("end"):
                t = tl["speech_end"] + (float(at[3:]) if len(at) > 3 else 0.0)
            else:
                t = tl["start"] + float(at)
            ev = {"name": cue["name"], "t": max(0.0, t), "gain": cue.get("gain")}
            if cue.get("dur") == "segment":
                ev["dur"] = tl["end"] - tl["start"]
            elif "dur" in cue:
                ev["dur"] = float(cue["dur"])
            events.append(ev)
    # subtle whoosh on every scene change
    prev = None
    for tl in timeline["segments"]:
        if prev is not None and tl["scene"] != prev:
            events.append({"name": "whoosh", "t": max(0, tl["start"] - 0.45)})
        prev = tl["scene"]
    events.append({"name": "logo_sting", "t": end_start + 0.2})
    return sorted(events, key=lambda e: e["t"])


def mix(ep_dir, script, timeline, total, events, log):
    narr, sr = sf.read(ep_dir / "narration.wav", dtype="float32")
    assert sr == SR
    n = int((total + 0.05) * SR)
    voice = np.zeros(n, np.float32)
    voice[: min(n, len(narr))] = narr[:n]

    # ambience bed with intensity rising towards the twist, ducked under speech
    bed = sfx.drone(total + 0.5, seed=len(script["title"]))[:n]
    bed = np.pad(bed, (0, max(0, n - len(bed))))
    twist = next((s["start"] for s in timeline["segments"] if s["type"] == "twist"), total * 0.75)
    tt = np.arange(n) / SR
    intensity = 0.55 + 0.45 * np.clip(tt / max(1, twist), 0, 1)
    envw = int(0.05 * SR)
    venv = np.sqrt(np.convolve(voice ** 2, np.ones(envw) / envw, mode="same"))
    venv = np.convolve(venv, np.ones(int(0.3 * SR)) / int(0.3 * SR), mode="same")
    duck = 1 - 0.45 * np.clip(venv / (venv.max() + 1e-9) * 4, 0, 1)
    bed_gain = 0.16

    fxtrack = np.zeros(n, np.float32)
    for e in events:
        gen = sfx.LIBRARY.get(e["name"])
        if not gen:
            log.warning(f"unknown sfx {e['name']}")
            continue
        snd = gen(dur=e.get("dur", 5.0)) if "dur" in e else gen()
        g = e.get("gain") or sfx.GAINS.get(e["name"], 0.5)
        i = int(e["t"] * SR)
        if i >= n:
            continue
        m = min(len(snd), n - i)
        fxtrack[i:i + m] += snd[:m] * g

    # stereo: voice centered; bed and fx get a slight decorrelated width
    left = voice + bed * bed_gain * intensity * duck + fxtrack
    right = voice + np.roll(bed, int(0.011 * SR)) * bed_gain * intensity * duck + np.roll(fxtrack, int(0.006 * SR))
    stereo = np.stack([left, right], axis=1)
    stereo = np.tanh(stereo * 1.1) / np.tanh(1.1)  # soft limiter
    pre = ep_dir / "mix_pre.wav"
    sf.write(pre, stereo.astype(np.float32), SR)
    ff, _ = which_ffmpeg()
    out = ep_dir / "final_audio.wav"
    run([ff, "-y", "-v", "error", "-i", str(pre), "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", str(SR), "-ac", "2", str(out)])
    pre.unlink(missing_ok=True)
    log.info(f"Mixed audio with {len(events)} sound events")
    return out
