"""Narration with Kokoro-82M (ONNX, local, Apache-2.0).

Each segment is split into sentences; every sentence is synthesized separately so
we know its exact start/end time. Word times inside a sentence are estimated from
word length (no forced aligner needed). Voice FX is applied per voice track with
FFmpeg, then all tracks are summed into one 48 kHz narration stem.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

from .core import ROOT, character, run, settings, which_ffmpeg
from .textutil import split_chunks

SR_TTS = 24000
SR_OUT = 48000

FX_CHAINS = {
    # Elias: warm, close, slightly darker, subtle archive room.
    "elias": "highpass=f=55,bass=g=3.5:f=110:w=0.7,treble=g=-1.5:f=6500,"
             "acompressor=threshold=-20dB:ratio=2.8:attack=8:release=180:makeup=2,"
             "aecho=0.85:0.55:38|67|113:0.10|0.07|0.04",
    # Recovered tape: band-limited, wobbly, slightly distorted.
    "tape": "highpass=f=320,lowpass=f=3300,vibrato=f=0.9:d=0.12,"
            "acompressor=threshold=-25dB:ratio=6:attack=2:release=60:makeup=3,"
            "aecho=0.7:0.6:23:0.25",
    "none": "anull",
}

_kokoro = None


def _engine():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        v = character()["voice"]
        _kokoro = Kokoro(str(ROOT / v["model_file"]), str(ROOT / v["voices_file"]))
    return _kokoro


def voice_style(blend: dict):
    k = _engine()
    style = None
    for name, w in blend.items():
        s = k.get_voice_style(name) * float(w)
        style = s if style is None else style + s
    return style


def _trim(audio: np.ndarray, thresh=0.012, pad=0.03) -> np.ndarray:
    idx = np.where(np.abs(audio) > thresh)[0]
    if len(idx) == 0:
        return audio
    p = int(pad * SR_TTS)
    return audio[max(0, idx[0] - p): min(len(audio), idx[-1] + p)]


def _word_times(display: str, t0: float, t1: float) -> list[dict]:
    ws = display.split()
    weights = np.array([len([c for c in w if c.isalnum()]) + 2.2 for w in ws], dtype=float)
    # punctuation inside the sentence -> the speaker lingers a little
    weights += np.array([1.5 if w[-1] in ",;:—-" else 0 for w in ws])
    edges = t0 + (t1 - t0) * np.concatenate([[0], np.cumsum(weights) / weights.sum()])
    return [{"w": w, "start": round(float(edges[i]), 3), "end": round(float(edges[i + 1]), 3)} for i, w in enumerate(ws)]


def synthesize(script: dict, out_dir, log) -> dict:
    """Render narration.wav (48 kHz mono) and timeline.json data for a script."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ch = character()
    tm = settings()["timing"]
    voices = {"elias": ch["voice"], **{k: v for k, v in ch.get("secondary_voices", {}).items()}}
    styles = {}
    tracks: dict[str, list] = {}
    t = float(tm["lead_in"])
    timeline = {"segments": []}
    k = _engine()

    for si, seg in enumerate(script["segments"]):
        vkey = seg.get("voice", "elias")
        vcfg = voices[vkey]
        if vkey not in styles:
            styles[vkey] = voice_style(vcfg["blend"])
        fx = vcfg.get("fx", "elias") if vkey != "elias" else "elias"
        seg_start = t
        sentences = []
        for ch_ in split_chunks(seg["text"]):
            audio, sr = k.create(ch_["spoken"], voice=styles[vkey], speed=float(vcfg["speed"]), lang=vcfg.get("lang", "en-us"))
            assert sr == SR_TTS, sr
            audio = _trim(np.asarray(audio, dtype=np.float32))
            dur = len(audio) / SR_TTS
            tracks.setdefault(fx, []).append((t, audio))
            sentences.append({"text": ch_["display"], "start": round(t, 3), "end": round(t + dur, 3),
                              "words": _word_times(ch_["display"], t, t + dur)})
            t += dur + (tm["ellipsis_pause"] if ch_["pause"] == "ellipsis" else tm["sentence_gap"])
        t += max(0.0, tm["segment_gap"] - tm["sentence_gap"]) + float(seg.get("extra_pause", 0))
        timeline["segments"].append({"index": si, "type": seg.get("type", "story"), "scene": seg.get("scene"),
                                     "voice": vkey, "start": round(seg_start, 3), "end": round(t, 3),
                                     "speech_end": sentences[-1]["end"] if sentences else seg_start,
                                     "sentences": sentences})
        log.info(f"  narration segment {si+1}/{len(script['segments'])} done ({t:.1f}s)")

    total = t
    timeline["narration_end"] = round(total, 3)
    ff, _ = which_ffmpeg()
    mix = np.zeros(int((total + 2.5) * SR_OUT), dtype=np.float32)
    for fx, items in tracks.items():
        raw = np.zeros(int((total + 2.0) * SR_TTS), dtype=np.float32)
        for start, audio in items:
            i = int(start * SR_TTS)
            raw[i:i + len(audio)] += audio[: len(raw) - i]
        rp, fp = out_dir / f"voice_{fx}_raw.wav", out_dir / f"voice_{fx}_fx.wav"
        sf.write(rp, raw, SR_TTS)
        run([ff, "-y", "-v", "error", "-i", str(rp), "-af", FX_CHAINS.get(fx, "anull"), "-ar", str(SR_OUT), "-ac", "1", str(fp)])
        processed, _ = sf.read(fp, dtype="float32")
        n = min(len(processed), len(mix))
        mix[:n] += processed[:n]
        rp.unlink(missing_ok=True)
    peak = float(np.abs(mix).max() or 1.0)
    mix *= 0.89 / peak
    sf.write(out_dir / "narration.wav", mix, SR_OUT)
    return timeline
