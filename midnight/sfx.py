"""Procedural horror sound design.

Every sound here is synthesized from scratch with numpy (sines, noise, filters,
convolution reverb), so it is 100% original and royalty-free. No samples are
downloaded.
"""
from __future__ import annotations

import numpy as np

SR = 48000
RNG = np.random.default_rng(317)


# ---------------------------------------------------------------- primitives
def t_axis(dur):
    return np.arange(int(dur * SR)) / SR


def fft_filter(x, lo=None, hi=None, slope=1.5):
    """Zero-phase band filter in the frequency domain (smooth roll-off)."""
    n = len(x)
    if n == 0:
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    g = np.ones_like(f)
    if lo:
        g *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1e-3)) ** (2 * slope))
    if hi:
        g *= 1 / np.sqrt(1 + (f / hi) ** (2 * slope))
    return np.fft.irfft(X * g, n).astype(np.float32)


def noise(dur, color="white"):
    n = RNG.standard_normal(int(dur * SR)).astype(np.float32)
    if color == "brown":
        n = np.cumsum(n)
        n -= np.convolve(n, np.ones(4801) / 4801, mode="same")
        n /= np.abs(n).max() + 1e-9
    elif color == "pink":
        X = np.fft.rfft(n)
        f = np.fft.rfftfreq(len(n), 1 / SR)
        X /= np.sqrt(np.maximum(f, 20))
        n = np.fft.irfft(X, len(n)).astype(np.float32)
        n /= np.abs(n).max() + 1e-9
    return n


def env_exp(dur, decay):
    return np.exp(-t_axis(dur) / decay).astype(np.float32)


def fade(x, fin=0.01, fout=0.05):
    x = x.copy()
    a, b = int(fin * SR), int(fout * SR)
    if a:
        x[:a] *= np.linspace(0, 1, a)
    if b:
        x[-b:] *= np.linspace(1, 0, b)
    return x


def impulse_response(dur=2.2, decay=0.55, lo=150, hi=6000, seed=0):
    r = np.random.default_rng(seed)
    ir = r.standard_normal(int(dur * SR)).astype(np.float32) * env_exp(dur, decay)
    ir = fft_filter(ir, lo, hi)
    ir[0] = 1.0
    return ir / np.abs(ir).sum() * 40


def reverb(x, wet=0.35, dur=2.2, decay=0.55, seed=0, lo=150, hi=6000):
    ir = impulse_response(dur, decay, lo, hi, seed)
    n = len(x) + len(ir) - 1
    nfft = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[:n].astype(np.float32)
    y /= np.abs(y).max() + 1e-9
    dry = np.zeros(n, dtype=np.float32)
    dry[: len(x)] = x / (np.abs(x).max() + 1e-9)
    return (1 - wet) * dry + wet * y


def norm(x, peak=1.0):
    return (x / (np.abs(x).max() + 1e-9) * peak).astype(np.float32)


# ---------------------------------------------------------------- sounds
def knock(n=3, spacing=0.62, far=False):
    one_dur = 0.35
    t = t_axis(one_dur)
    hits = []
    for i in range(n):
        body = sum(a * np.sin(2 * np.pi * f * t + RNG.uniform(0, 6)) * np.exp(-t / d)
                   for f, a, d in [(118, 1.0, 0.07), (186, 0.8, 0.05), (297, 0.5, 0.035), (455, 0.3, 0.02)])
        click = fft_filter(noise(one_dur), 300, 2500) * np.exp(-t / 0.006) * 1.6
        hits.append(norm(body + click) * (1.0 - 0.06 * i))
    out = np.zeros(int((spacing * (n - 1) + one_dur) * SR), dtype=np.float32)
    for i, h in enumerate(hits):
        j = int(i * spacing * SR * RNG.uniform(0.96, 1.04))
        out[j:j + len(h)] += h[: len(out) - j]
    if far:
        out = fft_filter(out, 60, 900)
        return norm(reverb(out, wet=0.55, decay=0.8, seed=3), 0.55)
    return norm(reverb(out, wet=0.22, decay=0.35, seed=1), 0.95)


def boom(dur=3.5):
    t = t_axis(dur)
    f = 62 * np.exp(-t / 1.4) + 26
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 1.1)
    hit = fft_filter(noise(dur, "brown"), 20, 400) * np.exp(-t / 0.25)
    return norm(reverb(norm(sub) + 0.6 * norm(hit), wet=0.4, dur=3, decay=1.0, seed=4, lo=30, hi=2000), 1.0)


def stinger(dur=4.0):
    """Dissonant orchestral-style hit: detuned harmonic cluster + boom."""
    t = t_axis(dur)
    tone = np.zeros_like(t, dtype=np.float32)
    for base in (146.8, 207.7, 220.0, 311.1, 415.3):  # D, G#, A, D#, G# - tritones/minor 2nds
        for h in range(1, 7):
            tone += (np.sin(2 * np.pi * base * h * t * RNG.uniform(0.998, 1.002)) / h ** 1.2).astype(np.float32)
    att = np.minimum(1, t / 0.015)
    tone = fft_filter(tone * att * np.exp(-t / 1.3), 80, 5000)
    screech = fft_filter(noise(dur), 2500, 7000) * np.exp(-t / 0.5) * 0.25
    hit = norm(reverb(norm(tone) + screech, wet=0.45, dur=3.5, decay=1.2, seed=5), 0.85)
    return _mix_len(hit, boom(dur) * 0.9)


def _mix_len(a, b):
    n = max(len(a), len(b))
    out = np.zeros(n, dtype=np.float32)
    out[: len(a)] += a
    out[: len(b)] += b
    return norm(out, 1.0)


def riser(dur=2.5):
    t = t_axis(dur)
    f = 180 * (6 ** (t / dur))
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.35
    nz = np.zeros_like(t, dtype=np.float32)
    raw = noise(dur)
    hop = 4096
    win = np.hanning(hop * 2).astype(np.float32)
    for s in range(0, len(t) - hop * 2, hop):
        c = 300 * (12 ** (s / len(t)))
        seg = fft_filter(raw[s:s + hop * 2] * win, c * 0.6, c * 1.8)
        nz[s:s + hop * 2] += seg
    env = (t / dur) ** 2.2
    return norm(reverb((tone + norm(nz)) * env, wet=0.3, decay=0.6, seed=6), 0.8)


def reverse_swell(dur=1.8):
    t = t_axis(dur)
    src = norm(sum(np.sin(2 * np.pi * f * t) for f in (110, 164.8, 233.1)) * np.exp(-t / 0.15))
    src = norm(src + 0.5 * fft_filter(noise(dur), 200, 3000) * np.exp(-t / 0.1))
    wet = reverb(src, wet=0.9, dur=dur, decay=0.7, seed=7)[: int(dur * SR)]
    return norm(fade(wet[::-1], 0.05, 0.005), 0.8)


def tape_click():
    t = t_axis(0.25)
    clk = fft_filter(noise(0.25), 800, 6000) * np.exp(-t / 0.004)
    clunk = np.sin(2 * np.pi * 90 * t) * np.exp(-t / 0.03) * 0.6
    return norm(reverb(clk + clunk, wet=0.2, decay=0.2), 0.7)


def tape_hiss(dur):
    t = t_axis(dur)
    h = fft_filter(noise(dur), 2000, 9000) * 0.25
    hum = (np.sin(2 * np.pi * 60 * t) + 0.4 * np.sin(2 * np.pi * 120 * t)) * 0.08
    wow = 1 + 0.15 * np.sin(2 * np.pi * 0.7 * t)
    return fade(((h + hum) * wow).astype(np.float32), 0.05, 0.2)


def heartbeat(beats=4, bpm=62):
    period = 60 / bpm
    out = np.zeros(int((beats * period + 0.6) * SR), dtype=np.float32)
    t = t_axis(0.35)
    thump = norm(np.sin(2 * np.pi * 48 * t) * np.exp(-t / 0.06) + 0.3 * fft_filter(noise(0.35), 20, 150) * np.exp(-t / 0.05))
    for b in range(beats):
        for off, amp in ((0, 1.0), (0.24, 0.7)):
            j = int((b * period + off) * SR)
            out[j:j + len(thump)] += thump[: len(out) - j] * amp
    return norm(out, 0.9)


def watch_tick(n=4, interval=0.5):
    t = t_axis(0.06)
    tick = fft_filter(noise(0.06), 2500, 9000) * np.exp(-t / 0.004) + np.sin(2 * np.pi * 3400 * t) * np.exp(-t / 0.008) * 0.4
    out = np.zeros(int((n * interval + 0.5) * SR), dtype=np.float32)
    for i in range(n):
        j = int(i * interval * SR)
        amp = 1.0 if i % 2 == 0 else 0.7
        out[j:j + len(tick)] += tick * amp
    return norm(reverb(out, wet=0.25, decay=0.3, seed=8), 0.6)


def whoosh(dur=0.9):
    t = t_axis(dur)
    env = np.sin(np.pi * t / dur) ** 2
    return norm(fft_filter(noise(dur, "pink"), 150, 2200) * env, 0.5)


def glitch(dur=0.6):
    out = np.zeros(int(dur * SR), dtype=np.float32)
    pos = 0
    while pos < len(out):
        L = int(RNG.uniform(0.015, 0.06) * SR)
        if RNG.random() < 0.6:
            f = RNG.choice([220, 440, 880, 1760, 3520])
            tt = np.arange(L) / SR
            out[pos:pos + L] = (np.sign(np.sin(2 * np.pi * f * tt)) * 0.4)[: len(out) - pos]
        pos += L
    return norm(fft_filter(out, 100, 8000), 0.5)


def drone(dur, seed=0):
    """Ambient bed: detuned sub tones, slow movement, room tone and a faint high whine."""
    r = np.random.default_rng(seed)
    t = t_axis(dur)
    lfo = lambda f, ph=0: 0.5 + 0.5 * np.sin(2 * np.pi * f * t + ph)  # noqa: E731
    low = (np.sin(2 * np.pi * 41.2 * t) * lfo(0.05)
           + np.sin(2 * np.pi * 41.7 * t) * lfo(0.07, 1)
           + 0.6 * np.sin(2 * np.pi * 61.7 * t) * lfo(0.031, 2)
           + 0.35 * np.sin(2 * np.pi * 87.3 * t + 0.3 * np.sin(2 * np.pi * 0.2 * t)) * lfo(0.043, 3))
    room = fft_filter(noise(dur, "brown"), 25, 300) * 0.8
    air = fft_filter(noise(dur, "pink"), 400, 2500) * 0.12 * lfo(0.021, 4)
    whine = np.sin(2 * np.pi * 2349 * t + 2 * np.sin(2 * np.pi * 0.13 * t)) * 0.015 * lfo(0.017, 5)
    bed = norm(low) * 0.8 + norm(room) * 0.45 + air + whine
    bed = reverb(bed.astype(np.float32), wet=0.3, dur=3.0, decay=1.4, seed=r.integers(100), lo=25, hi=4000)[: len(t)]
    return fade(norm(bed, 1.0), 1.5, 2.5)


def logo_sting():
    return _mix_len(watch_tick(3, 0.5), np.pad(boom(3.0) * 0.8, (int(1.5 * SR), 0)))


LIBRARY = {
    "knock3": lambda **k: knock(3),
    "knock3_far": lambda **k: knock(3, far=True),
    "boom": lambda **k: boom(),
    "stinger": lambda **k: stinger(),
    "riser": lambda **k: riser(k.get("dur", 2.5)),
    "reverse_swell": lambda **k: reverse_swell(k.get("dur", 1.6)),
    "tape_click": lambda **k: tape_click(),
    "tape_hiss": lambda **k: tape_hiss(k.get("dur", 5.0)),
    "heartbeat": lambda **k: heartbeat(),
    "watch_tick": lambda **k: watch_tick(),
    "whoosh": lambda **k: whoosh(),
    "glitch": lambda **k: glitch(),
    "logo_sting": lambda **k: logo_sting(),
}

# Default gain (linear) of each effect in the final mix.
GAINS = {"knock3": 0.75, "knock3_far": 0.6, "boom": 0.7, "stinger": 0.75, "riser": 0.45, "reverse_swell": 0.5,
         "tape_click": 0.5, "tape_hiss": 0.22, "heartbeat": 0.6, "watch_tick": 0.4, "whoosh": 0.18,
         "glitch": 0.3, "logo_sting": 0.6}
