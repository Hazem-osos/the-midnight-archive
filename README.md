# THE MIDNIGHT ARCHIVE

Zero-cost, fully local pipeline for original horror YouTube Shorts narrated by the
fictional host **Elias Voss**.

> Status: **work in progress.** Moved to the Windows PC (RTX 2060 6GB) to continue.
> The Mac used to start it (M3, 8GB RAM) could not run local image generation safely.

## Done so far
| Module | What it does |
|---|---|
| `midnight/core.py` | paths, settings, logging, ffmpeg + device detection (cuda / mps / cpu) |
| `midnight/textutil.py` | `{display\|spoken}` markup, sentence / pause splitting |
| `midnight/tts.py` | Kokoro-82M narration (Apache-2.0), fixed Elias voice blend, per-sentence timing, voice FX — **tested** |
| `midnight/sfx.py` | procedural sound design (knocks, drones, stingers, risers, tape hiss...) — original, royalty-free |
| `midnight/mix.py` | narration + ambience + SFX cues, ducking, loudness normalize (-14 LUFS) |
| `midnight/imagegen.py` | Dreamshaper-8 (SD1.5, OpenRAIL-M) + LCM-LoRA, fp16, low VRAM |
| `midnight/depth.py` | Depth Anything V2 Small (Apache-2.0) for 2.5D parallax |
| `midnight/visuals.py` | scene images (AI or procedural fallback), locked Elias reference image |
| `midnight/render.py` | 1080x1920 30fps renderer: parallax camera moves, grade, grain, flicker, glitch, karaoke subtitles, branding, end card |
| `midnight/branding.py` | pocket-watch-at-3:17 logo, avatar (drawn in code) — **tested** |
| `midnight/story.py` | SQLite story DB, quality / originality / policy validation, story bank + Claude Code CLI generator |
| `midnight/validate.py` | ffprobe checks: resolution, codec, fps, AAC, duration, A/V sync, subtitles, full decode |
| `midnight/metadata.py` | title, description, hashtags, thumbnails |
| `midnight/youtube.py` | official YouTube Data API OAuth upload — **private only**; public needs explicit approval flag |
| `character/elias_voss.json` | permanent character identity, visual prompt, voice config |
| `story_bank/001_*.json` | Episode 001 script "The Room That Doesn't Exist" |

## Still to do (on Windows)
- `midnight/pipeline.py` (step orchestration, resume, no-overwrite) and `archive.py` CLI (`python archive.py new`)
- local dashboard, daily scheduler (Task Scheduler)
- Windows setup script + model download command
- generate the Elias reference image, render and validate Episode 001
- more stories in `story_bank/`

## Windows setup (manual for now)
```powershell
# Python 3.11-3.13 and FFmpeg (winget install Gyan.FFmpeg) required
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
Models (not in git, ~2.6 GB): Kokoro `kokoro-v1.0.fp16.onnx` + `voices-v1.0.bin`
(github.com/thewh1teagle/kokoro-onnx releases, tag `model-files-v1.0`) into `models/kokoro/`;
`Lykon/dreamshaper-8` (fp16 files), `latent-consistency/lcm-lora-sdv1-5`,
`depth-anything/Depth-Anything-V2-Small-hf` from Hugging Face into `models/`.

## Licenses of what is used
Kokoro-82M Apache-2.0 · Dreamshaper-8 CreativeML OpenRAIL-M · LCM-LoRA OpenRAIL++ ·
Depth Anything V2 Small Apache-2.0 · fonts: Anton & Cinzel (OFL), Special Elite (Apache-2.0) ·
all sound effects synthesized in code.
