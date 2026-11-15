"""Shared paths, settings, logging and environment helpers."""
from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
CHARACTER_DIR = ROOT / "character"
STORY_BANK = ROOT / "story_bank"
EPISODES = ROOT / "episodes"
MODELS = ROOT / "models"
ASSETS = ROOT / "assets"
FONTS = ASSETS / "fonts"
LOGS = ROOT / "logs"

# Keep every model download inside the project so it is easy to find / delete.
os.environ.setdefault("HF_HOME", str(MODELS / "hf_home"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def load_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    """Atomic JSON write (write temp file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def settings() -> dict:
    return load_json(CONFIG_DIR / "settings.json", {})


def character() -> dict:
    return load_json(CHARACTER_DIR / "elias_voss.json", {})


def get_logger(name: str = "archive", logfile: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    targets = [LOGS / "archive.log"] + ([Path(logfile)] if logfile else [])
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == t.resolve() for h in logger.handlers):
            fh = logging.FileHandler(t, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def which_ffmpeg() -> tuple[str, str]:
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ff or not fp:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH. Install FFmpeg (see README).")
    return ff, fp


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(map(str, cmd))[:300]}\n{res.stderr[-2000:]}")
    return res


def torch_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def environment_report() -> dict:
    rep = {"os": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0]}
    try:
        rep["ffmpeg"] = run([which_ffmpeg()[0], "-version"]).stdout.splitlines()[0]
    except Exception as e:  # noqa
        rep["ffmpeg"] = f"missing ({e})"
    rep["torch_device"] = torch_device()
    try:
        rep["free_disk_gb"] = round(shutil.disk_usage(ROOT).free / 1e9, 1)
    except Exception:
        pass
    return rep
