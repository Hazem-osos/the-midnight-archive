"""Script text helpers: {display|spoken} markup, sentence/pause splitting, word lists."""
from __future__ import annotations

import re

MARKUP = re.compile(r"\{([^|{}]+)\|([^{}]+)\}")


def display_text(text: str) -> str:
    return MARKUP.sub(lambda m: m.group(1), text)


def spoken_text(text: str) -> str:
    return MARKUP.sub(lambda m: m.group(2), text)


def split_chunks(text: str) -> list[dict]:
    """Split a segment into speakable chunks.

    Each chunk is one sentence (or a clause ended by '...'). Returns dicts with
    display/spoken text and the pause type that follows ('ellipsis' or 'sentence').
    """
    # protect markup from splitting
    protected, store = text, []

    def _keep(m):
        store.append(m.group(0))
        return f"\u0000{len(store)-1}\u0000"

    protected = MARKUP.sub(_keep, protected)
    protected = protected.replace("…", "...")
    parts = re.split(r"(\.\.\.|(?<=[.!?])\s+)", protected)
    chunks, buf = [], ""
    for p in parts:
        if p is None:
            continue
        if p == "...":
            buf += "..."
            if buf.strip():
                chunks.append((buf.strip(), "ellipsis"))
            buf = ""
        elif p.strip() == "":
            if buf.strip():
                chunks.append((buf.strip(), "sentence"))
            buf = ""
        else:
            buf += p
    if buf.strip():
        chunks.append((buf.strip(), "sentence"))

    def _restore(s):
        return re.sub("\u0000(\\d+)\u0000", lambda m: store[int(m.group(1))], s)

    out = []
    for c, pause in chunks:
        c = _restore(c)
        disp = display_text(c)
        spk = spoken_text(c)
        if pause == "ellipsis":
            spk = spk.rstrip(".").rstrip() + "..."
        if re.search(r"[A-Za-z0-9]", disp):
            out.append({"display": disp, "spoken": spk, "pause": pause})
    return out


def words(text: str) -> list[str]:
    return [w for w in display_text(text).split() if w.strip()]


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", display_text(text)))
