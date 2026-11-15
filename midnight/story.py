"""Story database, quality/originality validation, and story sourcing.

Sources, in order:
  1. story_bank/*.json  - hand-written / pre-generated scripts not yet produced
  2. Claude Code CLI     - `claude -p` (uses your existing Claude Code login, no API key)
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

from .core import ROOT, STORY_BANK, load_json, save_json, settings
from .textutil import display_text, split_chunks, word_count

DB_PATH = ROOT / "data" / "stories.db"

CLICHES = ["little did", "a chill ran down", "blood ran cold", "in the dead of night", "suddenly", "eerie silence",
           "spine-chilling", "sent shivers", "unbeknownst", "it was a dark and stormy", "all of a sudden",
           "the end?", "to be continued", "nothing could have prepared"]
TRUE_CLAIMS = ["true story", "this really happened", "based on real events", "real footage", "100% real"]
POLICY_WORDS = ["gore", "dismember", "decapitat", "suicide", "self-harm", "rape", "torture", "child abuse", "mutilat"]
VALID_MOVES = {"push_in", "push_in_slow", "dolly_out", "pan_left", "pan_right", "tilt_up", "slow_orbit", "narrator"}


# ----------------------------------------------------------------- database
def db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS stories(
        id TEXT PRIMARY KEY, title TEXT, genre TEXT, logline TEXT, full_text TEXT, themes TEXT,
        source TEXT, episode TEXT, status TEXT, created TEXT)""")
    return con


def full_text(script) -> str:
    return " ".join(display_text(s["text"]) for s in script["segments"])


def register(script, source, episode=None, status="used"):
    with db() as con:
        con.execute("INSERT OR REPLACE INTO stories VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (script["id"], script["title"], script.get("genre", ""), script.get("logline", ""), full_text(script),
                     json.dumps(script.get("themes", [])), source, episode, status, datetime.now().isoformat(timespec="seconds")))


def used_stories():
    with db() as con:
        return [dict(zip(["id", "title", "logline", "full_text", "themes"], r))
                for r in con.execute("SELECT id,title,logline,full_text,themes FROM stories WHERE status='used'")]


def _shingles(text, k=3):
    w = re.findall(r"[a-z']+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}


def similarity(a: str, b: str) -> float:
    A, B = _shingles(a), _shingles(b)
    return len(A & B) / max(1, len(A | B))


def theme_overlap(a, b) -> float:
    A, B = {x.lower() for x in a}, {x.lower() for x in b}
    return len(A & B) / max(1, min(len(A), len(B)))


# ----------------------------------------------------------------- validation
def validate(script, check_originality=True):
    cfg = settings()["story"]
    issues, warnings = [], []
    for key in ("id", "title", "segments", "scenes"):
        if key not in script:
            issues.append(f"missing field '{key}'")
    if issues:
        return False, issues, warnings
    segs = script["segments"]
    text = full_text(script)
    low = text.lower()
    n = sum(word_count(s["text"]) for s in segs)
    est = n / cfg["words_per_second"]
    if not cfg["min_words"] <= n <= cfg["max_words"]:
        issues.append(f"word count {n} outside {cfg['min_words']}-{cfg['max_words']} (~{est:.0f}s)")
    if segs[0].get("type") != "hook":
        issues.append("first segment must be the hook")
    elif word_count(segs[0]["text"]) > 26:
        issues.append("hook is too long (>26 words) - must land in ~8 seconds")
    if not any(s.get("type") == "twist" for s in segs):
        issues.append("no twist segment")
    if not any(s.get("type") == "outro" for s in segs):
        issues.append("no outro segment")
    for s in segs:
        scene = s.get("scene")
        if scene not in script["scenes"]:
            issues.append(f"segment references unknown scene '{scene}'")
        for c in split_chunks(s["text"]):
            if word_count(c["display"]) > 30:
                warnings.append(f"long sentence ({word_count(c['display'])} words): {c['display'][:50]}...")
    for k, sc in script["scenes"].items():
        if sc.get("move") and sc["move"] not in VALID_MOVES:
            issues.append(f"scene {k}: unknown move {sc['move']}")
        if not sc.get("character") and not sc.get("prompt"):
            issues.append(f"scene {k}: needs a prompt")
    for c in CLICHES:
        if c in low:
            warnings.append(f"cliche: '{c}'")
    for c in TRUE_CLAIMS:
        if c in low:
            issues.append(f"claims to be real: '{c}' (stories must be presented as fiction)")
    for c in POLICY_WORDS:
        if c in low:
            issues.append(f"monetization risk word: '{c}'")
    if len(warnings) > 3:
        issues.append("too many cliches / weak sentences")
    if check_originality:
        for old in used_stories():
            if old["id"] == script["id"]:
                continue
            sim = similarity(text, old["full_text"])
            th = theme_overlap(script.get("themes", []), json.loads(old["themes"] or "[]"))
            if sim > cfg["similarity_threshold"]:
                issues.append(f"too similar to '{old['title']}' (text similarity {sim:.2f})")
            elif th >= 0.75:
                issues.append(f"themes repeat '{old['title']}' ({th:.0%} overlap)")
            if old["title"].lower() == script["title"].lower():
                issues.append(f"title already used: {old['title']}")
    return not issues, issues, warnings


# ----------------------------------------------------------------- sourcing
def bank_candidates():
    used = {s["id"] for s in used_stories()}
    out = []
    for p in sorted(STORY_BANK.glob("*.json")):
        s = load_json(p)
        if s and s.get("id") not in used:
            out.append((p, s))
    return out


PROMPT = """You write scripts for THE MIDNIGHT ARCHIVE, a YouTube Shorts channel of ORIGINAL FICTIONAL horror stories
narrated by the host Elias Voss (calm, deep, controlled). Write ONE new story as strict JSON, nothing else.

Rules:
- Natural, concrete American English. Short punchy sentences. No cliches ("suddenly", "little did he know", "a chill ran down").
- 165-200 spoken words total (about 75 seconds). Extremely strong first line (the hook, max 22 words).
- Escalation, then a twist that recontextualizes the story, then a short outro by Elias ending with "File closed."
- It is fiction: never claim it is real. Monetization-safe: no gore, no self-harm, no sexual content.
- Genre for this one: {genre}. Do NOT reuse these previous premises: {previous}
- Use "..." for dramatic pauses. Write numbers for speech with markup {{DISPLAY|spoken}}, e.g. {{3:17|three seventeen}}.
- 7-9 distinct visual scenes. Scene prompts describe a single cinematic still image (no text, no gore). Keep the
  protagonist mostly in silhouette / from behind / hands, so visual consistency is not required.

JSON schema:
{{"id": "kebab-case-id", "title": "Title Case, max 45 chars", "genre": "...", "logline": "one sentence",
 "themes": ["4-6 keywords"], "hook_text": "3-6 WORD ON-SCREEN HOOK",
 "segments": [
   {{"type": "hook", "scene": "<scene key>", "text": "...", "sfx": [{{"name": "boom", "at": "end"}}]}},
   {{"type": "narrator", "scene": "narrator", "text": "Welcome to the Midnight Archive. I'm Elias Voss. This is file {{NNN|...}}."}},
   {{"type": "story", "scene": "<scene key>", "text": "...", "stamp": "optional short time/place stamp"}},
   ... more story segments (1-3 sentences each) ...
   {{"type": "twist", "scene": "<scene key>", "text": "...", "sfx": [{{"name": "reverse_swell", "at": "end-1.6"}}, {{"name": "stinger", "at": "end"}}]}},
   {{"type": "outro", "scene": "narrator", "text": "... File closed."}}
 ],
 "scenes": {{"narrator": {{"character": "elias_voss", "move": "narrator"}},
            "<scene key>": {{"prompt": "...", "move": "push_in|push_in_slow|dolly_out|pan_left|pan_right|tilt_up|slow_orbit", "seed": 12345}}}},
 "thumbnail_scene": "<scene key>",
 "metadata": {{"title": "<Title> | Midnight Archive File NNN", "hashtags": ["#shorts", "#horrorstory", "..."], "tags": ["..."]}}}}
Available sfx names: knock3, knock3_far, boom, stinger, riser, reverse_swell, tape_click, tape_hiss, heartbeat, watch_tick, glitch.
Use NNN = {ep:03d}."""

GENRES = ["psychological horror", "supernatural mystery", "disturbing discovery", "strange disappearance",
          "unexplained event", "unexpected-ending thriller"]


def find_claude() -> str | None:
    cli = settings()["story"].get("claude_cli", "claude")
    return shutil.which(cli) or (cli if Path(cli).exists() else None)


def generate_with_claude(ep_number, log, attempts=3):
    cli = find_claude()
    if not cli:
        return None
    prev = "; ".join(f"{s['title']}: {s['logline']}" for s in used_stories())[:3000] or "none"
    genre = GENRES[ep_number % len(GENRES)]
    feedback = ""
    for a in range(attempts):
        prompt = PROMPT.format(genre=genre, previous=prev, ep=ep_number) + feedback
        log.info(f"Asking Claude Code CLI for a new story (attempt {a+1}, genre: {genre})")
        try:
            res = subprocess.run([cli, "-p", prompt, "--output-format", "text"], capture_output=True, text=True,
                                 timeout=settings()["story"]["claude_timeout_s"], encoding="utf-8")
        except Exception as e:  # noqa
            log.warning(f"Claude CLI failed: {e}")
            return None
        out = res.stdout
        try:
            script = json.loads(out[out.index("{"): out.rindex("}") + 1])
        except Exception:
            feedback = "\n\nYour previous answer was not valid JSON. Output ONLY the JSON object."
            continue
        ok, issues, warns = validate(script)
        if ok:
            script["source"] = "claude-code-cli"
            path = STORY_BANK / f"gen_{datetime.now():%Y%m%d_%H%M%S}_{script['id']}.json"
            save_json(path, script)
            return path, script
        log.warning(f"Generated story rejected: {issues}")
        feedback = "\n\nYour previous draft was rejected for: " + "; ".join(issues + warns) + ". Fix these."
    return None


def next_story(ep_number, log):
    cands = bank_candidates()
    for p, s in cands:
        ok, issues, warns = validate(s)
        for w in warns:
            log.warning(f"[{p.name}] {w}")
        if ok:
            return p, s
        log.warning(f"Skipping {p.name}: {issues}")
    got = generate_with_claude(ep_number, log)
    if got:
        return got
    raise RuntimeError("No unused valid story in story_bank/ and Claude Code CLI is not available. "
                       "Add a script JSON to story_bank/ or install Claude Code (`npm i -g @anthropic-ai/claude-code`).")
