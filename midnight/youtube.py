"""Official YouTube Data API v3 upload via Google OAuth (installed-app flow).

- Your Google password is never seen or stored: you sign in on Google's own page in
  your browser, and only an OAuth token is saved to config/youtube_token.json.
- Uploads are ALWAYS private. Making a video public requires the separate
  `publish` command with an explicit confirmation flag.
"""
from __future__ import annotations

from datetime import datetime

from .core import ROOT, load_json, save_json, settings

SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.readonly"]


def _paths():
    y = settings()["youtube"]
    return ROOT / y["client_secrets"], ROOT / y["token"]


def status() -> dict:
    secrets, token = _paths()
    return {"client_secrets": secrets.exists(), "authorized": token.exists()}


def credentials(interactive=False):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    secrets, token = _paths()
    creds = Credentials.from_authorized_user_file(str(token), SCOPES) if token.exists() else None
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token.write_text(creds.to_json())
    if not creds or not creds.valid:
        if not interactive:
            raise RuntimeError("YouTube not authorized. Run: python archive.py youtube-auth")
        if not secrets.exists():
            raise RuntimeError(f"Missing {secrets}. See README section 'YouTube setup'.")
        flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
        creds = flow.run_local_server(port=0, prompt="consent", open_browser=True)
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(creds.to_json())
    return creds


def service(interactive=False):
    from googleapiclient.discovery import build
    return build("youtube", "v3", credentials=credentials(interactive), cache_discovery=False)


def upload(ep_dir, log) -> dict:
    from googleapiclient.http import MediaFileUpload

    state = load_json(ep_dir / "state.json")
    if state.get("youtube", {}).get("video_id"):
        log.info(f"Already uploaded: {state['youtube']['video_id']}")
        return state["youtube"]
    md = load_json(ep_dir / "metadata.json")
    yt = service()
    body = {
        "snippet": {"title": md["title"], "description": md["description"], "tags": md["tags"],
                    "categoryId": md["category_id"], "defaultLanguage": "en", "defaultAudioLanguage": "en"},
        "status": {"privacyStatus": "private", "selfDeclaredMadeForKids": False,
                   "containsSyntheticMedia": bool(md.get("contains_synthetic_media", True))},
    }
    media = MediaFileUpload(str(ep_dir / "final.mp4"), mimetype="video/mp4", chunksize=4 * 1024 * 1024, resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        st, resp = req.next_chunk()
        if st:
            log.info(f"  upload {int(st.progress() * 100)}%")
    vid = resp["id"]
    result = {"video_id": vid, "url": f"https://youtu.be/{vid}", "privacy": "private",
              "uploaded_at": datetime.now().isoformat(timespec="seconds")}
    try:
        yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(ep_dir / "thumbnail.png"), mimetype="image/png")).execute()
        result["thumbnail"] = "set"
    except Exception as e:  # requires a phone-verified channel
        result["thumbnail"] = f"not set: {str(e)[:150]}"
    state["youtube"] = result
    save_json(ep_dir / "state.json", state)
    log.info(f"Uploaded PRIVATE: {result['url']}")
    return result


def publish(ep_dir, log, confirm: bool) -> dict:
    """Make an uploaded video public. Only runs with explicit confirmation."""
    if not confirm:
        raise RuntimeError("Refusing to publish without explicit approval. Re-run with --i-approve-public.")
    state = load_json(ep_dir / "state.json")
    vid = state.get("youtube", {}).get("video_id")
    if not vid:
        raise RuntimeError("Episode is not uploaded yet.")
    yt = service()
    yt.videos().update(part="status", body={"id": vid, "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False}}).execute()
    state["youtube"]["privacy"] = "public"
    state["youtube"]["published_at"] = datetime.now().isoformat(timespec="seconds")
    save_json(ep_dir / "state.json", state)
    log.info(f"Published publicly: https://youtu.be/{vid}")
    return state["youtube"]
