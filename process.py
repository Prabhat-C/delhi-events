#!/usr/bin/env python3
"""
process.py - turn WhatsApp Web screenshots in inbox/ into events.json.

Pipeline
  1. Hash every image in inbox/. Anything already in hash_log.json is moved to done/ untouched.
  2. Send new images in batches (default 15) to claude-haiku-5-5 with prompt.txt. The model
     groups screenshots into events and extracts the fields.
  3. Every event is validated in code. An event that fails validation, has a dead registration
     link, or is not confidence "high" is retried with ONLY its own images, climbing the ladder:
        1 haiku-5-5 effort low (the batch call itself)   2 haiku-5-5 medium
        3 haiku-5-5 high                                 4 sonnet-5-5 medium
        5 give up: keep the best attempt and flag it for manual review
  4. Expired events are dropped (images go to done/), recurring events never expire.
  5. Text is cleaned (emoji etc.), links are cleaned and tested, duplicates are merged.
  6. For each kept event two derived images are made from its poster screenshot (the one the model names in
     "poster_image"; the first screenshot if that is missing or invalid): a thumbnail
     (thumbs/<sha12>.jpg) and a lightweight display poster (posters/<sha12>.jpg, long edge <= 1600 px,
     JPEG q80). events.json "poster" points to the display poster and "source_files" still lists the
     originals in done/ (poster screenshot first), which are never modified. If the display poster cannot be made, "poster"
     falls back to the original in done/.
  7. events.json / needs_review.json / hash_log.json are written.

Usage (from the repo root; ANTHROPIC_API_KEY comes from the environment, or from a local .env):
    python process.py                       # normal run, same as the GitHub Action
    python process.py --dry-run --limit 3   # call the API, print results, change nothing on disk
    python process.py --skip-link-check     # do not make HTTP requests to registration links

Requires: pip install anthropic pillow
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

try:
    import anthropic
except ImportError:  # lets the pure helper functions be imported/tested without the SDK
    anthropic = None
from PIL import Image, ImageOps

log = logging.getLogger("process")

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
IST = timezone(timedelta(hours=5, minutes=30))  # India has no DST, so a fixed offset is exact
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
BATCH_SIZE = 15            # spec: about 15 to 20 images per request
MAX_EDGE = 1568            # images are downscaled to this long edge before sending
THUMB_WIDTH = 480
DISPLAY_MAX_EDGE = 1600    # display posters (posters/): long edge, never upscaled
DISPLAY_QUALITY = 80
SANE_PAST_DAYS = 366       # a date older than this is treated as a misread, not as "expired"
SANE_FUTURE_DAYS = 730

# Every key the model must return (order matches sample_events.json).
EVENT_KEYS = [
    "images", "grouping_uncertain", "title", "organiser", "description", "details",
    "date", "end_date", "start_time", "end_time", "venue", "registration_link",
    "confidence", "notes", "recurrence", "recurring",
]
STORED_KEYS = [k for k in EVENT_KEYS if k != "images"]  # "images" is batch-relative, so not stored
TEXT_FIELDS = ["title", "organiser", "description", "details", "venue", "notes", "recurrence"]
COMPLETENESS_FIELDS = [
    "title", "organiser", "description", "details", "date", "end_date",
    "start_time", "end_time", "venue", "registration_link", "recurrence",
]
CONF_RANK = {"low": 0, "medium": 1, "high": 2}


class Rung(NamedTuple):
    name: str
    model: str
    effort: str


# Haiku 5.5 must NOT receive temperature / top_p / top_k (400 error), so none are ever sent.
LADDER = [
    Rung("haiku-low", "claude-haiku-5-5", "low"),
    Rung("haiku-medium", "claude-haiku-5-5", "medium"),
    Rung("haiku-high", "claude-haiku-5-5", "high"),
    Rung("sonnet-medium", "claude-sonnet-5-5", "medium"),
]
# Thinking tokens count toward max_tokens, so higher efforts get more headroom.
THINKING_ALLOWANCE = {"low": 2000, "medium": 6000, "high": 12000}
MAX_TOKENS_CAP = 20000  # stays inside what the SDK allows for non-streaming requests

# USD per million tokens (input, output). Haiku figures are from the spec.
# Fill in Sonnet's numbers from the pricing page to get cost estimates for rung 4.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": None,
}

HTTP_UA = "Mozilla/5.0 (compatible; EventsListingLinkCheck/1.0)"
# Sites that block bots answer these for perfectly good links, so they count as "reachable".
SOFT_PASS_STATUSES = {401, 403, 429, 999}


# --------------------------------------------------------------------------------------
# Text cleaning (emoji / pictograph stripping, whitespace, stray separators)
# --------------------------------------------------------------------------------------
_ATOM = (
    "[\U0001F300-\U0001FAFF\U0001F1E6-\U0001F1FF"           # pictographs, emoticons, flags, skin tones
    "\u2600-\u27BF"                                          # misc symbols + dingbats (stars, ticks, phones)
    "\u2190-\u21FF\u27F0-\u27FF\u2900-\u297F\u2B00-\u2BFF"   # arrows, misc symbols and arrows (incl. U+2B50)
    "\u25A0-\u25FF\u2022\u2023"                              # geometric shapes, bullets
    "\u203C\u2049\u2139\u24C2\u231A\u231B\u23E9-\u23F3\u23F8-\u23FA]"
)
# An emoji, plus variation selectors / keycap marks, plus ZWJ ONLY when it joins two emoji.
# (A bare ZWJ/ZWNJ is left alone everywhere else: Hindi and Urdu need them.)
_EMOJI_RE = re.compile(_ATOM + "(?:[\uFE0E\uFE0F\u20E3]|\u200D(?=" + _ATOM + "))*|[\uFE0E\uFE0F\u20E3]+")
_LEAD_SEP = re.compile(r"^[|\-\u2013\u2014]+(?=\s|$)")
_TRAIL_SEP = re.compile(r"(?:^|\s)[|\-\u2013\u2014]+$")


def clean_text(s: Optional[str]) -> Optional[str]:
    """Strip emoji, collapse spaces, drop stray separators at line starts/ends. Keeps paragraph breaks."""
    if s is None:
        return None
    s = _EMOJI_RE.sub("", s.replace("\r\n", "\n").replace("\r", "\n"))
    lines = []
    for line in s.split("\n"):
        line = re.sub(r"[ \t\u00A0]+", " ", line).strip()
        prev = None
        while prev != line:
            prev = line
            line = _LEAD_SEP.sub("", line).strip()
            line = _TRAIL_SEP.sub("", line).strip()
        lines.append(line)
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return out.strip("\n")


def clean_event_text(ev: dict) -> None:
    for k in TEXT_FIELDS:
        v = ev.get(k)
        if v is not None and not isinstance(v, str):
            v = str(v)
        if isinstance(v, str):
            v = clean_text(v)
        ev[k] = v if v else ("" if k == "notes" else None)  # notes uses "" for "nothing"


# --------------------------------------------------------------------------------------
# Registration links: clean, then test
# --------------------------------------------------------------------------------------
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_BARE_DOMAIN_RE = re.compile(r"^[\w\-]+(\.[\w\-]+)+(/|$)")
_GFORM_ID_RE = re.compile(r"/forms/(?:u/\d+/)?d/e/([A-Za-z0-9_\-]+)")


def _is_tracking_param(name: str) -> bool:
    n = name.lower()
    return n.startswith("utm_") or n in {"fbclid", "sfnsn", "igshid"}


def clean_link(raw: str) -> Optional[str]:
    """Remove tracking parameters. Never repairs characters. Returns None if it is not an http(s) link."""
    url = raw.strip().strip("<>\"'").rstrip(".,;:!")
    if not _SCHEME_RE.match(url):
        if _BARE_DOMAIN_RE.match(url):
            url = "https://" + url            # only a missing scheme is added, nothing else
        else:
            return None
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None
    host = parts.netloc.lower()
    # Google Forms: /u/<n>/ (needs a particular logged-in account) and /formResponse (the submit
    # endpoint) variants are rewritten to the plain public form URL before anything is tested.
    if host == "docs.google.com":
        m = _GFORM_ID_RE.search(parts.path)
        if m and ("/u/" in parts.path or parts.path.endswith("/formResponse")):
            return f"https://docs.google.com/forms/d/e/{m.group(1)}/viewform"
    query, fragment = parts.query, parts.fragment
    if (host.endswith("docs.google.com") and parts.path.startswith("/forms")) or host == "forms.gle":
        query, fragment = "", ""              # Google Forms: the whole query string goes
    else:
        kept = [p for p in query.split("&") if p and not _is_tracking_param(p.split("=", 1)[0])]
        query = "&".join(kept)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, fragment))


def http_ok(url: str, timeout: int = 10) -> tuple[bool, str]:
    last = ""
    for _ in range(2):  # one retry for flaky networks
        try:
            req = Request(url, headers={"User-Agent": HTTP_UA, "Accept": "text/html,*/*"})
            with urlopen(req, timeout=timeout) as r:
                return (r.status < 400), f"HTTP {r.status}"
        except HTTPError as e:
            if e.code in SOFT_PASS_STATUSES:
                return True, f"HTTP {e.code} (treated as reachable)"
            return False, f"HTTP {e.code}"
        except (URLError, TimeoutError, OSError) as e:
            last = f"{type(e).__name__}: {e}"
    return False, last


class LinkResult(NamedTuple):
    ok: bool
    final: Optional[str]    # the link to publish (when ok)
    cleaned: Optional[str]  # the cleaned link as read from the poster/caption
    detail: str


class LinkChecker:
    def __init__(self, skip: bool = False, tester=http_ok):
        self.skip = skip
        self.tester = tester
        self.cache: dict[str, LinkResult] = {}

    def check(self, raw: str) -> LinkResult:
        cleaned = clean_link(raw)
        if cleaned is None:
            return LinkResult(False, None, None, "not an http(s) link")
        if self.skip:
            return LinkResult(True, cleaned, cleaned, "link check skipped")
        if cleaned in self.cache:
            return self.cache[cleaned]
        ok, detail = self.tester(cleaned)
        res = LinkResult(ok, cleaned if ok else None, cleaned, detail)
        self.cache[cleaned] = res
        return res


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
HARD_CODES = {"keys", "title", "date", "end_date"}  # these keep an event off the public page


def parse_iso(s) -> Optional[date]:
    if not isinstance(s, str) or not _DATE_RE.match(s):
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def validate(ev: dict, today: date) -> list[tuple[str, str]]:
    """Pure checks on a full-key event. Returns [(code, message)]; empty means valid."""
    probs: list[tuple[str, str]] = []
    if not isinstance(ev.get("title"), str) or not ev["title"].strip():
        probs.append(("title", "title is missing"))
    lo, hi = today - timedelta(days=SANE_PAST_DAYS), today + timedelta(days=SANE_FUTURE_DAYS)
    d = parse_iso(ev.get("date"))
    if d is None:
        probs.append(("date", f"date missing or not YYYY-MM-DD: {ev.get('date')!r}"))
    elif not lo <= d <= hi:
        probs.append(("date", f"date {d} is outside the sane window {lo} to {hi}"))
    if ev.get("end_date") is not None:
        ed = parse_iso(ev["end_date"])
        if ed is None:
            probs.append(("end_date", f"end_date not YYYY-MM-DD: {ev['end_date']!r}"))
        elif d is not None and ed < d:
            probs.append(("end_date", f"end_date {ed} is before date {d}"))
        elif ed > hi:
            probs.append(("end_date", f"end_date {ed} is outside the sane window"))
    for k in ("start_time", "end_time"):
        v = ev.get(k)
        if v is not None and not (isinstance(v, str) and _TIME_RE.match(v)):
            probs.append(("time", f"{k} is not HH:MM: {v!r}"))
    if ev.get("confidence") not in CONF_RANK:
        probs.append(("confidence", f"confidence invalid: {ev.get('confidence')!r}"))
    return probs


def is_expired(ev: dict, today: date) -> bool:
    if ev.get("recurring"):
        return False
    last = parse_iso(ev.get("end_date")) or parse_iso(ev.get("date"))
    return last is not None and last < today


def norm_title(t: str) -> str:
    s = "".join(ch if unicodedata.category(ch)[0] in "LNM" else " " for ch in t.casefold())
    return " ".join(s.split())  # keeps Hindi/Urdu letters and combining marks


def dedupe_key(entry: dict) -> Optional[tuple[str, str]]:
    if isinstance(entry.get("title"), str) and parse_iso(entry.get("date")):
        return norm_title(entry["title"]), entry["date"]
    return None


def completeness(entry: dict) -> int:
    return sum(1 for k in COMPLETENESS_FIELDS if entry.get(k))


# --------------------------------------------------------------------------------------
# Attempts
# --------------------------------------------------------------------------------------
@dataclass
class Attempt:
    event: dict
    rung: str
    problems: list = field(default_factory=list)
    expired: bool = False
    link_failed: Optional[str] = None     # cleaned link that did not pass the HTTP test
    poster_img: Optional["Img"] = None    # screenshot the model chose as the poster (None -> first image)

    @property
    def ok(self) -> bool:
        return not self.problems and not self.expired and self.event.get("confidence") == "high"

    def score(self):
        return (-len(self.problems), CONF_RANK.get(self.event.get("confidence"), 0), completeness(self.event))


@dataclass
class Outcome:
    kind: str                       # event | expired | not_event | failed
    attempt: Optional[Attempt] = None
    sticky_uncertain: bool = False


class ModelCallError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind            # fatal (stop the run) | rejected (a 400 about one image)


def parse_events(text: str) -> Optional[list]:
    cands = [text.strip()]
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    cands.append(fenced)
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        cands.append(text[i:j + 1])
    for c in cands:
        try:
            obj = json.loads(c)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("events"), list) and all(isinstance(e, dict) for e in obj["events"]):
            return obj["events"]
    return None


def reconcile_groups(events: list, n: int):
    """Make every image index 1..n appear in exactly one event. Returns (groups, orphans)."""
    seen: set[int] = set()
    groups = []
    for ev in events:
        idxs = []
        for v in ev.get("images") if isinstance(ev.get("images"), list) else []:
            if isinstance(v, int) and 1 <= v <= n and v not in seen:
                seen.add(v)
                idxs.append(v)
        if idxs:
            groups.append((ev, sorted(idxs)))
        else:
            log.warning("batch event %r had no usable image indices; dropped", ev.get("title"))
    orphans = [i for i in range(1, n + 1) if i not in seen]
    return groups, orphans


# --------------------------------------------------------------------------------------
# API errors: what stops the run, what is retried, what only affects one image
# --------------------------------------------------------------------------------------
RETRY_WAITS = (3, 10, 30)   # seconds to wait before retry 1, 2, 3 (429, 5xx/529, network errors)


def classify_api_error(e: Exception):
    """(kind, message) for an Anthropic SDK error, or None for any other exception.
    kind: "fatal" = stop the whole run | "rejected" = a 400 about the image itself | "retry"."""
    if anthropic is None or not isinstance(e, anthropic.APIError):
        return None
    status = getattr(e, "status_code", None)
    detail = str(e)
    low = detail.lower()
    if status == 401:
        return "fatal", ("The API rejected your key (401). Replace ANTHROPIC_API_KEY: create a new key in the "
                         f"Anthropic Console, then update the GitHub secret (or your local .env). Detail: {detail}")
    if status == 403:
        return "fatal", ("The API refused this key (403). Replace ANTHROPIC_API_KEY with a key that has access: "
                         f"create a new key in the Anthropic Console, then update the GitHub secret (or .env). Detail: {detail}")
    if status == 404:
        return "fatal", f"The API says a model or endpoint was not found (404). Check the model names in LADDER. Detail: {detail}"
    if status == 400:
        if "credit balance" in low:
            return "fatal", ("Your Anthropic credit balance is too low. Top up credits in the Anthropic Console "
                             f"(Plans & Billing), then run again. Detail: {detail}")
        if "image" in low:
            return "rejected", detail
        return "fatal", ("The API rejected the request (400) for a reason that is not about a single image, so the "
                         f"run was stopped. Detail: {detail}")
    if status == 413:
        return "rejected", detail            # request too large: handled per image
    if status is None or status == 429 or status >= 500:
        return "retry", detail               # no status = connection error or timeout
    return "fatal", f"Unexpected API error {status}: {detail}"


def retry_after_seconds(e: Exception) -> Optional[float]:
    try:
        return float(e.response.headers.get("retry-after"))
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Run context: state files, API calls, image handling
# --------------------------------------------------------------------------------------
@dataclass
class Img:
    path: Path
    sha: str

    @property
    def name(self) -> str:
        return self.path.name


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name.lower())]


def load_json(path: Path, default):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def est_cost(model: str, tin: int, tout: int) -> Optional[float]:
    p = PRICES.get(model)
    return None if not p else (tin * p[0] + tout * p[1]) / 1e6


def render_display_poster(src: Path, dst: Path) -> None:
    """Display copy of a screenshot: EXIF transpose, flatten transparency onto white, RGB, long edge at most
    DISPLAY_MAX_EDGE (never upscaled), JPEG quality DISPLAY_QUALITY, optimized. The original is not modified."""
    with Image.open(src) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        img = bg.convert("RGB")
    scale = DISPLAY_MAX_EDGE / max(img.size)
    if scale < 1:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")          # write, then replace: a failed save leaves no half file
    try:
        img.save(tmp, "JPEG", quality=DISPLAY_QUALITY, optimize=True)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


class Ctx:
    def __init__(self, base: Path, args, client, today: date):
        self.base, self.args, self.client, self.today = base, args, client, today
        self.inbox = base / args.inbox
        self.done, self.thumbs, self.posters = base / "done", base / "thumbs", base / "posters"
        self.events_path = base / "events.json"
        self.review_path = base / "needs_review.json"
        self.hash_path = base / "hash_log.json"
        prompt = (base / "prompt.txt").read_text(encoding="utf-8")
        if "{TODAY}" not in prompt:
            raise SystemExit("prompt.txt has no {TODAY} placeholder")
        self.system = prompt.replace("{TODAY}", today.isoformat())  # the only change made to the prompt
        self.links = LinkChecker(skip=args.skip_link_check)
        self.events: list = load_json(self.events_path, {"events": []})["events"]
        self.review: list = load_json(self.review_path, {"review": []})["review"]
        self.hashlog: dict = load_json(self.hash_path, {})
        self.usage: dict[str, dict] = {}
        self.counts: Counter = Counter()
        self._b64: dict[str, dict] = {}
        self.api_calls = 0                 # every create() attempt, including retries
        self.api_ok = 0                    # attempts that got a response
        self.pending_failed: list = []     # images with no usable output; decided at the end of the run
        self.dupe_imgs: list = []          # inbox files already in hash_log; moved at the end of the run

    # -- persistence ---------------------------------------------------------------
    def save(self) -> None:
        if self.args.dry_run:
            return
        write_json(self.events_path, {"events": self.events})
        write_json(self.review_path, {"review": self.review})
        write_json(self.hash_path, self.hashlog)

    def move_to_done(self, im: Img) -> str:
        """Move an inbox image to done/. Returns its repo-relative path."""
        if self.args.dry_run:
            return f"inbox/{im.name} (dry run)"
        self.done.mkdir(parents=True, exist_ok=True)
        dest = self.done / im.path.name
        if dest.exists():
            dest = self.done / f"{im.path.stem}_{im.sha[:8]}{im.path.suffix}"
        shutil.move(str(im.path), str(dest))
        return dest.relative_to(self.base).as_posix()

    def record(self, imgs, outcome: str, event_id=None, title=None) -> None:
        for im in imgs:
            self.hashlog[im.sha] = {
                "file": im.name, "outcome": outcome, "event_id": event_id,
                "title": title, "processed": self.today.isoformat(),
            }

    # -- images --------------------------------------------------------------------
    def image_block(self, im: Img) -> dict:
        if im.sha not in self._b64:
            with Image.open(im.path) as raw:
                img = ImageOps.exif_transpose(raw).convert("RGBA")
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                bg.alpha_composite(img)
                img = bg.convert("RGB")
            scale = MAX_EDGE / max(img.size)
            if scale < 1:
                img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=88, optimize=True)
            self._b64[im.sha] = {"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.standard_b64encode(buf.getvalue()).decode()}}
        return self._b64[im.sha]

    def make_thumb(self, src: Path, sha: str) -> Optional[str]:
        if self.args.dry_run:
            return None
        try:
            self.thumbs.mkdir(parents=True, exist_ok=True)
            dst = self.thumbs / f"{sha[:12]}.jpg"
            with Image.open(src) as raw:
                img = ImageOps.exif_transpose(raw).convert("RGB")
                img.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 2))
                img.save(dst, "JPEG", quality=78, optimize=True)
            return dst.relative_to(self.base).as_posix()
        except Exception as e:  # a missing thumbnail must not lose the event
            log.warning("thumbnail failed for %s: %s", src.name, e)
            return None

    def make_poster(self, src: Path, sha: str) -> Optional[str]:
        """Write posters/<sha12>.jpg from src. Returns the repo-relative path, or None (with a warning) on failure."""
        if self.args.dry_run:
            return None
        dst = self.posters / f"{sha[:12]}.jpg"
        try:
            render_display_poster(src, dst)
            return dst.relative_to(self.base).as_posix()
        except Exception as e:  # a missing display copy must not lose the event
            log.warning("display poster failed for %s: %s", src.name, e)
            return None

    def remove_orphan_poster(self, rel, keep) -> None:
        """Delete a superseded display copy. Only files directly inside posters/ are ever removed, never done/."""
        if self.args.dry_run or not isinstance(rel, str) or rel == keep or not rel.startswith("posters/"):
            return
        p = (self.base / rel).resolve()
        if p.parent == self.posters.resolve():
            try:
                p.unlink(missing_ok=True)
            except OSError as e:
                log.warning("could not remove %s: %s", rel, e)

    # -- the API call ----------------------------------------------------------------
    def call(self, rung: Rung, imgs: list, bump: float = 1.0) -> str:
        content = []
        for i, im in enumerate(imgs, 1):
            content.append({"type": "text", "text": f"Screenshot {i}:"})
            content.append(self.image_block(im))
        content.append({"type": "text", "text": "Return the JSON object."})
        max_tokens = min(MAX_TOKENS_CAP, int((2000 + 800 * len(imgs) + THINKING_ALLOWANCE[rung.effort]) * bump))
        resp = self._create(
            model=rung.model,
            max_tokens=max_tokens,
            system=self.system,
            messages=[{"role": "user", "content": content}],
            # sent via extra_body so it works whichever SDK version is installed
            extra_body={"output_config": {"effort": rung.effort}},
        )
        self._log_usage(rung, len(imgs), resp)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def _create(self, **kwargs):
        """One API request with retries for 429 / 5xx / 529 / network errors. Raises ModelCallError."""
        total = len(RETRY_WAITS) + 1
        for n in range(1, total + 1):
            self.api_calls += 1
            try:
                resp = self.client.messages.create(**kwargs)
            except Exception as e:
                cls = classify_api_error(e)
                if cls is None:
                    raise                       # not an API error: a bug, let it surface
                kind, detail = cls
                if kind == "retry":
                    if n < total:
                        wait = RETRY_WAITS[n - 1]
                        ra = retry_after_seconds(e)
                        if ra:
                            wait = min(max(wait, ra), 60)
                        log.warning("API error, retrying in %ss (retry %d of %d): %s", wait, n, len(RETRY_WAITS), detail[:200])
                        time.sleep(wait)
                        continue
                    raise ModelCallError("fatal", (
                        f"The API was rate-limited or unavailable and still failing after {len(RETRY_WAITS)} retries. "
                        f"Wait a few minutes (see status.anthropic.com), then run again. Last error: {detail}")) from e
                raise ModelCallError(kind, detail) from e
            self.api_ok += 1
            return resp

    def _log_usage(self, rung: Rung, n_images: int, resp) -> None:
        u = resp.usage
        tin, tout = getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0
        cost = est_cost(rung.model, tin, tout)
        extra = ""
        for k in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            if getattr(u, k, None):
                extra += f" {k}={getattr(u, k)}"
        log.info("usage rung=%s model=%s effort=%s images=%d input_tokens=%d output_tokens=%d stop=%s est_usd=%s%s",
                 rung.name, rung.model, rung.effort, n_images, tin, tout, resp.stop_reason,
                 "n/a" if cost is None else f"{cost:.6f}", extra)
        if resp.stop_reason == "max_tokens":
            log.warning("response hit max_tokens (thinking tokens count toward it)")
        t = self.usage.setdefault(rung.model, {"calls": 0, "in": 0, "out": 0})
        t["calls"] += 1
        t["in"] += tin
        t["out"] += tout  # output_tokens already includes thinking tokens

    def call_events(self, rung: Rung, imgs: list) -> Optional[list]:
        """One API call; if the JSON does not parse, retry once on the same rung."""
        for attempt in (1, 2):
            text = self.call(rung, imgs, bump=1.0 if attempt == 1 else 1.5)
            events = parse_events(text)
            if events is not None:
                return events
            log.warning("JSON did not parse (rung %s, attempt %d)", rung.name, attempt)
        return None


# --------------------------------------------------------------------------------------
# Evaluating one model answer
# --------------------------------------------------------------------------------------
def pick_poster(raw_value, numbers: dict) -> Optional["Img"]:
    """poster_image from the model. `numbers` maps the screenshot numbers used in THIS answer to this group's
    own images. Anything that is not an int (bool excluded) naming one of them gives None, meaning "first image".
    It is optional: it is never part of the required-keys check, so it can never hold an event or cause a retry."""
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        return None
    return numbers.get(raw_value)


def evaluate(ctx: Ctx, raw: dict, rung_name: str, sticky_uncertain: bool, numbers: dict,
             extra_problems=()) -> Attempt:
    missing = [k for k in EVENT_KEYS if k not in raw]
    poster_img = pick_poster(raw.get("poster_image"), numbers)
    ev = {k: raw.get(k) for k in STORED_KEYS}
    ev["grouping_uncertain"] = bool(ev["grouping_uncertain"]) or sticky_uncertain
    if isinstance(ev.get("confidence"), str):
        ev["confidence"] = ev["confidence"].strip().lower()
    clean_event_text(ev)
    ev["recurring"] = ev["recurrence"] is not None   # recurring is true exactly when recurrence is set

    problems = list(extra_problems)
    if missing:
        problems.append(("keys", f"missing keys: {missing}"))
    problems += validate(ev, ctx.today)

    date_ok = not any(c in ("date", "end_date") for c, _ in problems)
    if date_ok and is_expired(ev, ctx.today):
        return Attempt(ev, rung_name, problems, expired=True, poster_img=poster_img)   # no escalation, no link test

    link_failed = None
    link = ev.get("registration_link")
    if not isinstance(link, str) or not link.strip():
        ev["registration_link"] = None
    else:
        res = ctx.links.check(link)
        if res.ok:
            ev["registration_link"] = res.final
        else:
            link_failed = res.cleaned or link.strip()
            ev["registration_link"] = link_failed
            problems.append(("link", f"registration link failed the check: {link_failed} ({res.detail})"))
    return Attempt(ev, rung_name, problems, link_failed=link_failed, poster_img=poster_img)


def resolve_group(ctx: Ctx, imgs: list, initial: Optional[dict], start_rung: int = 0,
                  force_uncertain: bool = False, numbers: Optional[dict] = None) -> Outcome:
    """Run the escalation ladder for one group of images.
    `numbers` maps the batch screenshot numbers used by the batch answer (`initial`) to this group's images.
    A retry sends only this group's images, numbered 1..n, so its answers use `retry_numbers`."""
    retry_numbers = {i: im for i, im in enumerate(imgs, 1)}
    sticky = bool(force_uncertain or (initial or {}).get("grouping_uncertain"))
    attempts: list[Attempt] = []
    empties = 0
    first = 0
    if initial is not None:
        a = evaluate(ctx, initial, LADDER[0].name, sticky, numbers or retry_numbers)
        attempts.append(a)
        if a.expired or a.ok:
            return Outcome("expired" if a.expired else "event", a, sticky)
        first = 1
    for ri in range(max(first, start_rung), len(LADDER)):
        rung = LADDER[ri]
        try:
            events = ctx.call_events(rung, imgs)
        except ModelCallError as e:
            if e.kind == "fatal":
                raise
            log.warning("image rejected by the API on rung %s for %s: %s", rung.name, [i.name for i in imgs], e)
            break
        if events is None:
            continue                            # unparseable twice -> next rung
        if not events:
            empties += 1
            continue
        extra = []
        if len(events) > 1:
            extra.append(("grouping", f"retry on this group's images returned {len(events)} events"))
        a = evaluate(ctx, events[0], rung.name, sticky, retry_numbers, extra)
        attempts.append(a)
        if a.expired or a.ok:
            break
    if not attempts:
        return Outcome("not_event" if empties else "failed", None, sticky)
    expired = [a for a in attempts if a.expired]
    if expired:
        return Outcome("expired", expired[-1], sticky)
    best = max(enumerate(attempts), key=lambda p: (p[1].score(), p[0]))[1]   # ties -> later rung
    return Outcome("event", best, sticky)


# --------------------------------------------------------------------------------------
# Turning an outcome into stored entries
# --------------------------------------------------------------------------------------
def finalize_event(best: Optional[Attempt], sticky: bool):
    """Returns (event, status, reasons, link_problem)."""
    if best is None:
        ev = {k: None for k in STORED_KEYS}
        ev.update(grouping_uncertain=sticky, notes="", confidence="low", recurring=False)
        return ev, "held", ["extraction failed: no usable model output on any rung"], None
    ev = dict(best.event)
    reasons = [msg for code, msg in best.problems if code != "link"]
    link_problem = None
    if best.link_failed:
        link_problem = best.link_failed
        ev["registration_link"] = None                  # published without a Register button
        reasons.append(f"registration link failed the check (read as {best.link_failed}); paste the correct link from WhatsApp")
    for k in ("start_time", "end_time"):
        if ev.get(k) is not None and not (isinstance(ev[k], str) and _TIME_RE.match(ev[k])):
            ev[k] = None
    conf = ev.get("confidence")
    if conf != "high":
        reasons.append(f"confidence is {conf}")
    if ev.get("grouping_uncertain"):
        reasons.append("grouping uncertain: check whether these screenshots belong to one post")
    if not ev.get("venue"):
        reasons.append("venue missing")
    held = any(c in HARD_CODES for c, _ in best.problems) or conf not in ("high", "medium")
    return ev, ("held" if held else "published"), reasons, link_problem


def upsert(ctx: Ctx, entry: dict) -> tuple[str, str, Optional[dict]]:
    """Insert into events, merging duplicates (same normalised title + start date).
    Returns (action, id, replaced_entry); replaced_entry is the old entry when action == "replaced"."""
    key = dedupe_key(entry)
    if key:
        for i, ex in enumerate(ctx.events):
            if dedupe_key(ex) != key:
                continue
            merged_files = ex.get("source_files", []) + entry["source_files"]
            merged_hashes = ex.get("image_hashes", []) + entry["image_hashes"]
            new_files = entry["source_files"] + ex.get("source_files", [])     # used if the new entry wins:
            new_hashes = entry["image_hashes"] + ex.get("image_hashes", [])    # its poster screenshot stays first
            new_better = (completeness(entry), CONF_RANK.get(entry.get("confidence"), 0)) > \
                         (completeness(ex), CONF_RANK.get(ex.get("confidence"), 0))
            if new_better:
                entry["id"] = ex["id"]
                entry["source_files"], entry["image_hashes"] = new_files, new_hashes
                ctx.events[i] = entry
                ctx.review = [r for r in ctx.review if r["id"] != ex["id"]]
                return "replaced", entry["id"], ex
            ex["source_files"], ex["image_hashes"] = merged_files, merged_hashes
            return "duplicate", ex["id"], None
    ctx.events.append(entry)
    return "added", entry["id"], None


def commit(ctx: Ctx, imgs: list, out: Outcome) -> None:
    names = [i.name for i in imgs]
    if out.kind in ("expired", "not_event"):
        ev = out.attempt.event if out.attempt else {}
        for im in imgs:
            ctx.move_to_done(im)
            log.info("finished image=%s rung=%s result=%s%s", im.name,
                     out.attempt.rung if out.attempt else "all", out.kind,
                     f" (ended {ev.get('end_date') or ev.get('date')})" if out.kind == "expired" else "")
        ctx.record(imgs, out.kind, title=ev.get("title"))
        ctx.counts[out.kind] += len(imgs)
        ctx.save()
        return

    best = out.attempt
    ev, status, reasons, link_problem = finalize_event(best, out.sticky_uncertain)
    # The poster screenshot is the model's poster_image when valid, else the first image. It goes first in
    # source_files / image_hashes and is the source of the thumbnail, the display poster and "poster".
    chosen = best.poster_img if best and best.poster_img in imgs else imgs[0]
    ordered = [chosen] + [im for im in imgs if im is not chosen]
    rel_paths = [ctx.move_to_done(im) for im in ordered]
    poster = rel_paths[0]
    thumb = ctx.make_thumb(ctx.base / rel_paths[0], chosen.sha) if not ctx.args.dry_run else None
    hashes = [im.sha for im in ordered]
    entry = {
        "id": hashlib.sha1("".join(sorted(hashes)).encode()).hexdigest()[:12],   # sorted: independent of the choice
        "status": status,
        **{k: ev.get(k) for k in STORED_KEYS},
        "poster": poster, "thumb": thumb,
        "source_files": rel_paths, "image_hashes": hashes,
        "rung": best.rung if best else "none",
        "added": ctx.today.isoformat(),
    }
    action, final_id, replaced = upsert(ctx, entry)
    if action != "duplicate" and not ctx.args.dry_run:
        # The display poster is made only now, for the entry that was kept, so a losing duplicate leaves no
        # file behind. It comes from the chosen poster screenshot (the one the thumbnail uses).
        poster_rel = ctx.make_poster(ctx.base / rel_paths[0], chosen.sha)
        if poster_rel is None:
            log.warning("using the original screenshot as the poster for %r", entry.get("title"))
        else:
            entry["poster"] = poster_rel
        if replaced:   # the old entry's display copy is superseded; a legacy done/ path is never deleted
            ctx.remove_orphan_poster(replaced.get("poster"), keep=entry["poster"])

    for im in imgs:
        log.info("finished image=%s rung=%s result=%s review=%s", im.name, entry["rung"],
                 status if action != "duplicate" else "duplicate", "yes" if reasons else "no")
    ctx.record(imgs, "duplicate_event" if action == "duplicate" else status, final_id, entry.get("title"))

    if action == "duplicate":
        ctx.counts["duplicate"] += len(imgs)
        log.info("duplicate of existing event %s (%r); kept the more complete entry", final_id, entry.get("title"))
    else:
        ctx.counts[status] += 1
        if reasons:
            item = {"id": final_id, "status": status, "title": entry.get("title"), "date": entry.get("date"),
                    "confidence": entry.get("confidence"), "reasons": reasons,
                    "poster": entry["poster"], "thumb": entry["thumb"], "source_files": entry["source_files"]}
            if link_problem:
                item["link_problem"] = link_problem
            ctx.review = [r for r in ctx.review if r["id"] != final_id] + [item]
            ctx.counts["review"] += 1
    if ctx.args.dry_run:
        print(json.dumps({"action": action, "status": status, "reasons": reasons, "event": entry},
                         ensure_ascii=False, indent=2))
    ctx.save()


# --------------------------------------------------------------------------------------
# Batches and the main run
# --------------------------------------------------------------------------------------
def handle_group(ctx: Ctx, imgs: list, initial, start_rung=0, force_uncertain=False, numbers=None) -> None:
    out = resolve_group(ctx, imgs, initial, start_rung, force_uncertain, numbers)
    if out.kind == "failed":
        # No usable output. Not committed yet: if the whole run turns out to have failed, nothing is changed.
        ctx.pending_failed.append((imgs, out))
    else:
        commit(ctx, imgs, out)


def process_batch(ctx: Ctx, imgs: list) -> None:
    n = len(imgs)
    log.info("batch of %d image(s): %s", n, [i.name for i in imgs])
    rejected = False
    try:
        events = ctx.call_events(LADDER[0], imgs)
    except ModelCallError as e:
        if e.kind == "fatal":
            raise
        log.warning("batch call rejected by the API (%s)", e)
        events, rejected = None, True
    if events is None:
        if n == 1:
            if rejected:
                ctx.pending_failed.append((imgs, Outcome("failed")))   # this image itself is the problem
            else:
                handle_group(ctx, imgs, None, start_rung=1)       # rung 1 was already spent on this image
        else:
            log.warning("could not group the batch; processing images one by one (marked grouping_uncertain)")
            for im in imgs:
                handle_group(ctx, [im], None, start_rung=0, force_uncertain=True)
        return
    groups, orphans = reconcile_groups(events, n)
    for ev, idxs in groups:
        handle_group(ctx, [imgs[i - 1] for i in idxs], ev, numbers={i: imgs[i - 1] for i in idxs})
    for i in orphans:
        log.info("image %s was not placed in any event; processing it on its own", imgs[i - 1].name)
        handle_group(ctx, [imgs[i - 1]], None, start_rung=0)


def run(ctx: Ctx) -> None:
    files = sorted((p for p in ctx.inbox.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=natural_key)
    todo: list[Img] = []
    seen: set[str] = set()
    for p in files:
        sha = sha256_file(p)
        if sha in ctx.hashlog or sha in seen:
            log.info("already processed (same file hash), will move to done/ at the end: %s", p.name)
            ctx.dupe_imgs.append(Img(p, sha))
            continue
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as e:
            log.warning("cannot read %s (%s); left in inbox", p.name, e)
            continue
        seen.add(sha)
        todo.append(Img(p, sha))
    if ctx.args.limit:
        todo = todo[:ctx.args.limit]
    log.info("%d new image(s) to process (today = %s IST)", len(todo), ctx.today)
    for i in range(0, len(todo), ctx.args.batch_size):
        process_batch(ctx, todo[i:i + ctx.args.batch_size])
    if ctx.api_calls and not ctx.api_ok:
        raise ModelCallError("fatal", "Every API call in this run failed, so the run was treated as failed and "
                                      "nothing was changed. See the errors above.")
    for imgs, out in ctx.pending_failed:
        commit(ctx, imgs, out)
    for im in ctx.dupe_imgs:
        ctx.move_to_done(im)
    ctx.save()


def summarize(ctx: Ctx) -> None:
    log.info("---- summary ----")
    log.info("results: %s", dict(ctx.counts) or "nothing processed")
    total = 0.0
    unknown = False
    for model, t in ctx.usage.items():
        cost = est_cost(model, t["in"], t["out"])
        if cost is None:
            unknown = True
        else:
            total += cost
        log.info("tokens %s: calls=%d input=%d output=%d est_usd=%s", model, t["calls"], t["in"], t["out"],
                 "n/a (no price set)" if cost is None else f"{cost:.6f}")
    log.info("estimated total cost: $%.6f%s", total, " (excluding models with no price set)" if unknown else "")
    if ctx.counts["review"]:
        log.info("%d item(s) added to needs_review.json", ctx.counts["review"])


def load_dotenv(path: Path) -> None:
    """Local convenience only: read KEY=VALUE lines from .env. Real environment variables always win,
    so in GitHub Actions (where there is no .env) the secret from the workflow is what gets used."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("\"'")
        if k and not os.environ.get(k):
            os.environ[k] = v


def make_client():
    if anthropic is None:
        raise SystemExit("The anthropic package is not installed: pip install anthropic pillow")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(max_retries=0, timeout=180.0)  # retries are handled in Ctx._create


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Extract events from WhatsApp screenshots in inbox/.")
    ap.add_argument("--base-dir", default=str(Path(__file__).resolve().parent), help="repo root (default: next to this script)")
    ap.add_argument("--inbox", default="inbox", help="inbox folder, relative to the base dir")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--limit", type=int, default=0, help="only process the first N new images")
    ap.add_argument("--today", help="override today's date (YYYY-MM-DD), useful for testing expiry")
    ap.add_argument("--dry-run", action="store_true", help="call the API and print results; move/write nothing")
    ap.add_argument("--skip-link-check", action="store_true", help="clean links but make no HTTP requests")
    return ap.parse_args(argv)


def main(argv=None, client=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    today = date.fromisoformat(args.today) if args.today else datetime.now(IST).date()
    load_dotenv(Path(args.base_dir) / ".env")
    ctx = Ctx(Path(args.base_dir), args, client or make_client(), today)
    try:
        run(ctx)
    except ModelCallError as e:
        log.error("")
        log.error("RUN STOPPED: %s", e)
        kept = sum(ctx.counts[k] for k in ("published", "held", "expired", "not_event", "duplicate"))
        log.error("Images not yet finished were left in inbox/. Nothing was marked as processed or held for them.%s",
                  f" {kept} result(s) finished before the error were kept." if kept else "")
        if os.environ.get("GITHUB_ACTIONS"):
            print("::error title=process.py stopped::" + str(e).replace("\n", " "))
        summarize(ctx)
        return 1
    summarize(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
