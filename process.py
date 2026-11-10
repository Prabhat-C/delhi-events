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
  6. The model labels every screenshot "poster", "text_caption" or "other" (image_labels). The poster screenshot
     of an event is always one labelled "poster" (the model's poster_image when it is one; otherwise code picks).
     With no "poster" at all, the best remaining screenshot is used and the event gets the review reason
     "no poster". For each kept event two derived images are made from the poster screenshot: a thumbnail
     (thumbs/<sha12>.jpg) and a lightweight display poster (posters/<sha12>.jpg, long edge <= 1600 px,
     JPEG q80). events.json "poster" points to the display poster and "source_files" still lists the
     originals in done/ (poster screenshot first), which are never modified. If the display poster cannot be made, "poster"
     falls back to the original in done/. "poster_blur" is a tiny base64 JPEG (about 20 px wide) of the same image,
     used by the zoom view as a placeholder; anything that replaces "poster" must replace "poster_blur" too. "source_labels" holds one label per source file.
  7. Merge step (after all batches): events from this run are compared with each other and with the existing
     entries by date + normalised venue + start_time (title ignored). Matches are merged into one entry;
     same date and venue with a different start time goes to needs_review.json instead.
  8. Image overrides (no API call): an inbox file named poster-<key>, thumb-<key> or zoom-<key> replaces the card
     thumbnail and zoom poster (poster-), only the thumbnail (thumb-) or only the zoom poster (zoom-) of the event
     the key points to (event id, or words from its title). Applied after the merge step.
  9. events.json / needs_review.json / hash_log.json are written.
  Inbox images are processed in order of the time in their file name (the file's modified time when the name has
  none), so a poster and its caption screenshot usually land in the same batch.

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
BLUR_WIDTH = 20            # poster_blur placeholder: tiny JPEG of the poster, base64 (about 0.5 to 1 KB)
BLUR_QUALITY = 40
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
LABELS = ("poster", "text_caption", "other")   # what the model calls each screenshot (image_labels)
# Manual image overrides: a file in inbox/ named poster-<key>.jpg, thumb-<key>.jpg or zoom-<key>.jpg is not sent to
# the model. <key> is an event id (4+ hex characters) or words from its title.
OVERRIDE_RE = re.compile(r"^(poster|thumb|zoom)[-_ ]+(.+)$", re.I)


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
# Poster labels, venue matching and the cross-event merge
# (everything here works on plain dicts, so dedupe_existing.py can reuse it unchanged)
# --------------------------------------------------------------------------------------
_SCREENSHOT_NAME_RE = re.compile(r"^screen ?shot", re.I)
_VENUE_SYNONYMS = {"center": "centre", "sri": "shri", "shree": "shri"}
_VENUE_FILLER = {"delhi", "ncr", "the"}


def is_screenshot_name(name: str) -> bool:
    """'Screenshot ...' files come from the screenshot tool and may show a WhatsApp timestamp or chat edge;
    anything else (for example 'WhatsApp Image ...') is a downloaded original. Only ever used as a tie-breaker."""
    return bool(_SCREENSHOT_NAME_RE.match(name.strip()))


def guess_label(path: Path) -> Optional[str]:
    """Pixel-based guess, 'poster' or 'text_caption', for images that have no model label: entries made before
    labels existed, or a model answer without image_labels. A dark-mode WhatsApp text screenshot is mostly dark
    neutral grey (half or more of its pixels). Returns None when the file cannot be read."""
    try:
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw).convert("RGB")
        img.thumbnail((160, 160))
        data = img.tobytes()
    except Exception:
        return None
    total = len(data) // 3
    if not total:
        return None
    # Dark-mode WhatsApp grey is roughly (34, 37, 38). Count pixels that are dark and neutral, but not near-black
    # (black-background posters are common), so text on that grey is not split across colour buckets.
    dark = sum(1 for r, g, b in zip(data[0::3], data[1::3], data[2::3])
               if min(r, g, b) >= 20 and max(r, g, b) <= 80 and max(r, g, b) - min(r, g, b) <= 24)
    return "text_caption" if dark / total >= 0.5 else "poster"


def stored_labels(e: dict) -> list:
    """source_labels of a stored entry, one per source file; None where the label is unknown (older entries)."""
    files = e.get("source_files") or []
    s = e.get("source_labels")
    if not isinstance(s, list) or len(s) != len(files):
        return [None] * len(files)
    return [x if x in LABELS else None for x in s]


def entry_labels(base: Path, e: dict) -> list:
    """Like stored_labels, but unknown labels are filled with a pixel guess (never None)."""
    return [lab or guess_label(base / f) or "other" for f, lab in zip(e.get("source_files") or [], stored_labels(e))]


def poster_label(base: Path, e: dict) -> str:
    """Label of the entry's poster screenshot (the first source file)."""
    files = e.get("source_files") or []
    if not files:
        return "other"
    return stored_labels(e)[0] or guess_label(base / files[0]) or "other"


def entry_rank(base: Path, e: dict) -> tuple:
    """Which of two matching entries is kept: a real poster first, then a downloaded file over a screenshot
    (no timestamp or chat edge), then the more complete entry, then the higher confidence."""
    files = e.get("source_files") or [""]
    return (1 if poster_label(base, e) == "poster" else 0,
            0 if is_screenshot_name(Path(files[0]).name) else 1,
            completeness(e), CONF_RANK.get(e.get("confidence"), 0))


def norm_venue(v) -> str:
    """Venue key for matching: lowercase, bracketed text removed, punctuation stripped, filler words
    (new delhi, delhi, ncr, the, pin codes) dropped, a few spelling variants unified."""
    if not isinstance(v, str):
        return ""
    s = norm_title(re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", v))
    s = re.sub(r"\bnew delhi\b", " ", s)
    words = []
    for w in s.split():
        w = _VENUE_SYNONYMS.get(w, w)
        if w in _VENUE_FILLER or re.fullmatch(r"11\d{4}", w):
            continue
        words.append(w)
    return " ".join(words)


def match_kind(a: dict, b: dict) -> tuple[Optional[str], str]:
    """('merge', '') for the same event, ('borderline', why) for a pair a person should look at, or (None, '').
    Same date + same normalised venue + same start_time (or a start_time missing on either side) is a merge,
    provided the end dates also agree. Titles are ignored."""
    da, db = parse_iso(a.get("date")), parse_iso(b.get("date"))
    if da is None or da != db:
        return None, ""
    va, vb = norm_venue(a.get("venue")), norm_venue(b.get("venue"))
    if not va or not vb:
        return None, ""
    sa, sb = a.get("start_time"), b.get("start_time")
    if sa and sb and sa != sb:
        if va == vb:
            return "borderline", f"same date and venue but different start time ({sa} and {sb})"
        return None, ""
    if va == vb:
        ea, eb = parse_iso(a.get("end_date")) or da, parse_iso(b.get("end_date")) or da
        if ea != eb:
            return "borderline", f"same date, venue and start time but different end date ({ea} and {eb})"
        return "merge", ""
    ta, tb = set(va.split()), set(vb.split())
    small, large = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(small) >= 2 and small <= large:
        return "borderline", "same date and start time, and one venue is worded as a longer version of the other"
    return None, ""


def remove_derived(base: Path, rel, keep, dry_run: bool = False) -> None:
    """Delete a thumbnail or display poster that is no longer used. Only files directly inside thumbs/ or
    posters/ are ever removed, never anything in done/."""
    if dry_run or not isinstance(rel, str) or rel == keep:
        return
    p = (base / rel).resolve()
    if p.parent in {(base / "thumbs").resolve(), (base / "posters").resolve()}:
        try:
            p.unlink(missing_ok=True)
        except OSError as e:
            log.warning("could not remove %s: %s", rel, e)


@dataclass
class MergeStore:
    """The state the merge step works on. The lists and the dict are changed in place."""
    base: Path
    events: list
    review: list
    hashlog: dict
    dry_run: bool = False


def add_review(store: MergeStore, entry: dict, reason: str, **extra) -> None:
    item = next((r for r in store.review if r.get("id") == entry.get("id")), None)
    if item is None:
        item = {"id": entry.get("id"), "status": entry.get("status"), "title": entry.get("title"),
                "date": entry.get("date"), "confidence": entry.get("confidence"), "reasons": [],
                "poster": entry.get("poster"), "thumb": entry.get("thumb"),
                "source_files": entry.get("source_files")}
        store.review.append(item)
    if reason not in item["reasons"]:
        item["reasons"].append(reason)
    item.update(extra)


def _longer(a, b):
    if not a:
        return b or None
    if not b:
        return a
    return b if len(b) > len(a) else a


def merge_pair(store: MergeStore, w: dict, l: dict) -> None:
    """Fold entry l into entry w (in place). w keeps its id, title, poster and thumbnail. l's source images are
    added to w; a registration link is kept if either has one; the longer details and the better (longer)
    description are kept; empty fields are filled from l. l's thumbnail and display poster files are deleted."""
    base = store.base
    labels, l_labels = entry_labels(base, w), entry_labels(base, l)
    files, hashes = list(w.get("source_files") or []), list(w.get("image_hashes") or [])
    l_hashes = l.get("image_hashes") or []
    for k, f in enumerate(l.get("source_files") or []):
        if f in files:
            continue
        files.append(f)
        labels.append(l_labels[k])
        if k < len(l_hashes):
            hashes.append(l_hashes[k])
    w["source_files"], w["image_hashes"], w["source_labels"] = files, hashes, labels

    for k in ("title", "organiser", "start_time", "end_time", "end_date", "venue", "recurrence"):
        if not w.get(k) and l.get(k):
            w[k] = l[k]
    w["recurring"] = w.get("recurrence") is not None
    if not w.get("registration_link") and l.get("registration_link"):
        w["registration_link"] = l["registration_link"]
    w["details"] = _longer(w.get("details"), l.get("details"))
    w["description"] = _longer(w.get("description"), l.get("description"))
    wn, ln = (w.get("notes") or "").strip(), (l.get("notes") or "").strip()
    w["notes"] = wn if (not ln or ln in wn) else (f"{wn} {ln}" if wn else ln)
    w["confidence"] = max((w.get("confidence"), l.get("confidence")), key=lambda c: CONF_RANK.get(c, -1))
    if w.get("status") != "published" and l.get("status") == "published":
        w["status"] = "published"
    w["grouping_uncertain"] = False
    added = [a for a in (w.get("added"), l.get("added")) if a]
    if added:
        w["added"] = min(added)

    for h in l_hashes:                                     # the hash log must point at the surviving entry
        if h in store.hashlog:
            store.hashlog[h]["event_id"] = w["id"]
    remove_derived(base, l.get("thumb"), w.get("thumb"), store.dry_run)
    remove_derived(base, l.get("poster"), w.get("poster"), store.dry_run)

    # needs_review.json: l's item goes away, w's item is brought up to date
    l_item = next((r for r in store.review if r.get("id") == l.get("id")), None)
    store.review[:] = [r for r in store.review if r.get("id") != l.get("id")]
    if l_item and l_item.get("link_problem") and not w.get("registration_link"):
        add_review(store, w, f"registration link failed the check (read as {l_item['link_problem']}); "
                             "paste the correct link from WhatsApp", link_problem=l_item["link_problem"])
    poster_ok = bool(labels) and labels[0] == "poster"
    if not poster_ok:
        add_review(store, w, "no poster")
    item = next((r for r in store.review if r.get("id") == w.get("id")), None)
    if item:
        kept = []
        for reason in item.get("reasons", []):
            if reason.startswith("grouping uncertain"):
                continue
            if reason == "no poster" and poster_ok:
                continue
            if reason.startswith("registration link failed") and w.get("registration_link"):
                continue
            if reason.startswith("confidence is") and w.get("confidence") == "high":
                continue
            kept.append(reason)
        if w.get("registration_link"):
            item.pop("link_problem", None)
        if kept:
            item.update(reasons=kept, status=w.get("status"), title=w.get("title"), confidence=w.get("confidence"),
                        poster=w.get("poster"), thumb=w.get("thumb"), source_files=w["source_files"])
        else:
            store.review.remove(item)


def merge_events(store: MergeStore, only_ids=None) -> list[str]:
    """Merge duplicate entries in store.events, in place, and return one message per merge or review flag.
    Only pairs where at least one entry is in only_ids are looked at (None = every pair). Same date + venue
    with a different start time is never merged: the later entry gets a needs_review.json reason instead."""
    msgs: list[str] = []
    watch = None if only_ids is None else set(only_ids)
    base = store.base

    def looked_at(a: dict, b: dict) -> bool:
        return watch is None or a.get("id") in watch or b.get("id") in watch

    # Pass 1: merge until nothing more matches (a merge can bring two other entries together).
    restart = True
    while restart:
        restart = False
        ev = store.events
        for j in range(1, len(ev)):
            for i in range(j):
                a, b = ev[i], ev[j]
                if not looked_at(a, b) or match_kind(a, b)[0] != "merge":
                    continue
                if entry_rank(base, b) > entry_rank(base, a):
                    w, l, li = b, a, i
                else:
                    w, l, li = a, b, j
                msgs.append(f"merged '{l.get('title')}' into '{w.get('title')}' "
                            f"({w.get('date')}, {w.get('venue')}, {w.get('start_time') or 'no time'})")
                merge_pair(store, w, l)
                del ev[li]
                if watch is not None:
                    watch.add(w["id"])
                restart = True
                break
            if restart:
                break

    # Pass 2: on the entries that are left, flag pairs a person should look at (never merged automatically).
    ev = store.events
    for j in range(1, len(ev)):
        for i in range(j):
            a, b = ev[i], ev[j]
            if not looked_at(a, b):
                continue
            kind, why = match_kind(a, b)
            if kind == "borderline":
                add_review(store, b, f"possible duplicate of '{a.get('title')}' ({a.get('id')}): {why}")
                msgs.append(f"review: '{b.get('title')}' and '{a.get('title')}' - {why}")
    return msgs


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
    poster_img: Optional["Img"] = None    # screenshot the model chose as the poster (None -> chosen in code)
    labels: Optional[dict] = None         # image sha -> "poster" | "text_caption" | "other"; None = model gave none

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


# "Screenshot 2026-11-10 at 5.26.59 AM.png", "WhatsApp Image 2026-11-09 at 5.56.39 PM.jpeg" and similar
_NAME_TIME_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})\D+?(\d{1,2})[.:](\d{2})[.:](\d{2})\s*([AaPp][Mm])?")


def inbox_sort_key(p: Path):
    """Order inbox images by time, so a poster and its caption screenshot sit next to each other. The time in the
    file name is used first (a GitHub checkout resets every file's modified time), then the file's modified time."""
    ts = None
    m = _NAME_TIME_RE.search(p.name)
    if m:
        try:
            y, mo, d, h, mi, s = (int(x) for x in m.groups()[:6])
            ap = (m.group(7) or "").lower()
            if ap:
                h = h % 12 + (12 if ap == "pm" else 0)
            ts = datetime(y, mo, d, h, mi, s, tzinfo=IST).timestamp()
        except ValueError:
            ts = None
    if ts is None:
        try:
            ts = p.stat().st_mtime
        except OSError:
            ts = 0.0
    return (ts, natural_key(p))


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


def render_blur(src: Path) -> str:
    """Tiny placeholder of a poster for the zoom view: same proportions, BLUR_WIDTH px wide, low-quality JPEG,
    returned as a base64 string (no data: prefix; index.html adds it). Raises on an unreadable image."""
    with Image.open(src) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        img = bg.convert("RGB")
    w = min(BLUR_WIDTH, img.width)
    h = max(1, round(img.height * w / img.width))
    img = img.resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=BLUR_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


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
        self.new_ids: set = set()          # ids of entries added or changed in this run (the merge step looks at these)
        self.override_files: list = []     # inbox files named poster-/thumb-/zoom-<key>; applied after the merge step

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

    def make_blur(self, src: Path) -> Optional[str]:
        """poster_blur for the poster made from src, or None (with a warning) on failure. Never loses the event."""
        if self.args.dry_run:
            return None
        try:
            return render_blur(src)
        except Exception as e:
            log.warning("blur placeholder failed for %s: %s", src.name, e)
            return None

    def remove_derived(self, rel, keep) -> None:
        """Delete a superseded thumbnail or display copy. Only files directly inside thumbs/ or posters/ are
        ever removed, never done/."""
        remove_derived(self.base, rel, keep, self.args.dry_run)

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


def parse_labels(raw_value, numbers: dict) -> Optional[dict]:
    """image_labels from the model: {"5": "poster", "6": "text_caption"} keyed by screenshot number. Returns
    {image sha: label} for this group's images (an unlabelled or unknown label counts as "other"), or None when
    the model gave no usable labels at all. Like poster_image it is not a required key, so it never causes a retry."""
    if not isinstance(raw_value, dict):
        return None
    out: dict = {}
    for k, v in raw_value.items():
        try:
            n = int(str(k).strip())
        except ValueError:
            continue
        im = numbers.get(n)
        if im is None or not isinstance(v, str):
            continue
        v = v.strip().lower().replace(" ", "_").replace("-", "_")
        out[im.sha] = v if v in LABELS else "other"
    if not out:
        return None
    for im in numbers.values():
        out.setdefault(im.sha, "other")
    return out


def _pixel_area(im: "Img") -> int:
    try:
        with Image.open(im.path) as r:
            return r.size[0] * r.size[1]
    except Exception:
        return 0


def choose_poster(imgs: list, labels: dict, pick: Optional["Img"]) -> tuple["Img", bool]:
    """The screenshot that becomes the thumbnail and display poster. Returns (image, is_real_poster).
    Always an image labelled "poster" when there is one, whatever poster_image said. Among several posters the
    model's pick wins, but a downloaded file beats a 'Screenshot ...' file (no timestamp or chat edge).
    With no poster at all: the model's pick if it is not a text screenshot, else the largest "other" image,
    else the largest image, and is_real_poster is False."""
    posters = [im for im in imgs if labels.get(im.sha) == "poster"]
    if posters:
        clean = [im for im in posters if not is_screenshot_name(im.name)]
        if pick in posters and (pick in clean or not clean):
            return pick, True
        return (clean or posters)[0], True
    pool = [im for im in imgs if labels.get(im.sha) == "other"] or list(imgs)
    if pick in pool:
        return pick, False
    return max(pool, key=_pixel_area), False


def evaluate(ctx: Ctx, raw: dict, rung_name: str, sticky_uncertain: bool, numbers: dict,
             extra_problems=()) -> Attempt:
    missing = [k for k in EVENT_KEYS if k not in raw]
    poster_img = pick_poster(raw.get("poster_image"), numbers)
    labels = parse_labels(raw.get("image_labels"), numbers)
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
        return Attempt(ev, rung_name, problems, expired=True, poster_img=poster_img, labels=labels)   # no escalation, no link test

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
    return Attempt(ev, rung_name, problems, link_failed=link_failed, poster_img=poster_img, labels=labels)


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


def _real_poster(e: dict) -> int:
    """1 unless the entry's poster screenshot is known to be a text or other image (an unknown label is fine)."""
    return 1 if (stored_labels(e) or [None])[0] in ("poster", None) else 0


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
            merged_labels = stored_labels(ex) + entry["source_labels"]
            new_files = entry["source_files"] + ex.get("source_files", [])     # used if the new entry wins:
            new_hashes = entry["image_hashes"] + ex.get("image_hashes", [])    # its poster screenshot stays first
            new_labels = entry["source_labels"] + stored_labels(ex)
            new_better = (_real_poster(entry), completeness(entry), CONF_RANK.get(entry.get("confidence"), 0)) > \
                         (_real_poster(ex), completeness(ex), CONF_RANK.get(ex.get("confidence"), 0))
            if new_better:
                entry["id"] = ex["id"]
                entry["source_files"], entry["image_hashes"], entry["source_labels"] = new_files, new_hashes, new_labels
                ctx.events[i] = entry
                ctx.review = [r for r in ctx.review if r["id"] != ex["id"]]
                return "replaced", entry["id"], ex
            ex["source_files"], ex["image_hashes"], ex["source_labels"] = merged_files, merged_hashes, merged_labels
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
    labels = best.labels if best is not None else None
    if labels is None:
        # No labels from the model (or no usable answer at all): guess from the pixels, before the files move.
        if best is not None:
            log.warning("no usable image_labels for %s; guessing the labels from the pixels", names)
        labels = {im.sha: guess_label(im.path) or "other" for im in imgs}
    # The poster screenshot is always one labelled "poster" when the group has one (the model's poster_image if it
    # is one). It goes first in source_files / image_hashes and is the source of the thumbnail, the display
    # poster and "poster". With no poster at all the best remaining image is used and the event is flagged.
    chosen, has_poster = choose_poster(imgs, labels, best.poster_img if best else None)
    if best is not None and not has_poster:
        reasons.append("no poster")
        log.warning("no poster among %s; using %s", names, chosen.name)
    ordered = [chosen] + [im for im in imgs if im is not chosen]
    source_labels = [labels.get(im.sha, "other") for im in ordered]
    rel_paths = [ctx.move_to_done(im) for im in ordered]
    poster = rel_paths[0]
    thumb = ctx.make_thumb(ctx.base / rel_paths[0], chosen.sha) if not ctx.args.dry_run else None
    hashes = [im.sha for im in ordered]
    entry = {
        "id": hashlib.sha1("".join(sorted(hashes)).encode()).hexdigest()[:12],   # sorted: independent of the choice
        "status": status,
        **{k: ev.get(k) for k in STORED_KEYS},
        "poster": poster, "thumb": thumb, "poster_blur": None,
        "source_files": rel_paths, "image_hashes": hashes, "source_labels": source_labels,
        "rung": best.rung if best else "none",
        "added": ctx.today.isoformat(),
    }
    action, final_id, replaced = upsert(ctx, entry)
    ctx.new_ids.add(final_id)
    if action == "duplicate":
        ctx.remove_derived(entry.get("thumb"), keep=None)   # the thumbnail made for the losing entry is not used
    elif not ctx.args.dry_run:
        # The display poster is made only now, for the entry that was kept, so a losing duplicate leaves no
        # file behind. It comes from the chosen poster screenshot (the one the thumbnail uses).
        poster_rel = ctx.make_poster(ctx.base / rel_paths[0], chosen.sha)
        if poster_rel is None:
            log.warning("using the original screenshot as the poster for %r", entry.get("title"))
        else:
            entry["poster"] = poster_rel
        entry["poster_blur"] = ctx.make_blur(ctx.base / rel_paths[0])   # same image as the display poster
        if replaced:   # the old entry's thumbnail and display copy are superseded; a legacy done/ path is never deleted
            ctx.remove_derived(replaced.get("poster"), keep=entry["poster"])
            ctx.remove_derived(replaced.get("thumb"), keep=entry["thumb"])

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


def merge_new(ctx: Ctx) -> None:
    """Merge step: compare every entry added or changed in this run with all other entries (events.json as it
    stands, including this run's other events) by date + venue + start_time, ignoring titles."""
    store = MergeStore(ctx.base, ctx.events, ctx.review, ctx.hashlog, ctx.args.dry_run)
    msgs = merge_events(store, ctx.new_ids)
    for m in msgs:
        log.info("merge step: %s", m)
        ctx.counts["review" if m.startswith("review:") else "merged"] += 1
    if msgs:
        ctx.save()


def override_parts(p: Path):
    """("poster"|"thumb"|"zoom", key) when the file name is an override name, else None."""
    m = OVERRIDE_RE.match(p.stem)
    if not m:
        return None
    key = re.sub(r"\s*\(\d+\)\s*$", "", m.group(2)).strip()   # "poster-karwaan (1).jpg" from a repeated download
    return (m.group(1).lower(), key) if key else None


def find_override_target(events: list, key: str) -> tuple[Optional[dict], str]:
    """The one event a key points to, or (None, why not). A key is an event id (or its first 4+ characters) or
    words that all appear in the event title."""
    k = key.strip().lower()
    if re.fullmatch(r"[0-9a-f]{4,12}", k):
        hits = [e for e in events if str(e.get("id", "")).lower() == k] or \
               [e for e in events if str(e.get("id", "")).lower().startswith(k)]
        if len(hits) == 1:
            return hits[0], ""
        if len(hits) > 1:
            return None, f"{len(hits)} events have an id starting {k!r}; use more characters"
    words = norm_title(key).split()
    if not words:
        return None, f"no event matches {key!r}"
    hits = [e for e in events if all(w in norm_title(str(e.get("title") or "")) for w in words)]
    if len(hits) > 1:
        exact = [e for e in hits if norm_title(str(e.get("title") or "")) == " ".join(words)]
        hits = exact if len(exact) == 1 else hits
    if len(hits) == 1:
        return hits[0], ""
    if not hits:
        return None, f"no event matches {key!r}"
    return None, f"{len(hits)} events match {key!r} (" + "; ".join(f"{e['id']} {e.get('title')}" for e in hits[:6]) + \
        "); use the event id in the file name"


def _derived_in_use(ctx: Ctx, rel, owner_id) -> bool:
    """True when an event other than owner_id still points to this thumbs/ or posters/ file."""
    return any(rel in (e.get("poster"), e.get("thumb")) for e in ctx.events if e.get("id") != owner_id)


def _override_problem(ctx: Ctx, name: str, why: str) -> None:
    msg = f"image {name} in inbox/ was not applied: {why}"
    log.warning(msg)
    if os.environ.get("GITHUB_ACTIONS"):
        print("::warning title=Image override not applied::" + msg)
    ctx.counts["override_failed"] += 1
    if ctx.args.dry_run:
        return
    item = next((r for r in ctx.review if r.get("id") is None and r.get("title") == name), None)
    if item is None:
        ctx.review.append({"id": None, "status": None, "title": name, "date": None, "confidence": None,
                           "reasons": [why], "poster": None, "thumb": None, "source_files": []})
    else:
        item["reasons"] = [why]


def apply_overrides(ctx: Ctx) -> None:
    """Handle inbox files named poster-<key>, thumb-<key> or zoom-<key> (no API call):
    poster- sets the card thumbnail and the zoom poster from the image, thumb- only the thumbnail, zoom- only the
    zoom poster. A poster- file is applied before thumb-/zoom- files, so those can override part of it. The image
    goes to done/ and into source_files; the replaced thumbs/ and posters/ files are deleted when nothing else uses
    them. A file that matches no event, or more than one, stays in inbox/ and gets an entry in needs_review.json."""
    order = {"poster": 0, "thumb": 1, "zoom": 1}
    todo = sorted(ctx.override_files, key=lambda t: (order[t[1]], t[0].name.lower()))
    for path, kind, key in todo:
        if not path.exists():
            continue
        ev, why = find_override_target(ctx.events, key)
        if ev is None:
            _override_problem(ctx, path.name, why)
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            sha = sha256_file(path)
        except Exception as e:
            log.warning("cannot read %s (%s); left in inbox", path.name, e)
            continue
        want_thumb, want_poster = kind in ("poster", "thumb"), kind in ("poster", "zoom")
        if ctx.args.dry_run:
            log.info("override %s -> event %s %r (dry run, nothing written)", path.name, ev["id"], ev.get("title"))
            continue
        new_thumb = ctx.make_thumb(path, sha) if want_thumb else None
        new_poster = ctx.make_poster(path, sha) if want_poster else None
        new_blur = ctx.make_blur(path) if want_poster else None
        if (want_thumb and not new_thumb) or (want_poster and not new_poster):
            _override_problem(ctx, path.name, "the image could not be turned into a thumbnail or poster")
            continue
        # where the original lives: reuse an identical file already in done/, otherwise move this one there
        prior = (ctx.hashlog.get(sha) or {}).get("file")
        existing = ctx.done / prior if prior else None
        if existing is not None and existing.is_file() and sha256_file(existing) == sha:
            rel = existing.relative_to(ctx.base).as_posix()
            path.unlink()
        else:
            rel = ctx.move_to_done(Img(path, sha))
        old_poster, old_thumb = ev.get("poster"), ev.get("thumb")
        files = list(ev.get("source_files") or [])
        hashes = list(ev.get("image_hashes") or [])
        labels = entry_labels(ctx.base, ev)
        rest = [(f, hashes[k] if k < len(hashes) else None, lab) for k, (f, lab) in enumerate(zip(files, labels))
                if f != rel and (k >= len(hashes) or hashes[k] != sha)]
        if want_poster:       # the zoom poster comes first in source_files, labelled "poster"
            ev["poster"] = new_poster
            ev["poster_blur"] = new_blur     # always replaced with the new poster's blur, never left stale
            front = [(rel, sha, "poster")]
            rest_all = front + rest
        else:                 # thumbnail only: the image is kept as an extra source, the poster screenshot stays first
            rest_all = rest[:1] + [(rel, sha, "other")] + rest[1:] if rest else [(rel, sha, "other")]
        if want_thumb:
            ev["thumb"] = new_thumb
        ev["source_files"] = [r[0] for r in rest_all]
        ev["image_hashes"] = [r[1] for r in rest_all if r[1]]
        ev["source_labels"] = [r[2] for r in rest_all]
        for old, new in ((old_poster, ev.get("poster")), (old_thumb, ev.get("thumb"))):
            if old != new and not _derived_in_use(ctx, old, ev["id"]):
                ctx.remove_derived(old, keep=new)
        kept = []
        for r in ctx.review:
            if r.get("id") is None and r.get("title") == path.name:
                continue                                     # an earlier "not applied" note for this file
            if r.get("id") == ev["id"]:
                r.update(poster=ev["poster"], thumb=ev["thumb"], source_files=ev["source_files"])
                if want_poster:
                    r["reasons"] = [x for x in r.get("reasons", []) if x != "no poster"]
                if not r["reasons"]:
                    continue
            kept.append(r)
        ctx.review = kept
        if sha not in ctx.hashlog:
            ctx.record([Img(Path(rel), sha)], "image_override", ev["id"], ev.get("title"))
            ctx.hashlog[sha]["file"] = Path(rel).name
        ctx.counts["override"] += 1
        log.info("override %s: %s of event %s %r replaced", Path(rel).name,
                 {"poster": "thumbnail and poster", "thumb": "thumbnail", "zoom": "poster"}[kind], ev["id"], ev.get("title"))
        ctx.save()


def run(ctx: Ctx) -> None:
    files = sorted((p for p in ctx.inbox.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=inbox_sort_key)
    ctx.override_files = [(p, *parts) for p in files if (parts := override_parts(p))]
    files = [p for p in files if not override_parts(p)]
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
    merge_new(ctx)
    apply_overrides(ctx)
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
        try:
            merge_new(ctx)      # results finished before the error are kept, so they are merged too
        except Exception as e2:  # a failed merge must not hide the real error
            log.warning("merge step after the error failed: %s", e2)
        try:
            apply_overrides(ctx)    # these need no API call, so they still go through
        except Exception as e3:
            log.warning("applying image overrides after the error failed: %s", e3)
        summarize(ctx)
        return 1
    summarize(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
