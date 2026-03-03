"""Text normalization for input preprocessing.

Applied before tokenization to reduce adversarial surface area:
unicode normalization, zero-width stripping, whitespace collapse.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width and invisible unicode characters
_ZERO_WIDTH = re.compile(
    "[\u200b\u200c\u200d\u200e\u200f"  # zero-width space, joiners, marks
    "\u2060\u2061\u2062\u2063\u2064"    # word joiner, invisible operators
    "\ufeff"                             # BOM / zero-width no-break space
    "\u00ad"                             # soft hyphen
    "\u034f"                             # combining grapheme joiner
    "\u061c"                             # arabic letter mark
    "\u115f\u1160"                       # hangul fillers
    "\u17b4\u17b5"                       # khmer vowel inherent
    "\u180e"                             # mongolian vowel separator
    "\uffa0"                             # halfwidth hangul filler
    "]"
)

# Collapse runs of whitespace (spaces, tabs, newlines) to single space
_WHITESPACE = re.compile(r"\s+")

# Detect base64-ish blobs (20+ chars of base64 alphabet without spaces)
_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/=]{20,}")

DEFAULT_MAX_CHARS = 2000


def normalize(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Normalize input text for classification.

    Steps:
    1. Unicode NFKC normalization (collapses fullwidth, compatibility chars)
    2. Strip zero-width and invisible characters
    3. Collapse whitespace
    4. Truncate to max length
    """
    # NFKC: decomposes then composes by compatibility
    # This handles fullwidth chars, superscripts, ligatures, etc.
    text = unicodedata.normalize("NFKC", text)

    # Strip invisible characters
    text = _ZERO_WIDTH.sub("", text)

    # Collapse whitespace
    text = _WHITESPACE.sub(" ", text).strip()

    # Truncate
    if len(text) > max_chars:
        text = text[:max_chars]

    return text


def has_encoding_tricks(text: str) -> bool:
    """Check if the text contains patterns that suggest encoding-based evasion.

    Returns True if the text has base64 blobs or heavy unicode mixing that
    may indicate an attempt to bypass classification.
    """
    if _BASE64_BLOB.search(text):
        return True

    if not text:
        return False

    # Check for unusually high ratio of non-ASCII characters
    # Normal multilingual text is fine; random unicode salad is suspicious
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ratio = non_ascii / len(text) if text else 0

    # Flag if >60% non-ASCII and text is short (likely obfuscation, not CJK prose)
    if ratio > 0.6 and len(text) < 200:
        return True

    return False
