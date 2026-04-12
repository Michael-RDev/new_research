from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, language_code: str):
    text = text.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    if language_code in {"en", "es", "fr", "de", "pt", "it", "nl", "pl", "id", "vi", "sw", "tr", "wo"}:
        text = text.lower()
    return text
