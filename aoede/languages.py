from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Union


@dataclass(frozen=True)
class LanguageSpec:
    code: str
    name: str
    family: str
    script: str
    production: bool

    def to_dict(self):
        return asdict(self)


_PRODUCTION = [
    LanguageSpec("en", "English", "Indo-European", "Latin", True),
    LanguageSpec("es", "Spanish", "Indo-European", "Latin", True),
    LanguageSpec("fr", "French", "Indo-European", "Latin", True),
    LanguageSpec("de", "German", "Indo-European", "Latin", True),
    LanguageSpec("pt", "Portuguese", "Indo-European", "Latin", True),
    LanguageSpec("it", "Italian", "Indo-European", "Latin", True),
    LanguageSpec("nl", "Dutch", "Indo-European", "Latin", True),
    LanguageSpec("pl", "Polish", "Indo-European", "Latin", True),
    LanguageSpec("ru", "Russian", "Indo-European", "Cyrillic", True),
    LanguageSpec("tr", "Turkish", "Turkic", "Latin", True),
    LanguageSpec("ar", "Arabic", "Afro-Asiatic", "Arabic", True),
    LanguageSpec("hi", "Hindi", "Indo-European", "Devanagari", True),
    LanguageSpec("id", "Indonesian", "Austronesian", "Latin", True),
    LanguageSpec("vi", "Vietnamese", "Austroasiatic", "Latin", True),
    LanguageSpec("sw", "Swahili", "Niger-Congo", "Latin", True),
    LanguageSpec("zh", "Mandarin Chinese", "Sino-Tibetan", "Han", True),
]

_EXPERIMENTAL = [
    LanguageSpec("ja", "Japanese", "Japonic", "Kana/Kanji", False),
    LanguageSpec("ko", "Korean", "Koreanic", "Hangul", False),
    LanguageSpec("th", "Thai", "Kra-Dai", "Thai", False),
    LanguageSpec("uk", "Ukrainian", "Indo-European", "Cyrillic", False),
    LanguageSpec("ro", "Romanian", "Indo-European", "Latin", False),
    LanguageSpec("cs", "Czech", "Indo-European", "Latin", False),
    LanguageSpec("fa", "Persian", "Indo-European", "Arabic", False),
    LanguageSpec("bn", "Bengali", "Indo-European", "Bengali", False),
    LanguageSpec("ta", "Tamil", "Dravidian", "Tamil", False),
    LanguageSpec("yo", "Yoruba", "Niger-Congo", "Latin", False),
]

LANGUAGE_REGISTRY: Dict[str, LanguageSpec] = {
    spec.code: spec for spec in (*_PRODUCTION, *_EXPERIMENTAL)
}


def production_languages():
    return list(_PRODUCTION)


def experimental_languages():
    return list(_EXPERIMENTAL)


def resolve_language(code: str):
    return LANGUAGE_REGISTRY.get(code, LanguageSpec(code, code.upper(), "unknown", "unknown", False))


def language_token(code: str):
    return f"<lang:{code}>"
