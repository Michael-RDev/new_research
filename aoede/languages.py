from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional


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
    LanguageSpec("wo", "Wolof", "Niger-Congo", "Latin", False),
    LanguageSpec("yo", "Yoruba", "Niger-Congo", "Latin", False),
]

LANGUAGE_REGISTRY: Dict[str, LanguageSpec] = {
    spec.code: spec for spec in (*_PRODUCTION, *_EXPERIMENTAL)
}

LANGUAGE_ALIASES: Dict[str, str] = {
    "cmn_hans_cn": "zh",
    "cmn_hant_tw": "zh",
    "de_de": "de",
    "en_gb": "en",
    "en_us": "en",
    "es_419": "es",
    "es_es": "es",
    "fr_fr": "fr",
    "hi_in": "hi",
    "it_it": "it",
    "ja_jp": "ja",
    "ko_kr": "ko",
    "nl_nl": "nl",
    "pt_br": "pt",
    "pt_pt": "pt",
    "ru_ru": "ru",
    "sw_ke": "sw",
    "sw_tz": "sw",
    "tr_tr": "tr",
    "vi_vn": "vi",
    "wo_sn": "wo",
    "wol": "wo",
    "zh_cn": "zh",
}

_LANGUAGE_LOOKUP: Dict[str, str] = {
    **{spec.code.lower(): spec.code for spec in (*_PRODUCTION, *_EXPERIMENTAL)},
    **{spec.name.lower(): spec.code for spec in (*_PRODUCTION, *_EXPERIMENTAL)},
}
LANGUAGE_INDEX: Dict[str, int] = {
    spec.code: index for index, spec in enumerate((*_PRODUCTION, *_EXPERIMENTAL), start=1)
}


def production_languages():
    return list(_PRODUCTION)


def experimental_languages():
    return list(_EXPERIMENTAL)


def all_languages():
    return [*_PRODUCTION, *_EXPERIMENTAL]


def canonical_language(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    raw = str(value).strip()
    if not raw or raw.lower() == "none":
        return None

    lowered = raw.lower()
    if lowered in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[lowered]
    if lowered in _LANGUAGE_LOOKUP:
        return _LANGUAGE_LOOKUP[lowered]

    prefix = lowered.split("-", 1)[0]
    if prefix in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[prefix]
    if prefix in _LANGUAGE_LOOKUP:
        return _LANGUAGE_LOOKUP[prefix]
    return None


def normalize_language(value: Optional[str], default: str = "en") -> str:
    canonical = canonical_language(value)
    if canonical is not None:
        return canonical

    if value is None:
        return default

    raw = str(value).strip()
    if not raw or raw.lower() == "none":
        return default

    prefix = raw.lower().split("-", 1)[0]
    return prefix or default


def language_index(value: Optional[str]) -> int:
    return LANGUAGE_INDEX.get(normalize_language(value), 0)


def resolve_language(code: str):
    normalized = normalize_language(code, default=code)
    return LANGUAGE_REGISTRY.get(
        normalized,
        LanguageSpec(normalized, normalized.upper(), "unknown", "unknown", False),
    )


def language_token(code: str):
    return f"<lang:{normalize_language(code, default=code)}>"
