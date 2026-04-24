from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

from aoede.languages import LANGUAGE_REGISTRY, language_token, normalize_language
from aoede.text.normalization import normalize_text


class UnicodeTokenizer:
    SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, vocab_path: Optional[Path] = None):
        self._sentencepiece = None
        self._model_path: Optional[Path] = None
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        if vocab_path and vocab_path.exists():
            payload = json.loads(vocab_path.read_text())
            self.token_to_id = {key: int(value) for key, value in payload["token_to_id"].items()}
            self.id_to_token = {int(key): value for key, value in payload["id_to_token"].items()}
            self._model_path = vocab_path
        else:
            self._bootstrap()

    def _bootstrap(self):
        tokens = list(self.SPECIAL) + [language_token(code) for code in LANGUAGE_REGISTRY]
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    @property
    def pad_id(self):
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self):
        return self.token_to_id["<unk>"]

    @property
    def size(self):
        return len(self.token_to_id)

    def fit(self, texts: Iterable[str], language_codes: Iterable[str]):
        for text, language_code in zip(texts, language_codes):
            resolved_language = normalize_language(language_code)
            normalized = normalize_text(text, resolved_language)
            self._ensure_token(language_token(resolved_language))
            for char in normalized:
                self._ensure_token(char)

    def _ensure_token(self, token: str):
        if token not in self.token_to_id:
            index = len(self.token_to_id)
            self.token_to_id[token] = index
            self.id_to_token[index] = token
        return self.token_to_id[token]

    def encode(self, text: str, language_code: str, add_new_tokens: bool = True):
        resolved_language = normalize_language(language_code)
        normalized = normalize_text(text, resolved_language)
        lang_tok = language_token(resolved_language)
        if add_new_tokens:
            self._ensure_token(lang_tok)
        lang_id = self.token_to_id.get(lang_tok, self.unk_id)
        ids = [self.token_to_id["<bos>"], lang_id]
        for char in normalized:
            if add_new_tokens:
                ids.append(self._ensure_token(char))
            else:
                ids.append(self.token_to_id.get(char, self.unk_id))
        ids.append(self.token_to_id["<eos>"])
        return ids

    def decode(self, ids: Sequence[int]):
        pieces = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), "<unk>")
            if token in self.SPECIAL or token.startswith("<lang:"):
                continue
            pieces.append(token)
        return "".join(pieces)

    def encode_batch(self, texts: Sequence[str], language_codes: Sequence[str]):
        return [self.encode(text, language_code) for text, language_code in zip(texts, language_codes)]

    def save(self, path: Path):
        payload = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(key): value for key, value in self.id_to_token.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path):
        return cls(vocab_path=path)
