"""Recovery parser for malformed JSON. Never fabricates data — drops what
it can't trust instead of guessing (no null-filling dangling keys, no
completing truncated keywords, no promoting stray tokens to keys)."""

from __future__ import annotations

import json
import re
from typing import Any, List, Tuple

__all__ = ["parse_dirty", "safe_load", "repair_json", "try_parse_llm_json"]

Token = Tuple[str, Any, bool]
_SENTINEL = object()


class JSONParser:
    """Tokenizes and parses malformed JSON in one pass."""

    _NUMBER_RE = re.compile(r"-?(\d+)(\.\d+)?([eE][+-]?\d+)?")
    _WHITESPACE = " \t\n\r"
    _STRUCTURAL_CHARS = "{}[]:,"
    _STRING_TERMINATOR_FOLLOWERS = ',:}]"'
    _ESCAPE_MAP = {
        "n": "\n", "t": "\t", "r": "\r", '"': '"',
        "'": "'", "\\": "\\", "/": "/", "b": "\b", "f": "\f",
    }
    _NULL_LITERALS = ("null", "none", "nan")
    _KEYWORD_PREFIXES = ("true", "false", "null")

    def __init__(self, text: str, keep_partial_strings: bool = True):
        self.keep_partial_strings = keep_partial_strings
        self.tokens: List[Token] = self._tokenize(text)
        self.pos = 0

    def parse(self) -> Any:
        if not self.tokens:
            return {}
        first_kind = self.tokens[0][0]
        second_kind = self.tokens[1][0] if len(self.tokens) > 1 else None
        if first_kind in ("STRING", "BAREWORD") and second_kind == ":":
            return self._parse_object()

        values = []
        while self._peek()[0] is not None:
            pos_before = self.pos
            value = self._parse_value()
            if value is not _SENTINEL:
                values.append(value)
            if self.pos == pos_before:
                self._advance()

        if not values:
            return {}
        if len(values) == 1:
            return values[0]
        if all(isinstance(v, dict) for v in values):
            merged: dict = {}
            for v in values:
                merged.update(v)
            return merged
        return values

    def _tokenize(self, text: str) -> List[Token]:
        tokens: List[Token] = []
        i, n = 0, len(text)

        while i < n:
            c = text[i]
            if c in self._WHITESPACE:
                i += 1
            elif c in self._STRUCTURAL_CHARS:
                tokens.append((c, c, True))
                i += 1
            elif c in "\"'":
                token, i = self._consume_string(text, i, quote=c)
                tokens.append(token)
            else:
                match = self._NUMBER_RE.match(text, i)
                if match:
                    token, i = self._consume_number(match)
                else:
                    token, i = self._consume_bareword(text, i)
                tokens.append(token)

        return tokens

    def _consume_string(self, text: str, start: int, quote: str) -> Tuple[Token, int]:
        i, n = start + 1, len(text)
        buf: List[str] = []
        closed = False

        while i < n:
            ch = text[i]

            if ch == "\\" and i + 1 < n:
                nxt = text[i + 1]
                if nxt == "u" and i + 5 < n:
                    try:
                        buf.append(chr(int(text[i + 2:i + 6], 16)))
                        i += 6
                        continue
                    except ValueError:
                        pass
                buf.append(self._ESCAPE_MAP.get(nxt, nxt))
                i += 2
                continue

            if ch == quote:
                j = i + 1
                while j < n and text[j] in self._WHITESPACE:
                    j += 1
                if j >= n or text[j] in self._STRING_TERMINATOR_FOLLOWERS:
                    i += 1
                    closed = True
                    break
                buf.append(ch)
                i += 1
                continue

            buf.append(ch)
            i += 1

        return ("STRING", "".join(buf), closed), i

    def _consume_number(self, match: "re.Match[str]") -> Tuple[Token, int]:
        raw = match.group(0)
        is_float = "." in raw or "e" in raw or "E" in raw
        return ("NUMBER", float(raw) if is_float else int(raw), True), match.end()

    def _consume_bareword(self, text: str, start: int) -> Tuple[Token, int]:
        i, n = start, len(text)
        while i < n and text[i] not in self._WHITESPACE \
                and text[i] not in self._STRUCTURAL_CHARS and text[i] not in "\"'":
            i += 1

        word = text[start:i]
        lowered = word.lower()

        if lowered == "true":
            return ("BOOL", True, True), i
        if lowered == "false":
            return ("BOOL", False, True), i
        if lowered in self._NULL_LITERALS:
            return ("NULL", None, True), i
        if i >= n and any(kw.startswith(lowered) for kw in self._KEYWORD_PREFIXES):
            return ("PARTIAL_LITERAL", word, False), i 
        return ("BAREWORD", word, True), i

    def _peek(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return (None, None, True)

    def _advance(self) -> Token:
        token = self._peek()
        self.pos += 1
        return token

    def _parse_value(self) -> Any:
        while True:
            kind, value, complete = self._peek()

            if kind is None:
                return _SENTINEL
            if kind == "{":
                self._advance()
                return self._parse_object()
            if kind == "[":
                self._advance()
                return self._parse_array()
            if kind == "STRING":
                self._advance()
                if not complete and not self.keep_partial_strings:
                    return _SENTINEL
                return value
            if kind in ("NUMBER", "BOOL", "NULL", "BAREWORD"):
                self._advance()
                return value
            if kind == "PARTIAL_LITERAL":
                self._advance()
                return _SENTINEL
            if kind in ("}", "]"):
                return _SENTINEL

            self._advance()

    def _parse_object(self) -> dict:
        obj: dict = {}

        while True:
            kind, value, complete = self._peek()

            if kind is None:
                return obj
            if kind in ("}", "]"):
                self._advance()
                return obj
            if kind in (",", ":"):
                self._advance()
                continue

            if kind in ("STRING", "BAREWORD"):
                key, key_complete = value, complete
                self._advance()
            elif kind in ("{", "["):
                self._parse_value()
                continue
            else:
                self._advance()
                continue

            if self._peek()[0] == ":":
                self._advance()

            kind = self._peek()[0]
            if kind in (None, "}", "]"):
                continue  
            if kind == ",":
                self._advance()
                continue

            parsed_value = self._parse_value()
            can_keep = parsed_value is not _SENTINEL and (key_complete or self.keep_partial_strings)
            if can_keep and (key != "" or parsed_value is not _SENTINEL):
                obj[key] = parsed_value

            if self._peek()[0] == ",":
                self._advance()

    def _parse_array(self) -> list:
        arr: list = []

        while True:
            kind = self._peek()[0]

            if kind is None:
                return arr
            if kind in ("]", "}"):
                self._advance()
                return arr
            if kind in (",", ":"):
                self._advance()
                continue

            value = self._parse_value()
            if value is not _SENTINEL:
                arr.append(value)

            if self._peek()[0] == ",":
                self._advance()

def parse_dirty(text: str, keep_partial_strings: bool = True) -> Any:
    return JSONParser(text, keep_partial_strings).parse()


def safe_load(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return parse_dirty(text)


def repair_json(text: str, indent: int = 2, keep_partial_strings: bool = True) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = parse_dirty(text, keep_partial_strings=keep_partial_strings)
    return json.dumps(data, ensure_ascii=False, indent=indent)


def try_parse_llm_json(raw: str) -> Tuple[Any, str]:
    try:
        return json.loads(raw), "clean"
    except json.JSONDecodeError:
        pass
    try:
        return parse_dirty(raw), "repaired"
    except Exception:
        return None, "failed"