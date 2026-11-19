r"""Encoding of key names into the byte sequences a terminal sends.

Key names are case-insensitive:

- Named keys: `Enter`, `Tab`, `Backspace`, `Escape`, `Space`, `Up`, `Down`, `Left`,
  `Right`, `Home`, `End`, `PageUp`, `PageDown`, `Insert`, `Delete`, `F1`-`F12`
- Control combinations: `C-a` ... `C-z`, `C-@`/`C-Space`, `C-[`, `C-\`, `C-]`
- Alt/Meta combinations: `M-<key>` prefixes the key's encoding with ESC (e.g. `M-x`,
  `M-Enter`, `M-C-c`)
- Any other single character is sent as-is (UTF-8)
"""

from __future__ import annotations


ESC = "\x1b"

_SIMPLE_KEYS: dict[str, str] = {
    "enter": "\r",
    "return": "\r",
    "tab": "\t",
    "backspace": "\x7f",
    "escape": ESC,
    "esc": ESC,
    "space": " ",
    "pageup": f"{ESC}[5~",
    "pagedown": f"{ESC}[6~",
    "insert": f"{ESC}[2~",
    "delete": f"{ESC}[3~",
    "f1": f"{ESC}OP",
    "f2": f"{ESC}OQ",
    "f3": f"{ESC}OR",
    "f4": f"{ESC}OS",
    "f5": f"{ESC}[15~",
    "f6": f"{ESC}[17~",
    "f7": f"{ESC}[18~",
    "f8": f"{ESC}[19~",
    "f9": f"{ESC}[20~",
    "f10": f"{ESC}[21~",
    "f11": f"{ESC}[23~",
    "f12": f"{ESC}[24~",
}

# Cursor keys depend on DECCKM (application cursor keys mode).
_CURSOR_KEYS: dict[str, str] = {
    "up": "A",
    "down": "B",
    "right": "C",
    "left": "D",
    "home": "H",
    "end": "F",
}

_CONTROL_SPECIALS: dict[str, str] = {
    "@": "\x00",
    "space": "\x00",
    "[": ESC,
    "\\": "\x1c",
    "]": "\x1d",
    "^": "\x1e",
    "_": "\x1f",
    "?": "\x7f",
}


def encode_key(name: str, *, application_cursor: bool = False) -> bytes:
    """Encode a key name as the bytes a terminal would send.

    Args:
        name: Key name (see module docstring)
        application_cursor: Whether the terminal is in application cursor keys
            mode (DECCKM), which changes the encoding of cursor keys

    Returns:
        Encoded key sequence

    Raises:
        ValueError: If the key name is not recognized
    """
    return _encode(name, application_cursor=application_cursor).encode()


def _encode(name: str, *, application_cursor: bool) -> str:
    if len(name) == 1:
        return name
    lowered = name.lower()
    if lowered.startswith("m-") and len(name) > 2:  # noqa: PLR2004
        return ESC + _encode(name[2:], application_cursor=application_cursor)
    if lowered.startswith("c-") and len(name) > 2:  # noqa: PLR2004
        return _control(name[2:])
    if lowered in _SIMPLE_KEYS:
        return _SIMPLE_KEYS[lowered]
    if lowered in _CURSOR_KEYS:
        prefix = f"{ESC}O" if application_cursor else f"{ESC}["
        return prefix + _CURSOR_KEYS[lowered]
    msg = f"Unknown key name: {name!r}"
    raise ValueError(msg)


def _control(key: str) -> str:
    lowered = key.lower()
    if lowered in _CONTROL_SPECIALS:
        return _CONTROL_SPECIALS[lowered]
    if len(key) == 1 and "a" <= lowered <= "z":
        return chr(ord(lowered) - ord("a") + 1)
    msg = f"Unknown control key: C-{key!r}"
    raise ValueError(msg)
