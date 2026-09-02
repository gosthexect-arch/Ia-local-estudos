"""Saida de terminal sem dependencias externas.

Cores sao desligadas automaticamente quando a saida nao e um TTY, quando
NO_COLOR esta definido ou quando TERM=dumb.
"""

from __future__ import annotations

import os
import sys
from typing import IO

_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "grey": "\033[90m",
}


def color_enabled(stream: IO[str] | None = None, env: dict[str, str] | None = None) -> bool:
    stream = stream or sys.stdout
    env = os.environ if env is None else env
    if env.get("NO_COLOR") is not None:
        return False
    if env.get("FORCE_COLOR"):
        return True
    if env.get("TERM") == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


class Painter:
    """Aplica (ou nao) codigos ANSI conforme o destino da saida."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, text: str, *styles: str) -> str:
        if not self.enabled or not styles:
            return text
        prefix = "".join(_CODES[s] for s in styles if s in _CODES)
        return f"{prefix}{text}{_CODES['reset']}" if prefix else text


def human_bytes(n: float) -> str:
    step = 1024.0
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < step or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= step
    return f"{n:.1f} TiB"


def human_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.2f} s"
