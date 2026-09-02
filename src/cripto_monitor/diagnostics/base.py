"""Tipos comuns dos diagnosticos."""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


# Quanto maior, pior. Usado para decidir o status agregado e o exit code.
_SEVERITY = {Status.SKIP: 0, Status.OK: 1, Status.WARN: 2, Status.FAIL: 3}


@dataclass(slots=True)
class CheckResult:
    name: str
    status: Status
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
            "details": self.details,
            "duration_ms": round(self.duration_ms, 2),
            "hint": self.hint,
        }


Check = Callable[[], CheckResult]


def worst(results: Iterable[CheckResult]) -> Status:
    """Status agregado. Sem resultados, considera SKIP."""
    return max((r.status for r in results), key=lambda s: _SEVERITY[s], default=Status.SKIP)


def severity(status: Status) -> int:
    return _SEVERITY[status]


def run_checks(checks: Iterable[tuple[str, Check]]) -> list[CheckResult]:
    """Executa cada verificacao isolando excecoes: um erro nao derruba o resto."""
    results: list[CheckResult] = []
    for name, check in checks:
        started = time.perf_counter()
        try:
            result = check()
        except Exception as exc:  # diagnostico nunca deve abortar por uma falha isolada
            result = CheckResult(
                name=name,
                status=Status.FAIL,
                summary=f"erro inesperado: {type(exc).__name__}: {exc}",
                details={"traceback": traceback.format_exc(limit=4)},
            )
        if not result.duration_ms:
            result.duration_ms = (time.perf_counter() - started) * 1000
        results.append(result)
    return results
