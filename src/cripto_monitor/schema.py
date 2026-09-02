"""Contrato de saida do modelo.

O alerta NAO e uma ordem: e uma leitura de cenario, rastreavel ate os dados que
a originaram. Qualquer resposta que nao passe por `validate_alert` deve ser
descartada e registrada como INVALID_RESPONSE.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

SCHEMA_VERSION = "alert-v1"

DECISIONS = ("BUY_BIAS", "SELL_BIAS", "HOLD", "DATA_UNAVAILABLE")
REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE", "UNKNOWN")
ALIGNMENTS = ("BULLISH", "BEARISH", "NEUTRAL")
DATA_QUALITY = ("GOOD", "DEGRADED", "STALE", "UNAVAILABLE")
ALIGNMENT_TIMEFRAMES = ("1h", "15m", "5m")

REQUIRED_FIELDS = (
    "decision",
    "confidence",
    "regime",
    "timeframe_alignment",
    "thesis",
    "invalidation",
    "evidence_ids",
    "data_quality",
    "generated_at",
)

EXAMPLE_ALERT: dict[str, Any] = {
    "decision": "HOLD",
    "confidence": 0.0,
    "regime": "RANGE",
    "timeframe_alignment": {"1h": "NEUTRAL", "15m": "NEUTRAL", "5m": "NEUTRAL"},
    "thesis": "Sem confirmacao suficiente.",
    "invalidation": "A hipotese perde validade se os niveis estruturais mudarem.",
    "evidence_ids": [],
    "data_quality": "GOOD",
    "generated_at": "2026-01-01T00:00:00Z",
}


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate_alert(payload: Any, *, known_evidence: set[str] | None = None) -> list[str]:
    """Retorna a lista de erros. Lista vazia significa resposta aceitavel."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return [f"resposta deve ser um objeto JSON, veio {type(payload).__name__}"]

    faltando = [f for f in REQUIRED_FIELDS if f not in payload]
    if faltando:
        errors.append(f"campos obrigatorios ausentes: {', '.join(faltando)}")
    extras = [k for k in payload if k not in REQUIRED_FIELDS]
    if extras:
        errors.append(f"campos desconhecidos: {', '.join(sorted(extras))}")

    decision = payload.get("decision")
    if "decision" in payload and decision not in DECISIONS:
        errors.append(f"decision invalida: {decision!r}")

    confidence = payload.get("confidence")
    if "confidence" in payload:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            errors.append("confidence deve ser numerica")
        elif not 0.0 <= float(confidence) <= 1.0:
            errors.append(f"confidence fora do intervalo [0,1]: {confidence}")

    if "regime" in payload and payload["regime"] not in REGIMES:
        errors.append(f"regime invalido: {payload['regime']!r}")

    if "data_quality" in payload and payload["data_quality"] not in DATA_QUALITY:
        errors.append(f"data_quality invalido: {payload['data_quality']!r}")

    alignment = payload.get("timeframe_alignment")
    if "timeframe_alignment" in payload:
        if not isinstance(alignment, dict):
            errors.append("timeframe_alignment deve ser um objeto")
        else:
            for timeframe in ALIGNMENT_TIMEFRAMES:
                if timeframe not in alignment:
                    errors.append(f"timeframe_alignment sem o periodo {timeframe}")
                elif alignment[timeframe] not in ALIGNMENTS:
                    errors.append(
                        f"timeframe_alignment[{timeframe}] invalido: {alignment[timeframe]!r}"
                    )

    for campo in ("thesis", "invalidation"):
        if campo in payload and not isinstance(payload[campo], str):
            errors.append(f"{campo} deve ser texto")

    evidence = payload.get("evidence_ids")
    if "evidence_ids" in payload:
        if not isinstance(evidence, list) or not all(isinstance(e, str) for e in evidence):
            errors.append("evidence_ids deve ser uma lista de strings")
        elif known_evidence is not None:
            desconhecidas = [e for e in evidence if e not in known_evidence]
            if desconhecidas:
                errors.append(f"evidencias inexistentes: {', '.join(desconhecidas)}")

    generated_at = payload.get("generated_at")
    if "generated_at" in payload:
        if not isinstance(generated_at, str):
            errors.append("generated_at deve ser texto ISO-8601 em UTC")
        else:
            try:
                parsed = _parse_timestamp(generated_at)
            except ValueError:
                errors.append(f"generated_at nao e ISO-8601 valido: {generated_at!r}")
            else:
                if parsed.tzinfo is None:
                    errors.append("generated_at precisa de fuso explicito (UTC)")

    # Coerencia: sem dados nao ha viés direcional.
    if decision in ("BUY_BIAS", "SELL_BIAS") and payload.get("data_quality") == "UNAVAILABLE":
        errors.append("vies direcional incompativel com data_quality=UNAVAILABLE")

    return errors
