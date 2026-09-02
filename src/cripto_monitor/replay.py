"""Replay do historico armazenado.

O criterio de aceitacao do projeto e que os resultados sejam reproduziveis por
replay. Este modulo reentrega as velas gravadas na ordem original, para que a
Fase 3 recalcule indicadores sobre exatamente os mesmos fatos.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from cripto_monitor.candles import Candle, expected_open_times, find_gaps, iso, timeframe_ms
from cripto_monitor.storage import Storage


class ReplayError(ValueError):
    pass


def parse_instante(texto: str) -> int:
    """Aceita ISO-8601 (UTC assumido quando o fuso e omitido) ou epoch em ms."""
    if texto.isdigit():
        return int(texto)
    try:
        momento = datetime.fromisoformat(texto.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReplayError(f"instante invalido: {texto!r} (use ISO-8601 ou epoch em ms)") from exc
    if momento.tzinfo is None:
        momento = momento.replace(tzinfo=timezone.utc)
    return int(momento.timestamp() * 1000)


@dataclass(slots=True)
class ReplayResumo:
    symbol: str
    timeframe: str
    velas: int
    primeira_utc: str | None
    ultima_utc: str | None
    lacunas: list[dict[str, Any]]
    continuo: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "velas": self.velas,
            "primeira_utc": self.primeira_utc,
            "ultima_utc": self.ultima_utc,
            "lacunas": self.lacunas,
            "continuo": self.continuo,
        }


def replay(
    storage: Storage,
    symbol: str,
    timeframe: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int | None = None,
) -> Iterator[Candle]:
    """Velas fechadas em ordem cronologica estrita.

    A ordenacao vem do banco, nao da ordem de chegada: eventos fora de ordem no
    stream nao contaminam a reproducao.
    """
    anterior: int | None = None
    passo = timeframe_ms(timeframe)
    for vela in storage.iter_candles(
        symbol, timeframe, start_ms=start_ms, end_ms=end_ms, limit=limit
    ):
        if anterior is not None and vela.open_time_ms <= anterior:
            raise ReplayError(
                f"historico fora de ordem em {vela.open_time_utc}: o banco esta corrompido"
            )
        if anterior is not None and (vela.open_time_ms - anterior) % passo:
            raise ReplayError(f"vela fora da grade de {timeframe} em {vela.open_time_utc}")
        anterior = vela.open_time_ms
        yield vela


def resumir(
    storage: Storage,
    symbol: str,
    timeframe: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> ReplayResumo:
    velas = list(replay(storage, symbol, timeframe, start_ms=start_ms, end_ms=end_ms))
    aberturas = [v.open_time_ms for v in velas]
    lacunas = find_gaps(aberturas, timeframe)
    return ReplayResumo(
        symbol=symbol,
        timeframe=timeframe,
        velas=len(velas),
        primeira_utc=velas[0].open_time_utc if velas else None,
        ultima_utc=velas[-1].open_time_utc if velas else None,
        lacunas=[
            {
                "de_utc": iso(inicio),
                "ate_utc": iso(fim),
                "velas": len(expected_open_times(inicio, fim, timeframe)),
            }
            for inicio, fim in lacunas
        ],
        continuo=not lacunas,
    )
