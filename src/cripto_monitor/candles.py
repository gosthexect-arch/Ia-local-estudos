"""Modelo de vela e aritmetica de timeframes.

Regra central do projeto: apenas velas fechadas alimentam decisao. Uma vela em
formacao existe como estado corrente, nunca como fato persistido.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class CandleError(ValueError):
    """Dado de vela inconsistente: nunca deve virar fato persistido."""


def timeframe_ms(timeframe: str) -> int:
    try:
        return TIMEFRAME_MS[timeframe]
    except KeyError:
        raise CandleError(f"timeframe nao suportado: {timeframe!r}") from None


def align_open_time(timestamp_ms: int, timeframe: str) -> int:
    """Inicio da vela que contem o instante informado."""
    passo = timeframe_ms(timeframe)
    return timestamp_ms - (timestamp_ms % passo)


def iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat(
        timespec="milliseconds"
    )


@dataclass(frozen=True, slots=True)
class Candle:
    source: str
    symbol: str
    timeframe: str
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades: int = 0
    closed: bool = False
    origin: str = "websocket"

    @property
    def key(self) -> tuple[str, str, str, int]:
        """Chave de idempotencia: fonte, simbolo, timeframe e abertura."""
        return (self.source, self.symbol, self.timeframe, self.open_time_ms)

    @property
    def open_time_utc(self) -> str:
        return iso(self.open_time_ms)

    @property
    def close_time_utc(self) -> str:
        return iso(self.close_time_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "open_time_ms": self.open_time_ms,
            "close_time_ms": self.close_time_ms,
            "open_time_utc": self.open_time_utc,
            "close_time_utc": self.close_time_utc,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "trades": self.trades,
            "state": "closed" if self.closed else "forming",
            "origin": self.origin,
        }


def validate(candle: Candle) -> Candle:
    """Rejeita vela impossivel. Em duvida, o dado nao entra no historico."""
    passo = timeframe_ms(candle.timeframe)
    if candle.open_time_ms % passo:
        raise CandleError(
            f"abertura desalinhada com o timeframe {candle.timeframe}: {candle.open_time_utc}"
        )
    duracao = candle.close_time_ms - candle.open_time_ms
    # A Binance fecha a vela em open + passo - 1 ms; aceitamos as duas convencoes.
    if duracao not in (passo, passo - 1):
        raise CandleError(f"duracao inesperada: {duracao} ms para {candle.timeframe}")
    precos = (candle.open, candle.high, candle.low, candle.close)
    if any(p <= 0 for p in precos):
        raise CandleError(f"preco invalido em {candle.open_time_utc}: {precos}")
    if candle.high < max(candle.open, candle.close) or candle.low > min(candle.open, candle.close):
        raise CandleError(f"maxima/minima incoerentes em {candle.open_time_utc}")
    if candle.high < candle.low:
        raise CandleError(f"maxima menor que minima em {candle.open_time_utc}")
    if candle.volume < 0 or candle.quote_volume < 0 or candle.trades < 0:
        raise CandleError(f"volume negativo em {candle.open_time_utc}")
    return candle


def expected_open_times(inicio_ms: int, fim_ms: int, timeframe: str) -> list[int]:
    """Aberturas esperadas no intervalo fechado [inicio, fim]."""
    passo = timeframe_ms(timeframe)
    if fim_ms < inicio_ms:
        return []
    return list(range(align_open_time(inicio_ms, timeframe), fim_ms + 1, passo))


def find_gaps(open_times: Iterable[int], timeframe: str) -> list[tuple[int, int]]:
    """Buracos na sequencia, como pares (primeira ausente, ultima ausente)."""
    passo = timeframe_ms(timeframe)
    ordenados = sorted(set(open_times))
    lacunas: list[tuple[int, int]] = []
    for anterior, proximo in zip(ordenados, ordenados[1:]):
        if proximo - anterior > passo:
            lacunas.append((anterior + passo, proximo - passo))
    return lacunas


def merge(anterior: Candle, nova: Candle) -> Candle:
    """Resolve o reencontro de duas versoes da mesma vela.

    Uma vela ja fechada nunca e rebaixada para em formacao: reconexao e replay
    reentregam eventos antigos, e o historico nao pode regredir.
    """
    if anterior.key != nova.key:
        raise CandleError("tentativa de mesclar velas de chaves diferentes")
    if anterior.closed and not nova.closed:
        return anterior
    return replace(nova, origin=nova.origin if nova.closed else anterior.origin)


def only_closed(candles: Iterable[Candle]) -> Iterator[Candle]:
    return (c for c in candles if c.closed)
