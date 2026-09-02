"""Interface abstrata de exchange.

A estrategia e o armazenamento falam apenas esta linguagem. Trocar Binance por
Coinbase ou Kraken deve ser escrever um adaptador novo, sem tocar no resto.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from cripto_monitor.candles import Candle


class ExchangeAdapter(ABC):
    """Traduz o dialeto de uma exchange para velas do dominio."""

    name: str

    def __init__(self, rest_base: str, ws_base: str) -> None:
        self.rest_base = rest_base.rstrip("/")
        self.ws_base = ws_base.rstrip("/")

    @abstractmethod
    def stream_url(self, symbol: str, timeframe: str) -> str:
        """URL do stream continuo de velas."""

    def subscribe_payload(self, symbol: str, timeframe: str) -> dict[str, Any] | None:
        """Mensagem de inscricao, para exchanges que exigem uma. None se a URL basta."""
        return None

    @abstractmethod
    def parse_message(self, raw: str) -> Candle | None:
        """Converte uma mensagem do stream em vela, ou None se nao for de vela."""

    @abstractmethod
    def backfill_url(
        self, symbol: str, timeframe: str, *, start_ms: int | None = None, limit: int = 500
    ) -> str:
        """URL REST para recuperar velas historicas (usada apos lacuna ou reinicio)."""

    @abstractmethod
    def parse_backfill(self, payload: Any, symbol: str, timeframe: str, *, now_ms: int) -> list[Candle]:
        """Converte a resposta REST em velas, marcando quais ja fecharam."""

    @property
    @abstractmethod
    def max_backfill_limit(self) -> int:
        """Maximo de velas por requisicao REST aceito pela exchange."""


def get_adapter(name: str, rest_base: str, ws_base: str) -> ExchangeAdapter:
    from cripto_monitor.exchanges.binance import BinanceAdapter

    adaptadores: dict[str, type[ExchangeAdapter]] = {"binance": BinanceAdapter}
    try:
        classe = adaptadores[name.lower()]
    except KeyError:
        disponiveis = ", ".join(sorted(adaptadores))
        raise ValueError(f"exchange sem adaptador: {name!r} (disponiveis: {disponiveis})") from None
    return classe(rest_base, ws_base)
