"""Adaptador da Binance: apenas dados publicos de mercado.

Nenhum endpoint autenticado, de conta ou de ordens e usado aqui, e nenhuma
assinatura de requisicao existe neste modulo.
"""

from __future__ import annotations

import json
from typing import Any

from cripto_monitor.candles import Candle, CandleError, validate
from cripto_monitor.exchanges.base import ExchangeAdapter

# Ordem dos campos em cada linha de /api/v3/klines.
KLINE_FIELDS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
)


class BinanceAdapter(ExchangeAdapter):
    name = "binance"

    @property
    def max_backfill_limit(self) -> int:
        return 1000

    def stream_url(self, symbol: str, timeframe: str) -> str:
        return f"{self.ws_base}/ws/{symbol.lower()}@kline_{timeframe}"

    def parse_message(self, raw: str) -> Candle | None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CandleError(f"mensagem nao e JSON: {exc}") from exc
        if not isinstance(payload, dict):
            return None
        # Respostas de controle (resultado de subscribe, pong) nao carregam vela.
        kline = payload.get("k")
        if payload.get("e") != "kline" or not isinstance(kline, dict):
            return None
        return validate(
            Candle(
                source=self.name,
                symbol=str(kline["s"]).upper(),
                timeframe=str(kline["i"]),
                open_time_ms=int(kline["t"]),
                close_time_ms=int(kline["T"]),
                open=float(kline["o"]),
                high=float(kline["h"]),
                low=float(kline["l"]),
                close=float(kline["c"]),
                volume=float(kline["v"]),
                quote_volume=float(kline.get("q", 0.0)),
                trades=int(kline.get("n", 0)),
                closed=bool(kline.get("x")),
                origin="websocket",
            )
        )

    def backfill_url(
        self, symbol: str, timeframe: str, *, start_ms: int | None = None, limit: int = 500
    ) -> str:
        limite = max(1, min(int(limit), self.max_backfill_limit))
        url = (
            f"{self.rest_base}/api/v3/klines?symbol={symbol.upper()}"
            f"&interval={timeframe}&limit={limite}"
        )
        if start_ms is not None:
            url += f"&startTime={int(start_ms)}"
        return url

    def parse_backfill(
        self, payload: Any, symbol: str, timeframe: str, *, now_ms: int
    ) -> list[Candle]:
        if not isinstance(payload, list):
            raise CandleError("resposta de klines nao e uma lista")
        velas: list[Candle] = []
        for linha in payload:
            if not isinstance(linha, list) or len(linha) < 9:
                raise CandleError(f"linha de kline com formato inesperado: {linha!r:.80}")
            dados = dict(zip(KLINE_FIELDS, linha))
            close_time = int(dados["close_time"])
            velas.append(
                validate(
                    Candle(
                        source=self.name,
                        symbol=symbol.upper(),
                        timeframe=timeframe,
                        open_time_ms=int(dados["open_time"]),
                        close_time_ms=close_time,
                        open=float(dados["open"]),
                        high=float(dados["high"]),
                        low=float(dados["low"]),
                        close=float(dados["close"]),
                        volume=float(dados["volume"]),
                        quote_volume=float(dados["quote_volume"]),
                        trades=int(dados["trades"]),
                        # A vela so e fato consumado depois do seu fechamento.
                        closed=close_time <= now_ms,
                        origin="rest_backfill",
                    )
                )
            )
        return velas
