"""Dublês compartilhados pelos testes: exchange REST e stream de velas."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from cripto_monitor.candles import Candle, align_open_time, timeframe_ms
from cripto_monitor.config import AppConfig, CollectorConfig, ExchangeConfig, NetworkConfig, PathsConfig

PASSO_5M = timeframe_ms("5m")


def kline_row(open_ms: int, *, close: float = 3500.0, timeframe: str = "5m") -> list[Any]:
    """Linha REST de kline, no formato exato da Binance."""
    passo = timeframe_ms(timeframe)
    return [
        open_ms, f"{close - 5:.2f}", f"{close + 10:.2f}", f"{close - 12:.2f}", f"{close:.2f}",
        "120.5", open_ms + passo - 1, "422000.0", 987, "60.2", "211000.0", "0",
    ]


def kline_event(
    open_ms: int, *, closed: bool, close: float = 3500.0, symbol: str = "ETHUSDT",
    timeframe: str = "5m",
) -> str:
    """Mensagem do stream de velas, no formato exato da Binance."""
    passo = timeframe_ms(timeframe)
    return json.dumps({
        "e": "kline",
        "E": open_ms + passo - 1,
        "s": symbol,
        "k": {
            "t": open_ms, "T": open_ms + passo - 1, "s": symbol, "i": timeframe,
            "o": f"{close - 5:.2f}", "h": f"{close + 10:.2f}", "l": f"{close - 12:.2f}",
            "c": f"{close:.2f}", "v": "120.5", "q": "422000.0", "n": 987, "x": closed,
        },
    })


def candle(open_ms: int, *, closed: bool = True, close: float = 3500.0) -> Candle:
    return Candle(
        source="binance", symbol="ETHUSDT", timeframe="5m",
        open_time_ms=open_ms, close_time_ms=open_ms + PASSO_5M - 1,
        open=close - 5, high=close + 10, low=close - 12, close=close,
        volume=120.5, quote_volume=422000.0, trades=987, closed=closed,
    )


def grade(quantidade: int, *, fim_ms: int | None = None) -> list[int]:
    """Aberturas consecutivas de 5m terminando em `fim_ms` (padrao: agora)."""
    import time

    fim = align_open_time(fim_ms or int(time.time() * 1000), "5m") - PASSO_5M
    return [fim - i * PASSO_5M for i in range(quantidade - 1, -1, -1)]


class _Handler(BaseHTTPRequestHandler):
    klines: list[list[Any]] = []
    chamadas: list[str] = []

    def log_message(self, *_: object) -> None:
        pass

    def do_GET(self) -> None:
        type(self).chamadas.append(self.path)
        rota = self.path.split("?")[0]
        if rota == "/api/v3/klines":
            corpo = json.dumps(type(self).klines).encode()
            self.send_response(200)
        else:
            corpo = json.dumps({"erro": "rota desconhecida"}).encode()
            self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)


class FakeRestServer:
    """Serve /api/v3/klines com uma lista de velas controlada pelo teste."""

    def __init__(self, klines: list[list[Any]] | None = None) -> None:
        _Handler.klines = klines or []
        _Handler.chamadas = []

    def __enter__(self) -> "FakeRestServer":
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={'poll_interval': 0.02}, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def definir(self, klines: list[list[Any]]) -> None:
        _Handler.klines = klines

    @property
    def chamadas(self) -> list[str]:
        return list(_Handler.chamadas)


def config_de_teste(
    tmp: Path, *, rest_port: int = 9, ws_port: int = 9, **collector: Any
) -> AppConfig:
    """Configuracao apontada para servidores locais, com tempos curtos."""
    padrao = dict(
        stale_after_s=0.5, ping_interval_s=5.0, ping_timeout_s=5.0,
        backoff_initial_s=0.05, backoff_max_s=0.2, backoff_jitter=0.0,
        backfill_limit=10, store_raw=True,
    )
    padrao.update(collector)
    return AppConfig(
        exchange=ExchangeConfig(
            rest_base=f"http://127.0.0.1:{rest_port}", ws_base=f"ws://127.0.0.1:{ws_port}"
        ),
        paths=PathsConfig(data_dir=tmp),
        network=NetworkConfig(connect_timeout_s=3.0),
        collector=CollectorConfig(**padrao),
    )


def com_collector(config: AppConfig, **campos: Any) -> AppConfig:
    return replace(config, collector=replace(config.collector, **campos))


class ServidorDeStreamEmThread:
    """Stream WebSocket rodando em outra thread, para testes sincronos da CLI."""

    def __init__(self, mensagens: list[str], *, manter_aberto: float = 2.0) -> None:
        self.mensagens = mensagens
        self.manter_aberto = manter_aberto
        self.port = 0
        self.conexoes = 0
        self._pronto = threading.Event()
        self._loop: Any = None

    def __enter__(self) -> "ServidorDeStreamEmThread":
        self._thread = threading.Thread(target=self._executar, daemon=True)
        self._thread.start()
        if not self._pronto.wait(timeout=10):
            raise RuntimeError("servidor de stream nao subiu")
        return self

    def __exit__(self, *_: object) -> None:
        import asyncio

        if self._loop is not None:
            futuro = asyncio.run_coroutine_threadsafe(self._encerrar(), self._loop)
            try:
                futuro.result(timeout=5)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    async def _encerrar(self) -> None:
        self._servidor.close()
        await self._servidor.wait_closed()

    def _executar(self) -> None:
        import asyncio

        import websockets

        async def handler(ws: Any, *_: Any) -> None:
            self.conexoes += 1
            try:
                for mensagem in self.mensagens:
                    await ws.send(mensagem)
                await asyncio.sleep(self.manter_aberto)
            except asyncio.CancelledError:
                pass

        async def principal() -> None:
            self._servidor = await websockets.serve(handler, "127.0.0.1", 0)
            self.port = next(iter(self._servidor.sockets)).getsockname()[1]
            self._pronto.set()

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(principal())
            self._loop.run_forever()
        finally:
            self._loop.close()


def escrever_config(
    caminho: Path, *, data_dir: Path, rest_port: int, ws_port: int, extras: str = ""
) -> Path:
    caminho.write_text(
        f'[exchange]\nrest_base = "http://127.0.0.1:{rest_port}"\n'
        f'ws_base = "ws://127.0.0.1:{ws_port}"\n\n'
        f'[paths]\ndata_dir = "{data_dir}"\n\n'
        f"[collector]\nstale_after_s = 0.3\nbackoff_initial_s = 0.05\n"
        f"backoff_max_s = 0.2\nbackoff_jitter = 0.0\nbackfill_limit = 10\n{extras}",
        encoding="utf-8",
    )
    return caminho
