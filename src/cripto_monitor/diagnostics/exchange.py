"""Diagnostico de acesso aos dados publicos de mercado.

Somente leitura: ping, hora do servidor, candles historicos e uma sondagem curta
do stream de velas. Nenhum endpoint de conta ou de ordens e tocado, e nenhuma
credencial e enviada.
"""

from __future__ import annotations

import json
import socket
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from cripto_monitor.config import AppConfig
from cripto_monitor.diagnostics.base import CheckResult, Status
from cripto_monitor.net import wsprobe
from cripto_monitor.net.http import HttpError, get_json
from cripto_monitor.term import human_ms

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


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(timespec="milliseconds")


def summarize_kline_row(row: list[Any], *, now_ms: int | None = None) -> dict[str, Any]:
    """Normaliza uma linha REST de kline. `closed` vem do relogio, nao do modelo."""
    if len(row) < 7:
        raise ValueError(f"linha de kline com formato inesperado: {len(row)} campos")
    data = dict(zip(KLINE_FIELDS, row))
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    close_time = int(data["close_time"])
    return {
        "open_time_utc": _iso(int(data["open_time"])),
        "close_time_utc": _iso(close_time),
        "open": float(data["open"]),
        "high": float(data["high"]),
        "low": float(data["low"]),
        "close": float(data["close"]),
        "volume": float(data["volume"]),
        "trades": int(data.get("trades", 0)),
        "state": "closed" if close_time <= now_ms else "forming",
    }


def summarize_kline_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Normaliza um evento `kline` do stream. `x` indica vela fechada."""
    kline = payload.get("k")
    if not isinstance(kline, dict):
        raise ValueError("evento sem o objeto 'k' de kline")
    return {
        "symbol": kline.get("s"),
        "interval": kline.get("i"),
        "open_time_utc": _iso(int(kline["t"])),
        "close_time_utc": _iso(int(kline["T"])),
        "close": float(kline["c"]),
        "volume": float(kline["v"]),
        "state": "closed" if kline.get("x") else "forming",
        "event_time_utc": _iso(int(payload["E"])) if "E" in payload else None,
    }


def check_dns(config: AppConfig) -> CheckResult:
    alvos = {
        "rest": urlparse(config.exchange.rest_base).hostname or "",
        "websocket": urlparse(config.exchange.ws_base).hostname or "",
    }
    resolvidos: dict[str, Any] = {}
    falhas: list[str] = []
    for papel, host in alvos.items():
        if not host:
            continue
        try:
            infos = socket.getaddrinfo(host, None)
            resolvidos[papel] = {"host": host, "ips": sorted({i[4][0] for i in infos})[:4]}
        except socket.gaierror as exc:
            falhas.append(f"{host} ({exc.strerror or exc})")
    if falhas:
        return CheckResult(
            name="dns",
            status=Status.FAIL,
            summary="falha de resolucao: " + ", ".join(falhas),
            details={"resolvidos": resolvidos},
            hint="Sem DNS nao ha coletor. Verifique rede, DNS ou bloqueio regional.",
        )
    return CheckResult(
        name="dns",
        status=Status.OK,
        summary=", ".join(f"{p}: {d['host']}" for p, d in resolvidos.items()),
        details=resolvidos,
    )


def check_rest_ping(config: AppConfig) -> CheckResult:
    url = f"{config.exchange.rest_base}/api/v3/ping"
    try:
        _, resp = get_json(url, timeout=config.network.connect_timeout_s)
    except HttpError as exc:
        return CheckResult(
            name="rest_ping",
            status=Status.FAIL,
            summary=str(exc),
            details={"url": url, "status": exc.status},
            hint=(
                "HTTP 451/403 costuma indicar bloqueio regional da Binance; nesse caso "
                "avaliar Coinbase ou Kraken como fonte inicial."
            ),
        )
    return CheckResult(
        name="rest_ping",
        status=Status.OK,
        summary=f"{config.exchange.rest_base} respondeu em {human_ms(resp.elapsed_ms)}",
        details={"url": url, "latencia_ms": round(resp.elapsed_ms, 1)},
    )


def check_clock_drift(config: AppConfig) -> CheckResult:
    """Compara o relogio local com o da exchange (requisito: tudo em UTC)."""
    url = f"{config.exchange.rest_base}/api/v3/time"
    try:
        antes = time.time() * 1000
        payload, resp = get_json(url, timeout=config.network.connect_timeout_s)
        depois = time.time() * 1000
    except HttpError as exc:
        return CheckResult("clock_drift", Status.SKIP, f"nao verificado: {exc}", {"url": url})
    servidor = float(payload["serverTime"])
    local = (antes + depois) / 2
    drift = local - servidor
    limite = config.network.max_clock_drift_ms
    dentro = abs(drift) <= limite
    return CheckResult(
        name="clock_drift",
        status=Status.OK if dentro else Status.WARN,
        summary=f"deriva de {drift:+.0f} ms contra a exchange (limite {limite:.0f} ms)",
        details={
            "drift_ms": round(drift, 1),
            "rtt_ms": round(depois - antes, 1),
            "servidor_utc": _iso(int(servidor)),
            "latencia_http_ms": round(resp.elapsed_ms, 1),
        },
        hint=None if dentro else "Sincronize o relogio (NTP) antes de confiar nos fechamentos.",
    )


def check_klines(config: AppConfig) -> CheckResult:
    """Baixa poucas velas historicas e valida forma, ordem e estado."""
    ex = config.exchange
    url = (
        f"{ex.rest_base}/api/v3/klines?symbol={ex.symbol}"
        f"&interval={ex.primary_timeframe}&limit=3"
    )
    try:
        rows, resp = get_json(url, timeout=config.network.connect_timeout_s)
    except HttpError as exc:
        return CheckResult("klines_rest", Status.FAIL, str(exc), {"url": url})
    if not isinstance(rows, list) or not rows:
        return CheckResult("klines_rest", Status.FAIL, "resposta vazia ou inesperada", {"url": url})
    velas = [summarize_kline_row(row) for row in rows]
    fechadas = [v for v in velas if v["state"] == "closed"]
    ordenadas = all(
        velas[i]["open_time_utc"] < velas[i + 1]["open_time_utc"] for i in range(len(velas) - 1)
    )
    ultima = velas[-1]
    status = Status.OK if fechadas and ordenadas else Status.WARN
    return CheckResult(
        name="klines_rest",
        status=status,
        summary=(
            f"{len(velas)} velas {ex.primary_timeframe} de {ex.symbol}; "
            f"ultima {ultima['state']} @ {ultima['close']}"
        ),
        details={
            "url": url,
            "latencia_ms": round(resp.elapsed_ms, 1),
            "em_ordem": ordenadas,
            "fechadas": len(fechadas),
            "velas": velas,
        },
        hint=None if status is Status.OK else "Velas fora de ordem ou nenhuma fechada na amostra.",
    )


def check_websocket(config: AppConfig) -> CheckResult:
    """Sondagem curta do stream de velas: prova conectividade e formato do evento."""
    url = config.exchange.ws_stream_url
    try:
        resultado = wsprobe.probe(
            url,
            messages=config.network.probe_messages,
            timeout=config.network.probe_timeout_s,
            connect_timeout=config.network.connect_timeout_s,
        )
    except (wsprobe.WebSocketError, OSError) as exc:
        return CheckResult(
            name="websocket",
            status=Status.FAIL,
            summary=f"{type(exc).__name__}: {exc}",
            details={"url": url},
            hint="Sem stream nao ha monitoramento continuo; testar Coinbase/Kraken como fonte.",
        )
    eventos = []
    invalidos = 0
    for bruto in resultado.messages:
        try:
            eventos.append(summarize_kline_event(json.loads(bruto)))
        except (json.JSONDecodeError, ValueError, KeyError):
            invalidos += 1
    status = Status.OK if eventos and not invalidos else Status.WARN
    ultimo = eventos[-1] if eventos else None
    return CheckResult(
        name="websocket",
        status=status,
        summary=(
            f"{len(resultado.messages)} eventos em {human_ms(resultado.first_message_ms)} "
            f"(handshake {human_ms(resultado.handshake_ms)})"
            + (f"; ultima vela {ultimo['state']} @ {ultimo['close']}" if ultimo else "")
        ),
        details={
            "url": url,
            "handshake_ms": round(resultado.handshake_ms, 1),
            "primeira_mensagem_ms": round(resultado.first_message_ms, 1),
            "pings_recebidos": resultado.pings_recebidos,
            "close_code": resultado.close_code,
            "eventos_invalidos": invalidos,
            "eventos": eventos,
        },
        hint=None if not invalidos else "Alguma mensagem nao seguiu o formato de kline esperado.",
    )


def build_checks(config: AppConfig) -> list[tuple[str, object]]:
    return [
        ("dns", lambda: check_dns(config)),
        ("rest_ping", lambda: check_rest_ping(config)),
        ("clock_drift", lambda: check_clock_drift(config)),
        ("klines_rest", lambda: check_klines(config)),
        ("websocket", lambda: check_websocket(config)),
    ]
