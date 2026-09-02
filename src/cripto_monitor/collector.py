"""Coletor continuo de velas.

Requisitos que este modulo existe para cumprir:

- permanecer conectado, com heartbeat e reconexao com backoff e jitter;
- detectar conexao aparentemente viva, porem sem mensagens recentes;
- recuperar por REST o periodo perdido e preencher lacunas antes de seguir;
- nunca deixar que uma falha do modelo, do disco ou de uma mensagem malformada
  interrompa a coleta.

O coletor nao interpreta o mercado e nao emite alerta: ele so produz fatos.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from cripto_monitor.candles import (
    Candle,
    CandleError,
    align_open_time,
    expected_open_times,
    find_gaps,
    iso,
    timeframe_ms,
)
from cripto_monitor.config import AppConfig
from cripto_monitor.exchanges import ExchangeAdapter, get_adapter
from cripto_monitor.net.http import HttpError, get_json
from cripto_monitor.storage import Storage, agora_ms

RetornoDeVela = Callable[[Candle], Awaitable[None] | None]

MENSAGEM_SEM_WEBSOCKETS = (
    "a biblioteca 'websockets' nao esta instalada.\n"
    "  No Arch Linux:  sudo pacman -S python-websockets\n"
    "  Em um venv:     pip install websockets\n"
    "O comando 'doctor' continua funcionando sem ela."
)


def carregar_websockets() -> Any:
    try:
        import websockets  # noqa: PLC0415 - dependencia opcional, so a Fase 2 exige
    except ImportError as exc:
        raise RuntimeError(MENSAGEM_SEM_WEBSOCKETS) from exc
    return websockets


@dataclass(slots=True)
class CollectorStats:
    conexoes: int = 0
    reconexoes: int = 0
    mensagens: int = 0
    velas_fechadas: int = 0
    velas_novas: int = 0
    descartadas: int = 0
    lacunas_detectadas: int = 0
    velas_recuperadas: int = 0
    conexoes_mortas: int = 0
    ultima_mensagem_ms: int | None = None
    vela_em_formacao: dict[str, Any] | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conexoes": self.conexoes,
            "reconexoes": self.reconexoes,
            "mensagens": self.mensagens,
            "velas_fechadas": self.velas_fechadas,
            "velas_novas": self.velas_novas,
            "descartadas": self.descartadas,
            "lacunas_detectadas": self.lacunas_detectadas,
            "velas_recuperadas": self.velas_recuperadas,
            "conexoes_mortas": self.conexoes_mortas,
            "ultima_mensagem_utc": (
                iso(self.ultima_mensagem_ms) if self.ultima_mensagem_ms else None
            ),
            "vela_em_formacao": self.vela_em_formacao,
        }


class Collector:
    """Mantem o fluxo de velas vivo e o armazenamento coerente."""

    def __init__(
        self,
        config: AppConfig,
        storage: Storage,
        *,
        adapter: ExchangeAdapter | None = None,
        on_closed_candle: RetornoDeVela | None = None,
        log: Callable[[str, dict[str, Any]], None] | None = None,
        backfill_ao_conectar: bool = True,
    ) -> None:
        self.config = config
        self.storage = storage
        self.adapter = adapter or get_adapter(
            config.exchange.name, config.exchange.rest_base, config.exchange.ws_base
        )
        self.on_closed_candle = on_closed_candle
        self.backfill_ao_conectar = backfill_ao_conectar
        self.stats = CollectorStats()
        self.symbol = config.exchange.symbol
        self.timeframe = config.exchange.primary_timeframe
        self._log = log or (lambda evento, dados: None)
        self._parar = asyncio.Event()

    # ------------------------------------------------------------------ ciclo

    def stop(self) -> None:
        self._parar.set()

    async def run(self, *, max_conexoes: int | None = None) -> CollectorStats:
        """Laco principal: conecta, coleta e reconecta ate receber ordem de parar."""
        websockets = carregar_websockets()
        url = self.adapter.stream_url(self.symbol, self.timeframe)
        tentativa = 0

        while not self._parar.is_set():
            if max_conexoes is not None and self.stats.conexoes >= max_conexoes:
                break
            try:
                async with websockets.connect(
                    url,
                    open_timeout=self.config.network.connect_timeout_s,
                    ping_interval=self.config.collector.ping_interval_s,
                    ping_timeout=self.config.collector.ping_timeout_s,
                    close_timeout=5,
                    max_queue=256,
                ) as ws:
                    self.stats.conexoes += 1
                    if self.stats.conexoes > 1:
                        self.stats.reconexoes += 1
                        self.storage.record_event(
                            "reconnect", symbol=self.symbol, timeframe=self.timeframe,
                            detail={"tentativa": tentativa, "url": url},
                        )
                    self._log("conectado", {"url": url, "conexao": self.stats.conexoes})
                    inscricao = self.adapter.subscribe_payload(self.symbol, self.timeframe)
                    if inscricao is not None:
                        await ws.send(json.dumps(inscricao))

                    # Antes de aceitar dados novos, fecha o buraco do periodo offline.
                    if self.backfill_ao_conectar:
                        await self.recuperar_historico()
                    tentativa = 0
                    await self._consumir(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # rede, protocolo, DNS: nada pode matar o coletor
                self._registrar_falha(exc, tentativa)
                tentativa += 1
                if self._parar.is_set():
                    break
                if max_conexoes is not None and self.stats.conexoes >= max_conexoes:
                    break
                await self._esperar(self._backoff(tentativa))
        return self.stats

    async def _consumir(self, ws: Any) -> None:
        """Le mensagens ate a conexao cair ou ficar silenciosa tempo demais."""
        limite = self.config.collector.stale_after_s
        while not self._parar.is_set():
            try:
                bruta = await asyncio.wait_for(ws.recv(), timeout=limite)
            except asyncio.TimeoutError:
                # Conexao aberta, porem muda: o caso que uma reconexao ingenua nao pega.
                self.stats.conexoes_mortas += 1
                self.storage.record_event(
                    "stale_connection", severity="warn", symbol=self.symbol,
                    timeframe=self.timeframe, detail={"sem_mensagens_por_s": limite},
                )
                self._log("conexao_muda", {"limite_s": limite})
                await ws.close()
                return
            await self.processar_mensagem(
                bruta if isinstance(bruta, str) else bruta.decode("utf-8", "replace")
            )

    def _registrar_falha(self, exc: BaseException, tentativa: int) -> None:
        self.storage.record_event(
            "connection_error", severity="warn", symbol=self.symbol, timeframe=self.timeframe,
            detail={"erro": f"{type(exc).__name__}: {exc}", "tentativa": tentativa},
        )
        self._log("falha_conexao", {"erro": f"{type(exc).__name__}: {exc}", "tentativa": tentativa})

    def _backoff(self, tentativa: int) -> float:
        cfg = self.config.collector
        base = min(cfg.backoff_initial_s * (2 ** max(0, tentativa - 1)), cfg.backoff_max_s)
        return base * (1 + random.random() * cfg.backoff_jitter)

    async def _esperar(self, segundos: float) -> None:
        """Espera interrompivel: uma ordem de parada nao aguarda o backoff inteiro."""
        try:
            await asyncio.wait_for(self._parar.wait(), timeout=segundos)
        except asyncio.TimeoutError:
            pass

    # -------------------------------------------------------------- mensagens

    async def processar_mensagem(self, bruta: str) -> Candle | None:
        """Trata uma mensagem do stream. Nunca levanta por dado ruim."""
        self.stats.mensagens += 1
        self.stats.ultima_mensagem_ms = agora_ms()
        if self.config.collector.store_raw:
            try:
                self.storage.append_raw(self.adapter.name, self.symbol, bruta)
            except OSError as exc:
                self._log("falha_trilha_bruta", {"erro": str(exc)})

        try:
            vela = self.adapter.parse_message(bruta)
        except (CandleError, KeyError, TypeError, ValueError) as exc:
            self.stats.descartadas += 1
            self.storage.record_event(
                "invalid_message", severity="warn", symbol=self.symbol,
                timeframe=self.timeframe, detail={"erro": str(exc), "trecho": bruta[:200]},
            )
            return None
        if vela is None:
            return None
        if vela.symbol != self.symbol or vela.timeframe != self.timeframe:
            return None

        if not vela.closed:
            self.stats.vela_em_formacao = vela.to_dict()
            return vela

        self.stats.velas_fechadas += 1
        self.stats.vela_em_formacao = None
        nova = self.storage.save_closed_candle(vela)
        self.storage.set_state("ultima_vela_fechada_ms", vela.close_time_ms)
        if not nova:
            # Reentrega apos reconexao ou replay: registrada, jamais duplicada.
            self._log("vela_repetida", {"abertura": vela.open_time_utc})
            return vela

        self.stats.velas_novas += 1
        self._log("vela_fechada", {"abertura": vela.open_time_utc, "close": vela.close})
        await self._verificar_lacuna(vela)
        if self.on_closed_candle is not None:
            resultado = self.on_closed_candle(vela)
            if asyncio.iscoroutine(resultado):
                await resultado
        return vela

    async def _verificar_lacuna(self, vela: Candle) -> None:
        """Uma vela nova cuja anterior nao existe denuncia periodo perdido."""
        passo = timeframe_ms(self.timeframe)
        anterior = vela.open_time_ms - passo
        if self.storage.get_candle((vela.source, vela.symbol, vela.timeframe, anterior)):
            return
        if self.storage.count_candles(self.symbol, self.timeframe) <= 1:
            return  # primeira vela do banco: nao ha lacuna, ha inicio
        self.stats.lacunas_detectadas += 1
        self.storage.record_event(
            "gap", severity="warn", symbol=self.symbol, timeframe=self.timeframe,
            detail={"faltando_desde_utc": iso(anterior)},
        )
        await self.recuperar_historico()

    # ------------------------------------------------------------- backfill

    async def recuperar_historico(self, *, limite: int | None = None) -> int:
        """Busca velas por REST para cobrir o que o stream nao entregou."""
        limite = limite or self.config.collector.backfill_limit
        passo = timeframe_ms(self.timeframe)
        ultima = self.storage.last_candle(self.symbol, self.timeframe)
        inicio = ultima.open_time_ms + passo if ultima else None
        url = self.adapter.backfill_url(
            self.symbol, self.timeframe, start_ms=inicio, limit=limite
        )
        try:
            payload, _ = await asyncio.to_thread(
                get_json, url, timeout=self.config.network.connect_timeout_s
            )
        except HttpError as exc:
            self.storage.record_event(
                "backfill_error", severity="warn", symbol=self.symbol,
                timeframe=self.timeframe, detail={"erro": str(exc), "url": url},
            )
            self._log("falha_backfill", {"erro": str(exc)})
            return 0

        try:
            velas = self.adapter.parse_backfill(
                payload, self.symbol, self.timeframe, now_ms=agora_ms()
            )
        except CandleError as exc:
            self.storage.record_event(
                "backfill_error", severity="warn", symbol=self.symbol,
                timeframe=self.timeframe, detail={"erro": str(exc)},
            )
            return 0

        fechadas = [v for v in velas if v.closed]
        novas = self.storage.save_many(fechadas)
        self.stats.velas_recuperadas += novas
        if novas:
            self.storage.record_event(
                "backfill", symbol=self.symbol, timeframe=self.timeframe,
                detail={"recebidas": len(fechadas), "novas": novas, "desde_utc": iso(inicio) if inicio else None},
            )
            self._log("backfill", {"novas": novas, "recebidas": len(fechadas)})
        return novas

    # ------------------------------------------------------------- integridade

    def relatorio_de_integridade(self, *, janela: int = 500) -> dict[str, Any]:
        """Lacunas remanescentes na janela recente, para o comando de saude."""
        passo = timeframe_ms(self.timeframe)
        recentes = list(
            self.storage.iter_candles(
                self.symbol, self.timeframe, limit=janela, newest_first=True
            )
        )
        aberturas = sorted(v.open_time_ms for v in recentes)
        lacunas = find_gaps(aberturas, self.timeframe)
        faltando = sum(len(expected_open_times(a, b, self.timeframe)) for a, b in lacunas)
        esperado_agora = align_open_time(int(time.time() * 1000), self.timeframe) - passo
        atraso = None
        if aberturas:
            atraso = max(0, (esperado_agora - aberturas[-1]) // passo)
        return {
            "velas_na_janela": len(aberturas),
            "lacunas": [
                {"de_utc": iso(a), "ate_utc": iso(b), "velas": len(expected_open_times(a, b, self.timeframe))}
                for a, b in lacunas
            ],
            "velas_faltando": faltando,
            "velas_atrasadas": atraso,
            # Sem vela alguma nao existe continuidade a declarar: isso e falha de coleta.
            "continuo": bool(aberturas) and not lacunas and (atraso or 0) <= 1,
        }
