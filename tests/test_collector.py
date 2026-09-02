"""Testes do coletor continuo: idempotencia, lacunas, conexao muda e reconexao."""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import websockets

from cripto_monitor.collector import Collector
from cripto_monitor.storage import Storage
from fakes import (
    PASSO_5M,
    FakeRestServer,
    com_collector,
    config_de_teste,
    grade,
    kline_event,
    kline_row,
)


class ServidorDeStream:
    """Servidor WebSocket real: envia mensagens e fecha, ou fica em silencio."""

    def __init__(self, mensagens: list[str], *, silencioso: bool = False,
                 manter_aberto: float = 0.0) -> None:
        self.mensagens = mensagens
        self.silencioso = silencioso
        self.manter_aberto = manter_aberto
        self.conexoes = 0

    async def __aenter__(self) -> "ServidorDeStream":
        self.server = await websockets.serve(self._handler, "127.0.0.1", 0)
        self.port = next(iter(self.server.sockets)).getsockname()[1]
        return self

    async def __aexit__(self, *_: object) -> None:
        self.server.close()
        await self.server.wait_closed()

    async def _handler(self, ws: object, *_: object) -> None:
        self.conexoes += 1
        if not self.silencioso:
            for mensagem in self.mensagens:
                await ws.send(mensagem)
        if self.manter_aberto:
            await asyncio.sleep(self.manter_aberto)


class BaseColetor(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.config = config_de_teste(self.dir)
        self.storage = Storage(self.config.paths.db_path, self.config.paths.raw_dir)
        self.storage.connect()

    def tearDown(self) -> None:
        self.storage.close()
        self._tmp.cleanup()

    def coletor(self, config=None, **kwargs) -> Collector:
        kwargs.setdefault("backfill_ao_conectar", False)
        return Collector(config or self.config, self.storage, **kwargs)


class TestTratamentoDeMensagens(BaseColetor):
    async def test_vela_em_formacao_nao_e_persistida(self) -> None:
        coletor = self.coletor()
        abertura = grade(1)[0]
        await coletor.processar_mensagem(kline_event(abertura, closed=False, close=3510.0))
        self.assertEqual(self.storage.count_candles("ETHUSDT", "5m"), 0)
        self.assertEqual(coletor.stats.vela_em_formacao["state"], "forming")
        self.assertEqual(coletor.stats.velas_fechadas, 0)

    async def test_vela_fechada_e_persistida_uma_unica_vez(self) -> None:
        coletor = self.coletor()
        mensagem = kline_event(grade(1)[0], closed=True)
        await coletor.processar_mensagem(mensagem)
        await coletor.processar_mensagem(mensagem)  # reentrega apos reconexao
        self.assertEqual(self.storage.count_candles("ETHUSDT", "5m"), 1)
        self.assertEqual(coletor.stats.velas_fechadas, 2)
        self.assertEqual(coletor.stats.velas_novas, 1)

    async def test_callback_so_dispara_para_vela_inedita(self) -> None:
        vistas: list[str] = []
        coletor = self.coletor(on_closed_candle=lambda vela: vistas.append(vela.open_time_utc))
        mensagem = kline_event(grade(1)[0], closed=True)
        await coletor.processar_mensagem(mensagem)
        await coletor.processar_mensagem(mensagem)
        self.assertEqual(len(vistas), 1, "alerta duplicado apos reentrega")

    async def test_callback_assincrono_e_aguardado(self) -> None:
        vistas: list[str] = []

        async def registrar(vela) -> None:
            await asyncio.sleep(0)
            vistas.append(vela.open_time_utc)

        coletor = self.coletor(on_closed_candle=registrar)
        await coletor.processar_mensagem(kline_event(grade(1)[0], closed=True))
        self.assertEqual(len(vistas), 1)

    async def test_mensagem_corrompida_nao_derruba_a_coleta(self) -> None:
        coletor = self.coletor()
        await coletor.processar_mensagem("{isso nao e json")
        await coletor.processar_mensagem(kline_event(grade(1)[0], closed=True))
        self.assertEqual(coletor.stats.descartadas, 1)
        self.assertEqual(self.storage.count_candles("ETHUSDT", "5m"), 1)
        self.assertEqual(self.storage.count_events("invalid_message"), 1)

    async def test_vela_impossivel_e_descartada(self) -> None:
        coletor = self.coletor()
        # abertura fora da grade de 5m: a exchange nunca produz isso
        await coletor.processar_mensagem(kline_event(grade(1)[0] + 1234, closed=True))
        self.assertEqual(coletor.stats.descartadas, 1)
        self.assertEqual(self.storage.count_candles("ETHUSDT", "5m"), 0)

    async def test_mensagem_sem_vela_e_ignorada_sem_erro(self) -> None:
        coletor = self.coletor()
        await coletor.processar_mensagem('{"result": null, "id": 1}')
        self.assertEqual(coletor.stats.descartadas, 0)
        self.assertEqual(coletor.stats.velas_fechadas, 0)

    async def test_simbolo_de_outro_par_e_ignorado(self) -> None:
        coletor = self.coletor()
        await coletor.processar_mensagem(kline_event(grade(1)[0], closed=True, symbol="BTCUSDT"))
        self.assertEqual(self.storage.count_candles("BTCUSDT", "5m"), 0)
        self.assertEqual(coletor.stats.velas_fechadas, 0)

    async def test_trilha_bruta_guarda_a_mensagem_original(self) -> None:
        coletor = self.coletor()
        mensagem = kline_event(grade(1)[0], closed=True)
        await coletor.processar_mensagem(mensagem)
        self.storage.close()
        arquivos = list((self.dir / "raw" / "binance" / "ETHUSDT").glob("*.jsonl.gz"))
        self.assertEqual(len(arquivos), 1)


class TestLacunasEBackfill(BaseColetor):
    async def test_lacuna_dispara_recuperacao_por_rest(self) -> None:
        aberturas = grade(4)
        faltante = aberturas[2]
        with FakeRestServer([kline_row(a) for a in aberturas[:3]]) as rest:
            config = config_de_teste(self.dir, rest_port=rest.port)
            coletor = self.coletor(config)
            for abertura in aberturas[:2]:
                await coletor.processar_mensagem(kline_event(abertura, closed=True))
            # a vela `faltante` nunca chega pelo stream; a seguinte denuncia o buraco
            await coletor.processar_mensagem(kline_event(aberturas[3], closed=True))

        self.assertEqual(coletor.stats.lacunas_detectadas, 1)
        self.assertEqual(self.storage.count_events("gap"), 1)
        guardadas = [v.open_time_ms for v in self.storage.iter_candles("ETHUSDT", "5m")]
        self.assertIn(faltante, guardadas, "a vela perdida deveria ter vindo do backfill")
        self.assertEqual(guardadas, sorted(aberturas))

    async def test_primeira_vela_do_banco_nao_e_lacuna(self) -> None:
        coletor = self.coletor()
        await coletor.processar_mensagem(kline_event(grade(1)[0], closed=True))
        self.assertEqual(coletor.stats.lacunas_detectadas, 0)
        self.assertEqual(self.storage.count_events("gap"), 0)

    async def test_backfill_pede_a_partir_da_ultima_vela_gravada(self) -> None:
        aberturas = grade(3)
        with FakeRestServer([kline_row(a) for a in aberturas]) as rest:
            config = config_de_teste(self.dir, rest_port=rest.port)
            coletor = self.coletor(config)
            self.assertEqual(await coletor.recuperar_historico(), 3)
            self.assertEqual(await coletor.recuperar_historico(), 0)  # idempotente
            ultima = self.storage.last_candle("ETHUSDT", "5m")
            assert ultima is not None
            self.assertIn(
                f"startTime={ultima.open_time_ms + PASSO_5M}", rest.chamadas[-1]
            )

    async def test_backfill_ignora_a_vela_ainda_em_formacao(self) -> None:
        import time

        atual = grade(1)[0] + PASSO_5M  # vela corrente, ainda aberta
        linhas = [kline_row(grade(1)[0]), kline_row(atual)]
        self.assertGreater(atual + PASSO_5M - 1, int(time.time() * 1000))
        with FakeRestServer(linhas) as rest:
            coletor = self.coletor(config_de_teste(self.dir, rest_port=rest.port))
            self.assertEqual(await coletor.recuperar_historico(), 1)
        self.assertEqual(self.storage.count_candles("ETHUSDT", "5m"), 1)

    async def test_rest_fora_do_ar_nao_interrompe_a_coleta(self) -> None:
        coletor = self.coletor()  # rest_base aponta para porta morta
        self.assertEqual(await coletor.recuperar_historico(), 0)
        self.assertEqual(self.storage.count_events("backfill_error"), 1)


class TestConexao(BaseColetor):
    async def test_recebe_velas_pelo_stream_real(self) -> None:
        aberturas = grade(2)
        mensagens = [kline_event(a, closed=True) for a in aberturas]
        async with ServidorDeStream(mensagens, manter_aberto=0.4) as servidor:
            config = config_de_teste(self.dir, ws_port=servidor.port, stale_after_s=0.25)
            coletor = self.coletor(config)
            await coletor.run(max_conexoes=1)
        self.assertEqual(coletor.stats.velas_novas, 2)
        self.assertEqual([v.open_time_ms for v in self.storage.iter_candles("ETHUSDT", "5m")],
                         aberturas)

    async def test_conexao_muda_e_detectada_e_derruba_o_socket(self) -> None:
        """Conexao aberta, porem sem mensagens: o caso que a spec exige tratar."""
        async with ServidorDeStream([], silencioso=True, manter_aberto=5) as servidor:
            config = config_de_teste(self.dir, ws_port=servidor.port, stale_after_s=0.2)
            coletor = self.coletor(config)
            await coletor.run(max_conexoes=1)
        self.assertEqual(coletor.stats.conexoes_mortas, 1)
        self.assertEqual(self.storage.count_events("stale_connection"), 1)

    async def test_queda_do_servidor_gera_reconexao_registrada(self) -> None:
        aberturas = grade(2)
        async with ServidorDeStream([kline_event(aberturas[0], closed=True)]) as servidor:
            config = config_de_teste(self.dir, ws_port=servidor.port, stale_after_s=0.2)
            coletor = self.coletor(config)
            await coletor.run(max_conexoes=2)
        self.assertEqual(coletor.stats.conexoes, 2)
        self.assertEqual(coletor.stats.reconexoes, 1)
        self.assertEqual(self.storage.count_events("reconnect"), 1)
        self.assertEqual(servidor.conexoes, 2)

    async def test_reconexao_nao_duplica_a_vela_ja_gravada(self) -> None:
        abertura = grade(1)[0]
        async with ServidorDeStream([kline_event(abertura, closed=True)]) as servidor:
            config = config_de_teste(self.dir, ws_port=servidor.port, stale_after_s=0.2)
            coletor = self.coletor(config)
            await coletor.run(max_conexoes=3)
        self.assertGreaterEqual(coletor.stats.velas_fechadas, 2)
        self.assertEqual(coletor.stats.velas_novas, 1)
        self.assertEqual(self.storage.count_candles("ETHUSDT", "5m"), 1)

    async def test_servidor_inacessivel_registra_erro_e_tenta_de_novo(self) -> None:
        config = config_de_teste(self.dir, ws_port=9)  # porta morta
        coletor = self.coletor(config)
        tarefa = asyncio.create_task(coletor.run())
        await asyncio.sleep(0.3)
        coletor.stop()
        await asyncio.wait_for(tarefa, timeout=5)
        self.assertGreaterEqual(self.storage.count_events("connection_error"), 1)
        self.assertEqual(coletor.stats.conexoes, 0)

    async def test_parada_interrompe_o_backoff_sem_esperar(self) -> None:
        config = com_collector(config_de_teste(self.dir, ws_port=9), backoff_initial_s=30.0)
        coletor = self.coletor(config)
        tarefa = asyncio.create_task(coletor.run())
        await asyncio.sleep(0.2)
        coletor.stop()
        await asyncio.wait_for(tarefa, timeout=3)  # nao espera os 30 s do backoff

    async def test_backoff_cresce_com_teto_e_jitter(self) -> None:
        config = com_collector(
            config_de_teste(self.dir), backoff_initial_s=1.0, backoff_max_s=8.0, backoff_jitter=0.5
        )
        coletor = self.coletor(config)
        esperas = [coletor._backoff(t) for t in range(1, 7)]
        self.assertTrue(all(1.0 <= e <= 1.5 for e in esperas[:1]), esperas)
        self.assertTrue(all(8.0 <= e <= 12.0 for e in esperas[4:]), esperas)
        self.assertLessEqual(esperas[0], esperas[3])


class TestIntegridade(BaseColetor):
    async def test_historico_continuo_e_reconhecido(self) -> None:
        coletor = self.coletor()
        for abertura in grade(4):
            await coletor.processar_mensagem(kline_event(abertura, closed=True))
        relatorio = coletor.relatorio_de_integridade()
        self.assertTrue(relatorio["continuo"], relatorio)
        self.assertEqual(relatorio["velas_faltando"], 0)

    async def test_lacuna_nao_recuperada_aparece_no_relatorio(self) -> None:
        coletor = self.coletor()  # sem REST: o backfill falha e o buraco permanece
        aberturas = grade(4)
        for abertura in (aberturas[0], aberturas[1], aberturas[3]):
            await coletor.processar_mensagem(kline_event(abertura, closed=True))
        relatorio = coletor.relatorio_de_integridade()
        self.assertFalse(relatorio["continuo"])
        self.assertEqual(relatorio["velas_faltando"], 1)
        self.assertEqual(len(relatorio["lacunas"]), 1)


if __name__ == "__main__":
    unittest.main()
