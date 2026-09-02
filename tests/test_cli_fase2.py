"""Testes da CLI da Fase 2: collect, backfill, replay e status."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from cripto_monitor import cli
from fakes import (
    FakeRestServer,
    ServidorDeStreamEmThread,
    escrever_config,
    grade,
    kline_event,
    kline_row,
)


def executar(argv: list[str]) -> tuple[int, str, str]:
    saida, erro = io.StringIO(), io.StringIO()
    with redirect_stdout(saida), redirect_stderr(erro):
        codigo = cli.main(argv)
    return codigo, saida.getvalue(), erro.getvalue()


class TestFluxoCompleto(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.data_dir = self.dir / "dados"
        self.config_path = self.dir / "monitor.toml"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _config(self, *, rest_port: int = 9, ws_port: int = 9) -> list[str]:
        escrever_config(
            self.config_path, data_dir=self.data_dir, rest_port=rest_port, ws_port=ws_port
        )
        return ["-c", str(self.config_path)]

    def test_status_sem_banco_orienta_o_usuario(self) -> None:
        codigo, _, erro = executar([*self._config(), "status"])
        self.assertEqual(codigo, 1)
        self.assertIn("nenhum banco", erro)

    def test_backfill_grava_e_status_confirma(self) -> None:
        aberturas = grade(4)
        with FakeRestServer([kline_row(a) for a in aberturas]) as rest:
            base = self._config(rest_port=rest.port)
            codigo, saida, _ = executar([*base, "backfill", "--json"])
        self.assertEqual(codigo, 0)
        resultado = json.loads(saida)
        self.assertEqual(resultado["novas"], 4)

        codigo, saida, _ = executar([*base, "status", "--json"])
        self.assertEqual(codigo, 0)
        estado = json.loads(saida)
        self.assertEqual(estado["estatisticas"]["candles"], 4)
        self.assertTrue(estado["continuidade"]["continuo"])

    def test_backfill_repetido_nao_duplica(self) -> None:
        aberturas = grade(3)
        with FakeRestServer([kline_row(a) for a in aberturas]) as rest:
            base = self._config(rest_port=rest.port)
            executar([*base, "backfill", "--json"])
            codigo, saida, _ = executar([*base, "backfill", "--json"])
        self.assertEqual(json.loads(saida)["novas"], 0)
        self.assertEqual(json.loads(saida)["estatisticas"]["candles"], 3)
        self.assertEqual(codigo, 0, "historico ja completo nao e falha")

    def test_replay_reentrega_o_que_foi_gravado(self) -> None:
        aberturas = grade(3)
        with FakeRestServer([kline_row(a) for a in aberturas]) as rest:
            base = self._config(rest_port=rest.port)
            executar([*base, "backfill"])
            codigo, saida, _ = executar([*base, "replay"])
        linhas = [json.loads(linha) for linha in saida.strip().splitlines()]
        velas, resumo = linhas[:-1], linhas[-1]["resumo"]
        self.assertEqual(codigo, 0)
        self.assertEqual([v["open_time_ms"] for v in velas], aberturas)
        self.assertTrue(all(v["state"] == "closed" for v in velas))
        self.assertTrue(resumo["continuo"])

    def test_replay_com_recorte_e_resumo(self) -> None:
        aberturas = grade(4)
        with FakeRestServer([kline_row(a) for a in aberturas]) as rest:
            base = self._config(rest_port=rest.port)
            executar([*base, "backfill"])
            codigo, saida, _ = executar(
                [*base, "replay", "--from", str(aberturas[2]), "--resumo"]
            )
        resumo = json.loads(saida)["resumo"]
        self.assertEqual(codigo, 0)
        self.assertEqual(resumo["velas"], 2)

    def test_replay_com_instante_invalido(self) -> None:
        codigo, _, erro = executar([*self._config(), "replay", "--from", "amanha"])
        self.assertEqual(codigo, 2)
        self.assertIn("instante invalido", erro)

    def test_collect_grava_velas_do_stream_e_encerra_no_prazo(self) -> None:
        aberturas = grade(2)
        mensagens = [kline_event(a, closed=True) for a in aberturas]
        with ServidorDeStreamEmThread(mensagens, manter_aberto=3) as stream:
            base = self._config(ws_port=stream.port)
            codigo, saida, _ = executar(
                [*base, "collect", "--duracao", "1.5", "--sem-backfill", "--log-json"]
            )
        eventos = [json.loads(linha) for linha in saida.strip().splitlines()]
        tipos = {e["event"] for e in eventos}
        self.assertEqual(codigo, 0)
        self.assertIn("conectado", tipos)
        self.assertIn("vela_fechada", tipos)
        encerramento = next(e for e in eventos if e["event"] == "encerrado")
        self.assertEqual(encerramento["velas_novas"], 2)

        codigo, saida, _ = executar([*base, "status", "--json"])
        self.assertEqual(json.loads(saida)["estatisticas"]["candles"], 2)

    def test_collect_sem_servidor_registra_erro_e_para(self) -> None:
        base = self._config(ws_port=9)
        codigo, saida, _ = executar(
            [*base, "collect", "--duracao", "0.5", "--sem-backfill", "--log-json"]
        )
        eventos = [json.loads(linha) for linha in saida.strip().splitlines()]
        self.assertIn("falha_conexao", {e["event"] for e in eventos})
        self.assertEqual(codigo, 1)

    def test_collect_avisa_quando_a_biblioteca_falta(self) -> None:
        import cripto_monitor.collector as collector_mod

        original = collector_mod.carregar_websockets
        collector_mod.carregar_websockets = lambda: (_ for _ in ()).throw(
            RuntimeError(collector_mod.MENSAGEM_SEM_WEBSOCKETS)
        )
        try:
            codigo, _, erro = executar([*self._config(), "collect", "--duracao", "0.2"])
        finally:
            collector_mod.carregar_websockets = original
        self.assertEqual(codigo, 3)
        self.assertIn("pacman -S python-websockets", erro)


if __name__ == "__main__":
    unittest.main()
