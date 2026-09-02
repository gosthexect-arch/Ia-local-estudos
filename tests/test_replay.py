"""Testes do replay: ordem, continuidade e reprodutibilidade."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cripto_monitor import replay as replay_mod
from cripto_monitor.storage import Storage
from fakes import PASSO_5M, candle, grade


class TestParseInstante(unittest.TestCase):
    def test_epoch_em_ms(self) -> None:
        self.assertEqual(replay_mod.parse_instante("1735689600000"), 1_735_689_600_000)

    def test_iso_sem_fuso_e_tratado_como_utc(self) -> None:
        self.assertEqual(
            replay_mod.parse_instante("2025-01-01T00:00:00"),
            replay_mod.parse_instante("2025-01-01T00:00:00Z"),
        )

    def test_iso_com_offset(self) -> None:
        self.assertEqual(
            replay_mod.parse_instante("2025-01-01T00:00:00-03:00"),
            replay_mod.parse_instante("2025-01-01T03:00:00Z"),
        )

    def test_texto_invalido(self) -> None:
        with self.assertRaises(replay_mod.ReplayError):
            replay_mod.parse_instante("ontem de manha")


class TestReplay(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.storage = Storage(Path(self._tmp.name) / "state.sqlite")
        self.storage.connect()
        self.aberturas = grade(6, fim_ms=1_735_700_000_000)

    def tearDown(self) -> None:
        self.storage.close()
        self._tmp.cleanup()

    def _gravar(self, aberturas: list[int]) -> None:
        self.storage.save_many(candle(a, close=3500.0 + i) for i, a in enumerate(aberturas))

    def test_reentrega_em_ordem_independente_da_ordem_de_chegada(self) -> None:
        self._gravar(list(reversed(self.aberturas)))
        lidas = [v.open_time_ms for v in replay_mod.replay(self.storage, "ETHUSDT", "5m")]
        self.assertEqual(lidas, self.aberturas)

    def test_duas_execucoes_produzem_exatamente_o_mesmo_resultado(self) -> None:
        """Criterio de aceitacao: o resultado precisa ser reproduzivel por replay."""
        self._gravar(self.aberturas)
        primeira = [v.to_dict() for v in replay_mod.replay(self.storage, "ETHUSDT", "5m")]
        segunda = [v.to_dict() for v in replay_mod.replay(self.storage, "ETHUSDT", "5m")]
        self.assertEqual(primeira, segunda)
        self.assertEqual(len(primeira), len(self.aberturas))

    def test_recorte_por_intervalo(self) -> None:
        self._gravar(self.aberturas)
        recorte = list(
            replay_mod.replay(
                self.storage, "ETHUSDT", "5m",
                start_ms=self.aberturas[1], end_ms=self.aberturas[3],
            )
        )
        self.assertEqual([v.open_time_ms for v in recorte], self.aberturas[1:4])

    def test_resumo_de_historico_continuo(self) -> None:
        self._gravar(self.aberturas)
        resumo = replay_mod.resumir(self.storage, "ETHUSDT", "5m")
        self.assertTrue(resumo.continuo)
        self.assertEqual(resumo.velas, 6)
        self.assertEqual(resumo.lacunas, [])

    def test_resumo_aponta_a_lacuna_e_quantas_velas_faltam(self) -> None:
        self._gravar(self.aberturas[:2] + self.aberturas[4:])
        resumo = replay_mod.resumir(self.storage, "ETHUSDT", "5m")
        self.assertFalse(resumo.continuo)
        self.assertEqual(len(resumo.lacunas), 1)
        self.assertEqual(resumo.lacunas[0]["velas"], 2)

    def test_historico_fora_da_grade_e_denunciado(self) -> None:
        self._gravar(self.aberturas[:2])
        # grava direto no banco uma vela desalinhada, simulando corrupcao
        self.storage.conn.execute(
            "UPDATE candles SET open_time_ms = open_time_ms + 7 WHERE open_time_ms = ?",
            (self.aberturas[1],),
        )
        with self.assertRaisesRegex(replay_mod.ReplayError, "fora da grade"):
            list(replay_mod.replay(self.storage, "ETHUSDT", "5m"))

    def test_banco_vazio_resume_sem_erro(self) -> None:
        resumo = replay_mod.resumir(self.storage, "ETHUSDT", "5m")
        self.assertEqual(resumo.velas, 0)
        self.assertIsNone(resumo.primeira_utc)
        self.assertTrue(resumo.continuo)

    def test_velas_do_replay_carregam_a_janela_completa(self) -> None:
        self._gravar(self.aberturas[:1])
        vela = next(iter(replay_mod.replay(self.storage, "ETHUSDT", "5m")))
        self.assertEqual(vela.close_time_ms - vela.open_time_ms, PASSO_5M - 1)
        self.assertEqual(vela.to_dict()["state"], "closed")


if __name__ == "__main__":
    unittest.main()
