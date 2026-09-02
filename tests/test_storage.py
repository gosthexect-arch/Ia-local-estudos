"""Testes da persistencia: idempotencia, ordem e trilha bruta."""

from __future__ import annotations

import gzip
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from cripto_monitor.candles import CandleError, align_open_time, find_gaps, merge, timeframe_ms
from cripto_monitor.candles import expected_open_times, validate
from cripto_monitor.storage import Storage
from fakes import PASSO_5M, candle, grade


class TestAritmeticaDeVelas(unittest.TestCase):
    def test_timeframes_conhecidos(self) -> None:
        self.assertEqual(timeframe_ms("5m"), 300_000)
        self.assertEqual(timeframe_ms("1h"), 3_600_000)
        with self.assertRaises(CandleError):
            timeframe_ms("7m")

    def test_alinhamento_na_grade(self) -> None:
        self.assertEqual(align_open_time(1_735_689_612_345, "5m"), 1_735_689_600_000)
        self.assertEqual(align_open_time(1_735_689_600_000, "5m"), 1_735_689_600_000)

    def test_lacunas_entre_aberturas(self) -> None:
        base = 1_735_689_600_000
        aberturas = [base, base + PASSO_5M, base + 4 * PASSO_5M]
        self.assertEqual(find_gaps(aberturas, "5m"), [(base + 2 * PASSO_5M, base + 3 * PASSO_5M)])
        self.assertEqual(find_gaps([base, base + PASSO_5M], "5m"), [])

    def test_aberturas_esperadas_no_intervalo(self) -> None:
        base = 1_735_689_600_000
        self.assertEqual(
            expected_open_times(base, base + 2 * PASSO_5M, "5m"),
            [base, base + PASSO_5M, base + 2 * PASSO_5M],
        )
        self.assertEqual(expected_open_times(base, base - 1, "5m"), [])

    def test_velas_impossiveis_sao_recusadas(self) -> None:
        base = 1_735_689_600_000
        casos = {
            "desalinhada": replace(candle(base), open_time_ms=base + 1),
            "duracao errada": replace(candle(base), close_time_ms=base + 1000),
            "preco zero": replace(candle(base), close=0.0),
            "maxima abaixo do fechamento": replace(candle(base), high=1.0),
            "volume negativo": replace(candle(base), volume=-1.0),
        }
        for nome, vela in casos.items():
            with self.subTest(nome), self.assertRaises(CandleError):
                validate(vela)

    def test_vela_fechada_nao_regride_para_em_formacao(self) -> None:
        base = 1_735_689_600_000
        fechada = candle(base, close=3500.0)
        reentrega = candle(base, closed=False, close=3400.0)
        self.assertIs(merge(fechada, reentrega), fechada)

    def test_mesclar_chaves_diferentes_e_erro(self) -> None:
        with self.assertRaises(CandleError):
            merge(candle(1_735_689_600_000), candle(1_735_689_900_000))


class TestStorage(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.storage = Storage(self.dir / "state.sqlite", self.dir / "raw")
        self.storage.connect()

    def tearDown(self) -> None:
        self.storage.close()
        self._tmp.cleanup()

    def test_primeira_gravacao_e_inedita_e_a_segunda_nao(self) -> None:
        vela = candle(1_735_689_600_000)
        self.assertTrue(self.storage.save_closed_candle(vela))
        self.assertFalse(self.storage.save_closed_candle(vela))
        self.assertEqual(self.storage.count_candles("ETHUSDT", "5m"), 1)

    def test_reentrega_apos_reconexao_atualiza_sem_duplicar(self) -> None:
        base = 1_735_689_600_000
        self.storage.save_closed_candle(candle(base, close=3500.0))
        corrigida = replace(candle(base, close=3501.0), origin="rest_backfill")
        self.assertFalse(self.storage.save_closed_candle(corrigida))
        guardada = self.storage.get_candle(corrigida.key)
        assert guardada is not None
        self.assertEqual(guardada.close, 3501.0)
        self.assertEqual(guardada.origin, "rest_backfill")
        self.assertEqual(self.storage.count_candles("ETHUSDT", "5m"), 1)

    def test_vela_em_formacao_nunca_e_persistida(self) -> None:
        with self.assertRaisesRegex(ValueError, "formacao"):
            self.storage.save_closed_candle(candle(1_735_689_600_000, closed=False))

    def test_leitura_em_ordem_cronologica(self) -> None:
        aberturas = grade(5, fim_ms=1_735_700_000_000)
        for abertura in reversed(aberturas):  # gravadas fora de ordem de proposito
            self.storage.save_closed_candle(candle(abertura))
        lidas = [v.open_time_ms for v in self.storage.iter_candles("ETHUSDT", "5m")]
        self.assertEqual(lidas, sorted(aberturas))
        ultima = self.storage.last_candle("ETHUSDT", "5m")
        assert ultima is not None
        self.assertEqual(ultima.open_time_ms, max(aberturas))

    def test_filtros_de_intervalo_e_limite(self) -> None:
        aberturas = grade(6, fim_ms=1_735_700_000_000)
        self.storage.save_many(candle(a) for a in aberturas)
        meio = list(
            self.storage.iter_candles(
                "ETHUSDT", "5m", start_ms=aberturas[2], end_ms=aberturas[4]
            )
        )
        self.assertEqual([v.open_time_ms for v in meio], aberturas[2:5])
        self.assertEqual(len(list(self.storage.iter_candles("ETHUSDT", "5m", limit=2))), 2)

    def test_lote_conta_apenas_as_ineditas(self) -> None:
        aberturas = grade(3, fim_ms=1_735_700_000_000)
        self.assertEqual(self.storage.save_many(candle(a) for a in aberturas), 3)
        self.assertEqual(self.storage.save_many(candle(a) for a in aberturas), 0)

    def test_diario_de_eventos(self) -> None:
        self.storage.record_event("reconnect", detail={"tentativa": 2})
        self.storage.record_event("gap", severity="warn", detail={"faltando": 3})
        self.assertEqual(self.storage.count_events("reconnect"), 1)
        eventos = self.storage.recent_events(limit=5)
        self.assertEqual(eventos[0]["kind"], "gap")
        self.assertEqual(eventos[0]["detail"], {"faltando": 3})

    def test_estado_sobrevive_a_reabertura(self) -> None:
        self.storage.set_state("ultima_vela_fechada_ms", 1_735_689_899_999)
        self.storage.close()
        with Storage(self.dir / "state.sqlite") as outra:
            self.assertEqual(outra.get_state("ultima_vela_fechada_ms"), 1_735_689_899_999)
            self.assertIsNone(outra.get_state("inexistente"))

    def test_historico_sobrevive_a_reabertura(self) -> None:
        self.storage.save_closed_candle(candle(1_735_689_600_000))
        self.storage.close()
        with Storage(self.dir / "state.sqlite") as outra:
            self.assertEqual(outra.count_candles("ETHUSDT", "5m"), 1)

    def test_trilha_bruta_em_jsonl_comprimido(self) -> None:
        self.storage.append_raw("binance", "ethusdt", '{"e":"kline"}')
        self.storage.append_raw("binance", "ethusdt", '{"e":"kline","n":2}')
        self.storage.close()
        arquivos = list((self.dir / "raw" / "binance" / "ETHUSDT").glob("*.jsonl.gz"))
        self.assertEqual(len(arquivos), 1)
        with gzip.open(arquivos[0], "rt", encoding="utf-8") as handle:
            linhas = [json.loads(linha) for linha in handle]
        self.assertEqual(len(linhas), 2)
        self.assertEqual(linhas[1]["raw"], '{"e":"kline","n":2}')
        self.assertIn("ts_ms", linhas[0])

    def test_trilha_bruta_desligada_nao_cria_arquivo(self) -> None:
        with Storage(self.dir / "outro.sqlite", None) as sem_raw:
            sem_raw.append_raw("binance", "ethusdt", "{}")
        self.assertFalse((self.dir / "raw" / "binance").exists() and
                         list((self.dir / "raw" / "binance").glob("**/*.gz")))

    def test_estatisticas_para_o_comando_status(self) -> None:
        aberturas = grade(3, fim_ms=1_735_700_000_000)
        self.storage.save_many(candle(a) for a in aberturas)
        self.storage.record_event("reconnect")
        stats = self.storage.stats("ETHUSDT", "5m")
        self.assertEqual(stats["candles"], 3)
        self.assertEqual(stats["reconexoes"], 1)
        self.assertTrue(stats["primeira_vela_utc"] < stats["ultima_vela_utc"])


if __name__ == "__main__":
    unittest.main()
