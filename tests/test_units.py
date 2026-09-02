"""Testes das funcoes puras: configuracao, schema e normalizacao de dados."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cripto_monitor import config as cfg
from cripto_monitor.diagnostics import report as rep
from cripto_monitor.diagnostics import system as sysd
from cripto_monitor.diagnostics.base import CheckResult, Status, run_checks, worst
from cripto_monitor.diagnostics.exchange import summarize_kline_event, summarize_kline_row
from cripto_monitor.diagnostics.model import extract_json_object
from cripto_monitor.schema import EXAMPLE_ALERT, validate_alert


class TestConfig(unittest.TestCase):
    def test_padroes_apontam_para_o_ambiente_do_projeto(self) -> None:
        c = cfg.load_config(env={})
        self.assertEqual(c.exchange.symbol, "ETHUSDT")
        self.assertEqual(c.exchange.primary_timeframe, "5m")
        self.assertEqual(c.llm.base_url, "http://127.0.0.1:8080/v1")
        self.assertEqual(c.llm.server_root, "http://127.0.0.1:8080")
        self.assertEqual(c.exchange.ws_stream_url, "wss://stream.binance.com:9443/ws/ethusdt@kline_5m")

    def test_arquivo_toml_sobrescreve_padroes(self) -> None:
        with TemporaryDirectory() as tmp:
            caminho = Path(tmp) / "monitor.toml"
            caminho.write_text(
                '[exchange]\nsymbol = "BTCUSDT"\nprimary_timeframe = "15m"\n'
                '[network]\nprobe_messages = 7\n',
                encoding="utf-8",
            )
            c = cfg.load_config(caminho, env={})
        self.assertEqual(c.exchange.symbol, "BTCUSDT")
        self.assertEqual(c.exchange.ws_stream_url.rsplit("/", 1)[-1], "btcusdt@kline_15m")
        self.assertEqual(c.network.probe_messages, 7)
        self.assertEqual(c.exchange.rest_base, cfg.ExchangeConfig().rest_base)

    def test_ambiente_tem_precedencia_e_converte_tipos(self) -> None:
        c = cfg.load_config(env={"CRIPTO_MONITOR_NETWORK_PROBE_MESSAGES": "9",
                                 "CRIPTO_MONITOR_LLM_MAX_LATENCY_S": "12.5"})
        self.assertEqual(c.network.probe_messages, 9)
        self.assertAlmostEqual(c.llm.max_latency_s, 12.5)

    def test_chave_desconhecida_e_erro_explicito(self) -> None:
        with self.assertRaisesRegex(ValueError, "chave desconhecida"):
            cfg.apply_toml(cfg.AppConfig(), {"exchange": {"api_key": "x"}})
        with self.assertRaisesRegex(ValueError, "secao desconhecida"):
            cfg.apply_toml(cfg.AppConfig(), {"trading": {}})

    def test_arquivo_inexistente_explicito_e_erro(self) -> None:
        with self.assertRaises(FileNotFoundError):
            cfg.load_config(Path("/nao/existe/monitor.toml"), env={})


class TestSchema(unittest.TestCase):
    def test_exemplo_da_spec_e_valido(self) -> None:
        self.assertEqual(validate_alert(EXAMPLE_ALERT), [])

    def test_decisao_fora_do_enum(self) -> None:
        alerta = dict(EXAMPLE_ALERT, decision="COMPRAR")
        self.assertIn("decision invalida: 'COMPRAR'", validate_alert(alerta))

    def test_confianca_fora_do_intervalo(self) -> None:
        self.assertTrue(any("intervalo" in e for e in validate_alert(dict(EXAMPLE_ALERT, confidence=1.4))))

    def test_campo_desconhecido_e_rejeitado(self) -> None:
        alerta = dict(EXAMPLE_ALERT, order_size=10)
        self.assertTrue(any("desconhecidos" in e for e in validate_alert(alerta)))

    def test_timestamp_sem_fuso(self) -> None:
        alerta = dict(EXAMPLE_ALERT, generated_at="2026-01-01T00:00:00")
        self.assertIn("generated_at precisa de fuso explicito (UTC)", validate_alert(alerta))

    def test_evidencia_inexistente(self) -> None:
        alerta = dict(EXAMPLE_ALERT, evidence_ids=["snap-1"])
        erros = validate_alert(alerta, known_evidence={"snap-0"})
        self.assertTrue(any("evidencias inexistentes" in e for e in erros))

    def test_vies_com_dados_indisponiveis_e_incoerente(self) -> None:
        alerta = dict(EXAMPLE_ALERT, decision="BUY_BIAS", data_quality="UNAVAILABLE")
        self.assertTrue(any("incompativel" in e for e in validate_alert(alerta)))

    def test_resposta_que_nao_e_objeto(self) -> None:
        self.assertEqual(len(validate_alert(["HOLD"])), 1)


class TestParsersDeSistema(unittest.TestCase):
    def test_meminfo_converte_para_bytes(self) -> None:
        mem = sysd.parse_meminfo("MemTotal:  16316164 kB\nMemAvailable: 12000000 kB\nHugePages_Total: 0\n")
        self.assertEqual(mem["MemTotal"], 16316164 * 1024)
        self.assertEqual(mem["HugePages_Total"], 0)

    def test_cpuinfo_conta_threads_e_sockets(self) -> None:
        texto = (
            "processor\t: 0\nmodel name\t: Intel(R) Xeon(R) CPU E5-2650 v4\nphysical id\t: 0\n\n"
            "processor\t: 1\nmodel name\t: Intel(R) Xeon(R) CPU E5-2650 v4\nphysical id\t: 0\n\n"
        )
        info = sysd.parse_cpuinfo(texto)
        self.assertEqual(info["logical_cpus"], 2)
        self.assertEqual(info["sockets"], 1)
        self.assertIn("E5-2650", info["model"])


class TestNormalizacaoDeVelas(unittest.TestCase):
    LINHA = [
        1735689600000, "3500.10", "3510.00", "3495.00", "3505.55", "120.5",
        1735689899999, "422000.0", 987, "60.2", "211000.0", "0",
    ]

    def test_vela_passada_e_closed(self) -> None:
        vela = summarize_kline_row(self.LINHA, now_ms=1735690000000)
        self.assertEqual(vela["state"], "closed")
        self.assertEqual(vela["close"], 3505.55)
        self.assertTrue(vela["close_time_utc"].endswith("+00:00"))

    def test_vela_em_andamento_e_forming(self) -> None:
        vela = summarize_kline_row(self.LINHA, now_ms=1735689700000)
        self.assertEqual(vela["state"], "forming")

    def test_linha_curta_e_rejeitada(self) -> None:
        with self.assertRaises(ValueError):
            summarize_kline_row([1, "2", "3"])

    def test_evento_de_stream(self) -> None:
        evento = summarize_kline_event(
            {"e": "kline", "E": 1735689899000, "k": {
                "t": 1735689600000, "T": 1735689899999, "s": "ETHUSDT", "i": "5m",
                "c": "3505.55", "v": "120.5", "x": True}}
        )
        self.assertEqual(evento["state"], "closed")
        self.assertEqual(evento["symbol"], "ETHUSDT")

    def test_evento_sem_kline(self) -> None:
        with self.assertRaises(ValueError):
            summarize_kline_event({"e": "trade"})


class TestExtracaoDeJson(unittest.TestCase):
    def test_json_puro(self) -> None:
        self.assertEqual(extract_json_object('{"decision": "HOLD"}'), {"decision": "HOLD"})

    def test_json_em_cerca_markdown(self) -> None:
        self.assertEqual(extract_json_object('```json\n{"a": 1}\n```'), {"a": 1})

    def test_json_com_texto_em_volta(self) -> None:
        bruto = 'Claro! Segue a analise:\n{"a": {"b": "}"}}\nEspero ter ajudado.'
        self.assertEqual(extract_json_object(bruto), {"a": {"b": "}"}})

    def test_sem_json(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_object("nao vou responder em JSON")

    def test_json_truncado(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_object('{"decision": "HOLD"')


class TestAgregacaoDeResultados(unittest.TestCase):
    def _r(self, status: Status) -> CheckResult:
        return CheckResult("x", status, "resumo")

    def test_pior_status_vence(self) -> None:
        self.assertIs(worst([self._r(Status.OK), self._r(Status.WARN)]), Status.WARN)
        self.assertIs(worst([self._r(Status.FAIL), self._r(Status.WARN)]), Status.FAIL)
        self.assertIs(worst([]), Status.SKIP)

    def test_excecao_em_um_check_nao_derruba_os_demais(self) -> None:
        def explode() -> CheckResult:
            raise RuntimeError("boom")

        resultados = run_checks([("ruim", explode), ("bom", lambda: self._r(Status.OK))])
        self.assertIs(resultados[0].status, Status.FAIL)
        self.assertIn("boom", resultados[0].summary)
        self.assertIs(resultados[1].status, Status.OK)

    def test_exit_code_so_falha_com_FAIL(self) -> None:
        avisos = [rep.Section("s", [self._r(Status.WARN), self._r(Status.OK)])]
        falhas = [rep.Section("s", [self._r(Status.FAIL)])]
        self.assertEqual(rep.exit_code(avisos), 0)
        self.assertEqual(rep.exit_code(falhas), 1)

    def test_proximos_passos_apontam_a_fase_seguinte(self) -> None:
        passos = rep.proximos_passos([CheckResult("websocket", Status.FAIL, "-")])
        self.assertTrue(any("Fase 2" in p or "Coinbase" in p for p in passos))


if __name__ == "__main__":
    unittest.main()
