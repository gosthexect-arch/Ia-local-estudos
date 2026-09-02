"""Testes de ponta a ponta dos diagnosticos contra servidores locais falsos."""

from __future__ import annotations

import json
import threading
import time
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cripto_monitor.config import AppConfig, ExchangeConfig, LLMConfig, NetworkConfig
from cripto_monitor.diagnostics import exchange as diag_exchange
from cripto_monitor.diagnostics import model as diag_model
from cripto_monitor.diagnostics.base import Status, run_checks
from cripto_monitor.diagnostics.report import proximos_passos
from cripto_monitor.schema import EXAMPLE_ALERT
from test_wsprobe import FakeWebSocketServer, server_frame
from cripto_monitor.net import wsprobe

AGORA_MS = int(time.time() * 1000)
VELA_FECHADA = [
    AGORA_MS - 600_000, "3500.10", "3510.00", "3495.00", "3505.55", "120.5",
    AGORA_MS - 300_001, "422000.0", 987, "60.2", "211000.0", "0",
]
VELA_ABERTA = [
    AGORA_MS - 300_000, "3505.55", "3512.00", "3501.00", "3508.00", "40.1",
    AGORA_MS + 299_999, "140000.0", 300, "20.0", "70000.0", "0",
]


class FakeHandler(BaseHTTPRequestHandler):
    """Responde tanto como exchange publica quanto como llama-server."""

    resposta_do_modelo = json.dumps(EXAMPLE_ALERT)
    drift_ms = 0

    def log_message(self, *_: object) -> None:  # silencia o log do servidor de teste
        pass

    def _send(self, payload: object, status: int = 200) -> None:
        corpo = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self) -> None:
        rota = self.path.split("?")[0]
        if rota == "/api/v3/ping":
            self._send({})
        elif rota == "/api/v3/time":
            self._send({"serverTime": int(time.time() * 1000) - type(self).drift_ms})
        elif rota == "/api/v3/klines":
            self._send([VELA_FECHADA, VELA_ABERTA])
        elif rota == "/health":
            self._send({"status": "ok"})
        elif rota == "/props":
            self._send({"n_ctx": 65536, "model_path": "/home/ght/models/gemma4/modelo.gguf"})
        elif rota == "/v1/models":
            self._send({"data": [{"id": "gemma-4-e4b"}]})
        else:
            self._send({"erro": "rota desconhecida"}, status=404)

    def do_POST(self) -> None:
        tamanho = int(self.headers.get("Content-Length", 0))
        self.rfile.read(tamanho)
        if self.path == "/v1/chat/completions":
            self._send({"choices": [{"message": {"content": type(self).resposta_do_modelo}}]})
        else:
            self._send({"erro": "rota desconhecida"}, status=404)


class ServidorLocal:
    def __enter__(self) -> "ServidorLocal":
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def config_para(port: int, **rede: object) -> AppConfig:
    return AppConfig(
        exchange=ExchangeConfig(rest_base=f"http://127.0.0.1:{port}", ws_base=f"ws://127.0.0.1:{port}"),
        llm=LLMConfig(base_url=f"http://127.0.0.1:{port}/v1", model_path="/inexistente.gguf"),
        network=NetworkConfig(connect_timeout_s=5.0, probe_messages=1, probe_timeout_s=5.0, **rede),
    )


class TestDiagnosticoDeMercado(unittest.TestCase):
    def test_ping_time_e_klines(self) -> None:
        FakeHandler.drift_ms = 0
        with ServidorLocal() as servidor:
            config = config_para(servidor.port)
            self.assertIs(diag_exchange.check_rest_ping(config).status, Status.OK)
            self.assertIs(diag_exchange.check_clock_drift(config).status, Status.OK)
            klines = diag_exchange.check_klines(config)
        self.assertIs(klines.status, Status.OK)
        self.assertEqual(klines.details["fechadas"], 1)
        self.assertEqual([v["state"] for v in klines.details["velas"]], ["closed", "forming"])

    def test_deriva_de_relogio_alem_do_limite_vira_aviso(self) -> None:
        FakeHandler.drift_ms = 5000
        try:
            with ServidorLocal() as servidor:
                resultado = diag_exchange.check_clock_drift(config_para(servidor.port))
        finally:
            FakeHandler.drift_ms = 0
        self.assertIs(resultado.status, Status.WARN)
        self.assertGreater(resultado.details["drift_ms"], 4000)

    def test_endpoint_fora_do_ar_e_falha_com_dica(self) -> None:
        config = config_para(9)  # porta 9 (discard): conexao recusada
        resultado = diag_exchange.check_rest_ping(replace(config, network=NetworkConfig(connect_timeout_s=2.0)))
        self.assertIs(resultado.status, Status.FAIL)
        self.assertIn("Coinbase", resultado.hint or "")

    def test_websocket_recebe_evento_de_vela(self) -> None:
        evento = json.dumps({
            "e": "kline", "E": AGORA_MS,
            "k": {"t": AGORA_MS - 300_000, "T": AGORA_MS, "s": "ETHUSDT", "i": "5m",
                  "c": "3505.55", "v": "120.5", "x": True},
        })
        with FakeWebSocketServer([server_frame(wsprobe.OP_TEXT, evento.encode())]) as servidor:
            resultado = diag_exchange.check_websocket(config_para(servidor.port))
        self.assertIs(resultado.status, Status.OK)
        self.assertEqual(resultado.details["eventos"][0]["state"], "closed")
        self.assertEqual(resultado.details["eventos_invalidos"], 0)

    def test_websocket_com_mensagem_invalida_vira_aviso(self) -> None:
        with FakeWebSocketServer([server_frame(wsprobe.OP_TEXT, b"nao e json")]) as servidor:
            resultado = diag_exchange.check_websocket(config_para(servidor.port))
        self.assertIs(resultado.status, Status.WARN)
        self.assertEqual(resultado.details["eventos_invalidos"], 1)


class TestDiagnosticoDoModelo(unittest.TestCase):
    def setUp(self) -> None:
        FakeHandler.resposta_do_modelo = json.dumps(EXAMPLE_ALERT)

    def test_servidor_saudavel_e_alerta_valido(self) -> None:
        with ServidorLocal() as servidor:
            config = config_para(servidor.port)
            self.assertIs(diag_model.check_server_health(config).status, Status.OK)
            self.assertIs(diag_model.check_models_endpoint(config).status, Status.OK)
            props = diag_model.check_server_props(config)
            resultado = diag_model.check_json_capability(config)
        self.assertEqual(props.details["n_ctx"], 65536)
        self.assertIs(resultado.status, Status.OK)
        self.assertEqual(resultado.details["erros_schema"], [])

    def test_resposta_em_prosa_e_falha(self) -> None:
        FakeHandler.resposta_do_modelo = "Acho que o ETH vai subir bastante hoje!"
        with ServidorLocal() as servidor:
            resultado = diag_model.check_json_capability(config_para(servidor.port))
        self.assertIs(resultado.status, Status.FAIL)
        self.assertIn("nao devolveu JSON", resultado.summary)

    def test_json_fora_do_schema_e_aviso(self) -> None:
        FakeHandler.resposta_do_modelo = json.dumps({"decision": "BUY_BIAS", "confidence": 3})
        with ServidorLocal() as servidor:
            resultado = diag_model.check_json_capability(config_para(servidor.port))
        self.assertIs(resultado.status, Status.WARN)
        self.assertTrue(resultado.details["erros_schema"])

    def test_llama_server_ausente_nao_derruba_o_diagnostico(self) -> None:
        resultado = diag_model.check_server_health(config_para(9))
        self.assertIs(resultado.status, Status.FAIL)
        self.assertIn("llama-server", resultado.hint or "")

    def test_modelo_ausente_no_disco_e_aviso(self) -> None:
        resultado = diag_model.check_model_file(config_para(9))
        self.assertIs(resultado.status, Status.WARN)

    def test_servidor_fora_do_ar_gera_uma_falha_e_o_resto_pulado(self) -> None:
        resultados = run_checks(diag_model.build_checks(config_para(9)))
        por_nome = {r.name: r.status for r in resultados}
        self.assertIs(por_nome["llama_health"], Status.FAIL)
        for nome in ("llama_props", "llama_models", "llama_json"):
            self.assertIs(por_nome[nome], Status.SKIP, nome)

    def test_servidor_no_ar_executa_a_sequencia_inteira(self) -> None:
        with ServidorLocal() as servidor:
            resultados = run_checks(diag_model.build_checks(config_para(servidor.port)))
        por_nome = {r.name: r.status for r in resultados}
        self.assertIs(por_nome["llama_health"], Status.OK)
        self.assertIs(por_nome["llama_json"], Status.OK)

    def test_proximo_passo_para_servidor_desligado(self) -> None:
        resultados = run_checks(diag_model.build_checks(config_para(9)))
        passos = proximos_passos(resultados)
        self.assertTrue(any("llama-server desligado" in p for p in passos), passos)


if __name__ == "__main__":
    unittest.main()
