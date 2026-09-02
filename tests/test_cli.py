"""Testes da interface de terminal."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout

from cripto_monitor import cli


def executar(argv: list[str]) -> tuple[int, str, str]:
    saida, erro = io.StringIO(), io.StringIO()
    with redirect_stdout(saida), redirect_stderr(erro):
        codigo = cli.main(argv)
    return codigo, saida.getvalue(), erro.getvalue()


class TestCli(unittest.TestCase):
    def test_version(self) -> None:
        codigo, saida, _ = executar(["version"])
        self.assertEqual(codigo, 0)
        self.assertIn("fase 1", saida)

    def test_config_json(self) -> None:
        codigo, saida, _ = executar(["config", "--json"])
        dados = json.loads(saida)
        self.assertEqual(codigo, 0)
        self.assertEqual(dados["exchange"]["symbol"], "ETHUSDT")
        self.assertNotIn("api_key", saida.lower())

    def test_doctor_offline_nao_toca_a_rede(self) -> None:
        codigo, saida, _ = executar(["doctor", "--offline"])
        self.assertIn(codigo, (0, 1))
        self.assertIn("ambiente local", saida)
        self.assertNotIn("dados de mercado", saida)
        self.assertNotIn("modelo local", saida)

    def test_doctor_json_tem_estrutura_estavel(self) -> None:
        codigo, saida, _ = executar(["doctor", "--offline", "--json"])
        payload = json.loads(saida)
        self.assertIn(codigo, (0, 1))
        self.assertEqual(payload["phase"], 1)
        self.assertIn(payload["status"], {"OK", "WARN", "FAIL", "SKIP"})
        nomes = [c["name"] for s in payload["sections"] for c in s["checks"]]
        self.assertIn("memoria", nomes)
        self.assertTrue(payload["generated_at"].endswith("+00:00"))

    def test_todos_os_grupos_pulados_e_uso_invalido(self) -> None:
        codigo, _, erro = executar(
            ["doctor", "--offline", "--skip", "sistema"]
        )
        self.assertEqual(codigo, 2)
        self.assertIn("nada a verificar", erro)

    def test_config_inexistente_retorna_codigo_2(self) -> None:
        codigo, _, erro = executar(["-c", "/nao/existe.toml", "version"])
        self.assertEqual(codigo, 2)
        self.assertIn("erro de configuracao", erro)

    def test_comando_obrigatorio(self) -> None:
        with self.assertRaises(SystemExit):
            executar([])

    def test_epilogo_declara_o_escopo(self) -> None:
        self.assertIn("Nenhum comando envia, altera ou cancela ordens", cli.EPILOGO)


if __name__ == "__main__":
    unittest.main()
