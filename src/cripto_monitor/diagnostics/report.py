"""Renderizacao do relatorio de diagnostico no terminal e em JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TextIO

from cripto_monitor import __version__
from cripto_monitor.diagnostics.base import CheckResult, Status, worst
from cripto_monitor.term import Painter, human_ms

MARCAS = {
    Status.OK: ("ok  ", "green"),
    Status.WARN: ("avis", "yellow"),
    Status.FAIL: ("FALH", "red"),
    Status.SKIP: ("--  ", "grey"),
}


@dataclass(slots=True)
class Section:
    title: str
    results: list[CheckResult]


def to_payload(sections: list[Section], *, config_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool": "cripto-monitor",
        "version": __version__,
        "phase": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": worst([r for s in sections for r in s.results]).value,
        "config": config_dict,
        "sections": [
            {"title": s.title, "checks": [r.to_dict() for r in s.results]} for s in sections
        ],
    }


def render_json(sections: list[Section], *, config_dict: dict[str, Any], stream: TextIO) -> None:
    json.dump(to_payload(sections, config_dict=config_dict), stream, ensure_ascii=False, indent=2)
    stream.write("\n")


def render_text(
    sections: list[Section],
    *,
    stream: TextIO,
    paint: Painter,
    verbose: bool = False,
) -> None:
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stream.write(
        paint(f"cripto-monitor {__version__}", "bold")
        + paint(f"  fase 1 - diagnostico somente leitura  {agora}", "grey")
        + "\n"
    )
    stream.write(
        paint("observacao apenas: este software nao envia, altera ou cancela ordens\n", "grey")
    )

    for section in sections:
        stream.write("\n" + paint(section.title, "bold", "cyan") + "\n")
        for result in section.results:
            marca, cor = MARCAS[result.status]
            stream.write(
                f"  {paint(marca, cor)} {result.name:<16} {result.summary}"
                f" {paint(f'({human_ms(result.duration_ms)})', 'grey')}\n"
            )
            if result.hint and result.status is not Status.OK:
                stream.write(f"       {paint('-> ' + result.hint, 'dim')}\n")
            if verbose and result.details:
                for chave, valor in result.details.items():
                    texto = json.dumps(valor, ensure_ascii=False, default=str)
                    if len(texto) > 300:
                        texto = texto[:297] + "..."
                    stream.write(f"       {paint(f'{chave}: {texto}', 'grey')}\n")

    todos = [r for s in sections for r in s.results]
    contagem = {status: sum(1 for r in todos if r.status is status) for status in Status}
    geral = worst(todos)
    _, cor = MARCAS[geral]
    stream.write(
        "\n"
        + paint("resultado: ", "bold")
        + paint(geral.value, cor, "bold")
        + paint(
            f"  ({contagem[Status.OK]} ok, {contagem[Status.WARN]} avisos, "
            f"{contagem[Status.FAIL]} falhas, {contagem[Status.SKIP]} pulados)",
            "grey",
        )
        + "\n"
    )
    for linha in proximos_passos(todos):
        stream.write(paint(f"  . {linha}", "dim") + "\n")


def proximos_passos(results: list[CheckResult]) -> list[str]:
    """Traduz o estado atual na proxima acao concreta do plano de fases."""
    por_nome = {r.name: r.status for r in results}
    passos: list[str] = []
    if por_nome.get("websocket") is Status.FAIL or por_nome.get("rest_ping") is Status.FAIL:
        passos.append(
            "Fonte de dados indisponivel: reavaliar Binance/Coinbase/Kraken antes da Fase 2."
        )
    if por_nome.get("llama_json") in (Status.FAIL, Status.WARN):
        passos.append(
            "Modelo ainda nao fecha o contrato JSON: a Fase 2 e a 3 nao dependem disso e "
            "podem seguir sem ele."
        )
    if por_nome.get("clock_drift") is Status.WARN:
        passos.append("Sincronizar o relogio local antes de confiar em fechamentos de vela.")
    if not passos:
        passos.append("Ambiente aprovado na Fase 1. Proximo marco: coletor 5m com replay (Fase 2).")
    return passos


def exit_code(sections: list[Section]) -> int:
    return 1 if worst([r for s in sections for r in s.results]) is Status.FAIL else 0
