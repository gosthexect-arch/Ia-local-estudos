"""Interface de terminal do cripto-monitor.

Comandos da Fase 1:
  doctor        diagnostico somente leitura do ambiente, dos dados e do modelo
  config        mostra a configuracao efetiva
  version       versao e fase atual
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from cripto_monitor import PHASE, __version__
from cripto_monitor.config import AppConfig, as_dict, default_config_path, load_config
from cripto_monitor.diagnostics import run_checks
from cripto_monitor.diagnostics import exchange as diag_exchange
from cripto_monitor.diagnostics import model as diag_model
from cripto_monitor.diagnostics import report as diag_report
from cripto_monitor.diagnostics import system as diag_system
from cripto_monitor.term import Painter, color_enabled

EPILOGO = (
    "Escopo: monitoramento e analise. Nenhum comando envia, altera ou cancela ordens, "
    "e nenhuma credencial de negociacao e lida."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cripto-monitor",
        description="Assistente local de monitoramento continuo de criptomoedas.",
        epilog=EPILOGO,
    )
    parser.add_argument("--version", action="version", version=f"cripto-monitor {__version__}")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        metavar="ARQUIVO",
        help=f"arquivo TOML de configuracao (padrao: {default_config_path()})",
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    doctor = sub.add_parser(
        "doctor",
        help="diagnostico somente leitura do ambiente (fase 1)",
        description=(
            "Verifica hardware, rede, dados publicos de mercado e o llama-server local. "
            "Nao instala nada, nao usa sudo e nao altera o sistema."
        ),
    )
    doctor.add_argument("--json", action="store_true", help="saida em JSON para automacao")
    doctor.add_argument("-v", "--verbose", action="store_true", help="mostra os detalhes brutos")
    doctor.add_argument(
        "--skip", action="append", default=[], choices=["sistema", "mercado", "modelo"],
        metavar="GRUPO", help="pula um grupo de verificacoes (sistema|mercado|modelo)",
    )
    doctor.add_argument(
        "--offline", action="store_true", help="atalho para --skip mercado --skip modelo"
    )

    cfg = sub.add_parser("config", help="mostra a configuracao efetiva")
    cfg.add_argument("--json", action="store_true", help="saida em JSON")

    sub.add_parser("version", help="versao e fase atual")
    return parser


def _grupos(args: argparse.Namespace) -> set[str]:
    pulados = set(args.skip)
    if args.offline:
        pulados |= {"mercado", "modelo"}
    return {"sistema", "mercado", "modelo"} - pulados


def cmd_doctor(args: argparse.Namespace, config: AppConfig) -> int:
    ativos = _grupos(args)
    sections: list[diag_report.Section] = []
    if "sistema" in ativos:
        sections.append(
            diag_report.Section("ambiente local", run_checks(diag_system.build_checks(config)))
        )
    if "mercado" in ativos:
        sections.append(
            diag_report.Section(
                f"dados de mercado ({config.exchange.name}, {config.exchange.symbol})",
                run_checks(diag_exchange.build_checks(config)),
            )
        )
    if "modelo" in ativos:
        sections.append(
            diag_report.Section("modelo local (llama-server)", run_checks(diag_model.build_checks(config)))
        )
    if not sections:
        print("nada a verificar: todos os grupos foram pulados", file=sys.stderr)
        return 2

    if args.json:
        diag_report.render_json(sections, config_dict=as_dict(config), stream=sys.stdout)
    else:
        diag_report.render_text(
            sections,
            stream=sys.stdout,
            paint=Painter(color_enabled(sys.stdout)),
            verbose=args.verbose,
        )
    return diag_report.exit_code(sections)


def cmd_config(args: argparse.Namespace, config: AppConfig) -> int:
    dados = as_dict(config)
    if args.json:
        json.dump(dados, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    paint = Painter(color_enabled(sys.stdout))
    origem = dados["source_path"] or "padroes do codigo (nenhum arquivo carregado)"
    print(paint(f"configuracao efetiva - origem: {origem}", "bold"))
    for secao, valores in dados.items():
        if not isinstance(valores, dict):
            continue
        print(paint(f"\n[{secao}]", "cyan"))
        for chave, valor in valores.items():
            print(f"  {chave:<20} {valor}")
    return 0


def cmd_version(_: argparse.Namespace, __: AppConfig) -> int:
    print(f"cripto-monitor {__version__} - fase {PHASE}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"erro de configuracao: {exc}", file=sys.stderr)
        return 2

    comandos = {"doctor": cmd_doctor, "config": cmd_config, "version": cmd_version}
    try:
        return comandos[args.comando](args, config)
    except KeyboardInterrupt:
        print("\ninterrompido pelo usuario", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # Saida cortada por `head`, `less` e afins: encerra sem stack trace.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 141


if __name__ == "__main__":
    raise SystemExit(main())
