"""Interface de terminal do cripto-monitor.

Fase 1 (somente leitura):
  doctor        diagnostico do ambiente, dos dados e do modelo
  config        mostra a configuracao efetiva
  version       versao e fase atual

Fase 2 (coleta):
  collect       mantem a conexao viva e grava velas fechadas
  backfill      recupera velas historicas por REST, sem WebSocket
  replay        reentrega o historico gravado, em ordem
  status        saude do armazenamento e continuidade das velas
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from cripto_monitor import PHASE, __version__
from cripto_monitor.config import AppConfig, as_dict, default_config_path, load_config
from cripto_monitor.diagnostics import run_checks
from cripto_monitor.diagnostics import exchange as diag_exchange
from cripto_monitor.diagnostics import model as diag_model
from cripto_monitor.diagnostics import report as diag_report
from cripto_monitor.diagnostics import system as diag_system
from cripto_monitor.storage import Storage
from cripto_monitor.term import Painter, color_enabled, human_ms

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

    coletar = sub.add_parser(
        "collect",
        help="coleta continua de velas (fase 2)",
        description=(
            "Mantem a conexao com o stream publico, grava velas fechadas, detecta "
            "conexao muda, reconecta com backoff e recupera lacunas por REST. "
            "Nao emite alerta e nao envia ordens."
        ),
    )
    coletar.add_argument("--duracao", type=float, metavar="SEG", help="para sozinho apos N segundos")
    coletar.add_argument(
        "--max-conexoes", type=int, metavar="N", help="encerra apos N conexoes (diagnostico)"
    )
    coletar.add_argument("--log-json", action="store_true", help="log estruturado em JSON")
    coletar.add_argument("--sem-backfill", action="store_true", help="nao recupera historico ao conectar")

    recuperar = sub.add_parser("backfill", help="recupera velas historicas por REST")
    recuperar.add_argument("--limit", type=int, default=500, metavar="N", help="velas por requisicao")
    recuperar.add_argument("--json", action="store_true", help="saida em JSON")

    reproduzir = sub.add_parser("replay", help="reentrega o historico gravado, em ordem")
    reproduzir.add_argument("--from", dest="inicio", metavar="INSTANTE", help="ISO-8601 ou epoch ms")
    reproduzir.add_argument("--to", dest="fim", metavar="INSTANTE", help="ISO-8601 ou epoch ms")
    reproduzir.add_argument("--limit", type=int, metavar="N", help="maximo de velas")
    reproduzir.add_argument("--resumo", action="store_true", help="so o resumo, sem as velas")

    saude = sub.add_parser("status", help="saude do armazenamento e continuidade das velas")
    saude.add_argument("--json", action="store_true", help="saida em JSON")

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


def _storage(config: AppConfig) -> Storage:
    return Storage(
        config.paths.db_path,
        config.paths.raw_dir if config.collector.store_raw else None,
    )


def _logger(paint: Painter, *, json_mode: bool) -> "object":
    """Log estruturado, sem dados sensiveis: o projeto nao lida com credenciais."""

    def registrar(evento: str, dados: dict[str, object]) -> None:
        agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if json_mode:
            print(json.dumps({"ts": agora, "event": evento, **dados}, ensure_ascii=False), flush=True)
            return
        campos = " ".join(f"{k}={v}" for k, v in dados.items())
        print(f"{paint(agora, 'grey')} {paint(evento, 'cyan')} {campos}", flush=True)

    return registrar


def cmd_collect(args: argparse.Namespace, config: AppConfig) -> int:
    from cripto_monitor.collector import Collector  # dependencia opcional: so aqui

    paint = Painter(color_enabled(sys.stdout))
    registrar = _logger(paint, json_mode=args.log_json)

    async def executar() -> int:
        with _storage(config) as storage:
            coletor = Collector(
                config, storage, log=registrar, backfill_ao_conectar=not args.sem_backfill
            )
            laco = asyncio.get_running_loop()
            for sinal in (signal.SIGINT, signal.SIGTERM):
                try:
                    laco.add_signal_handler(sinal, coletor.stop)
                except NotImplementedError:  # plataformas sem suporte a sinais
                    pass
            registrar("iniciando", {
                "symbol": config.exchange.symbol,
                "timeframe": config.exchange.primary_timeframe,
                "fonte": config.exchange.name,
                "banco": str(config.paths.db_path),
            })
            tarefa = asyncio.create_task(coletor.run(max_conexoes=args.max_conexoes))
            if args.duracao:
                laco.call_later(args.duracao, coletor.stop)
            estatisticas = await tarefa
            integridade = coletor.relatorio_de_integridade()
            registrar("encerrado", estatisticas.to_dict())
            registrar("integridade", integridade)
            # Coleta so e sucesso com historico continuo: servidor inacessivel,
            # conexao muda ou lacuna nao recuperada precisam falhar visivelmente.
            return 0 if integridade["continuo"] else 1

    try:
        return asyncio.run(executar())
    except RuntimeError as exc:  # websockets ausente
        print(f"erro: {exc}", file=sys.stderr)
        return 3


def cmd_backfill(args: argparse.Namespace, config: AppConfig) -> int:
    from cripto_monitor.collector import Collector

    async def executar() -> dict[str, object]:
        with _storage(config) as storage:
            coletor = Collector(config, storage)
            novas = await coletor.recuperar_historico(limite=args.limit)
            return {
                "novas": novas,
                "estatisticas": storage.stats(
                    config.exchange.symbol, config.exchange.primary_timeframe
                ),
            }

    resultado = asyncio.run(executar())
    if args.json:
        json.dump(resultado, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    stats = resultado["estatisticas"]
    paint = Painter(color_enabled(sys.stdout))
    print(paint(f"{resultado['novas']} velas novas gravadas", "bold"))
    print(f"  total no banco   {stats['candles']}")
    print(f"  primeira vela    {stats['primeira_vela_utc']}")
    print(f"  ultima vela      {stats['ultima_vela_utc']}")
    return 0 if resultado["novas"] or stats["candles"] else 1


def cmd_replay(args: argparse.Namespace, config: AppConfig) -> int:
    from cripto_monitor import replay as replay_mod

    symbol = config.exchange.symbol
    timeframe = config.exchange.primary_timeframe
    try:
        inicio = replay_mod.parse_instante(args.inicio) if args.inicio else None
        fim = replay_mod.parse_instante(args.fim) if args.fim else None
    except replay_mod.ReplayError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2

    with _storage(config) as storage:
        try:
            if not args.resumo:
                for vela in replay_mod.replay(
                    storage, symbol, timeframe, start_ms=inicio, end_ms=fim, limit=args.limit
                ):
                    print(json.dumps(vela.to_dict(), ensure_ascii=False), flush=True)
            resumo = replay_mod.resumir(storage, symbol, timeframe, start_ms=inicio, end_ms=fim)
        except replay_mod.ReplayError as exc:
            print(f"erro: {exc}", file=sys.stderr)
            return 1
    print(json.dumps({"resumo": resumo.to_dict()}, ensure_ascii=False))
    return 0 if resumo.velas else 1


def cmd_status(args: argparse.Namespace, config: AppConfig) -> int:
    symbol = config.exchange.symbol
    timeframe = config.exchange.primary_timeframe
    if not config.paths.db_path.exists():
        mensagem = f"nenhum banco em {config.paths.db_path}: rode 'collect' ou 'backfill' primeiro"
        if args.json:
            json.dump({"erro": mensagem}, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            print(mensagem, file=sys.stderr)
        return 1

    with _storage(config) as storage:
        estatisticas = storage.stats(symbol, timeframe)
        eventos = storage.recent_events(limit=5)
        from cripto_monitor.replay import resumir

        resumo = resumir(storage, symbol, timeframe)

    payload = {
        "estatisticas": estatisticas,
        "continuidade": resumo.to_dict(),
        "eventos_recentes": eventos,
    }
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
        sys.stdout.write("\n")
        return 0 if resumo.continuo else 1

    paint = Painter(color_enabled(sys.stdout))
    print(paint(f"{symbol} {timeframe} - {config.paths.db_path}", "bold"))
    print(f"  velas gravadas     {estatisticas['candles']}")
    print(f"  primeira           {estatisticas['primeira_vela_utc']}")
    print(f"  ultima             {estatisticas['ultima_vela_utc']}")
    idade = estatisticas["idade_ultima_vela_s"]
    if idade is not None:
        print(f"  idade da ultima    {human_ms(idade * 1000)}")
    print(f"  reconexoes         {estatisticas['reconexoes']}")
    print(f"  lacunas registradas {estatisticas['lacunas']}")
    print(f"  mensagens ruins    {estatisticas['descartes']}")
    cor = "green" if resumo.continuo else "yellow"
    print("  continuidade       " + paint("sem lacunas" if resumo.continuo else
                                          f"{len(resumo.lacunas)} lacuna(s)", cor))
    for lacuna in resumo.lacunas[:5]:
        print(f"      {lacuna['de_utc']} -> {lacuna['ate_utc']} ({lacuna['velas']} velas)")
    if eventos:
        print(paint("\n  ultimos eventos", "cyan"))
        for evento in eventos:
            quando = datetime.fromtimestamp(evento["ts_ms"] / 1000, tz=timezone.utc)
            print(f"      {quando.isoformat(timespec='seconds')} {evento['kind']} "
                  f"{evento['detail'] or ''}")
    return 0 if resumo.continuo else 1


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

    comandos = {
        "doctor": cmd_doctor,
        "collect": cmd_collect,
        "backfill": cmd_backfill,
        "replay": cmd_replay,
        "status": cmd_status,
        "config": cmd_config,
        "version": cmd_version,
    }
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
