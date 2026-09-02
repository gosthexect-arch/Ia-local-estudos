# cripto-monitor

Assistente local de monitoramento continuo de criptomoedas, para terminal.

**Escopo: observacao e analise.** O programa nao envia, altera ou cancela ordens,
nao pede chave de negociacao e nao tem permissao de saque. Um alerta e uma leitura
de cenario rastreavel ate os dados que a originaram, nunca uma ordem nem promessa
de resultado. A decisao financeira e sempre do usuario.

Documentos de origem: [`docs/00-conceito.md`](docs/00-conceito.md) e
[`docs/01-especificacao.md`](docs/01-especificacao.md).

## Instalacao

Python 3.11+. A Fase 1 nao precisa de nada alem da biblioteca padrao; a Fase 2
precisa de um cliente WebSocket.

```bash
git clone -b claude/project-summary-swe8g6 https://github.com/gosthexect-arch/Ia-local-estudos.git
cd Ia-local-estudos
sudo pacman -S python-websockets     # Arch; em outras distros: pip install websockets
PYTHONPATH=src python3 -m cripto_monitor doctor
```

O `-b` e necessario enquanto o trabalho nao estiver na branch principal. Ja tendo
o repositorio clonado, basta `git pull`.

Para ganhar o comando `cripto-monitor` no PATH e dispensar o `PYTHONPATH`:

```bash
python3 -m venv .venv --system-site-packages && . .venv/bin/activate
pip install -e . --no-build-isolation
cripto-monitor doctor
```

## Comandos

| Comando | O que faz |
|---|---|
| `doctor` | diagnostico somente leitura: ambiente, dados de mercado e modelo local |
| `collect` | coleta continua: conecta, grava velas fechadas, reconecta, recupera lacunas |
| `backfill` | recupera velas historicas por REST, sem abrir WebSocket |
| `replay` | reentrega o historico gravado, em ordem cronologica estrita |
| `status` | saude do armazenamento, continuidade das velas e ultimos eventos |
| `config` | mostra a configuracao efetiva e de onde ela veio |

Codigo de saida: `0` em sucesso, `1` em falha real (diagnostico reprovado, coleta
sem historico continuo, replay vazio), `2` em erro de uso ou configuracao, `3`
quando falta a biblioteca `websockets`.

## Fase 1 - diagnostico

Mede o ambiente antes de qualquer coisa ser construida em cima dele. Nao instala
pacotes, nao usa `sudo`, nao escreve em disco e nao altera o sistema.

```bash
cripto-monitor doctor              # tudo
cripto-monitor doctor --offline    # so o ambiente local, sem tocar a rede
cripto-monitor doctor --skip modelo   # util com o llama-server desligado
cripto-monitor doctor --json       # para automacao
cripto-monitor doctor -v           # com os detalhes brutos
```

**Ambiente local** — Python, host/sessao, CPU e load, RAM e swap, GPU e driver
(`amdgpu` esperado para a RX 580), dependencias, relogio em UTC e diretorio de
dados gravavel.

**Dados de mercado** — DNS, `ping` REST, deriva do relogio local contra a
exchange, velas historicas (validando ordem e o estado `closed`/`forming`) e uma
sondagem curta do stream, com handshake e formato do evento verificados.

**Modelo local** — arquivo GGUF, `/health` e `/props` do `llama-server` (inclusive
o `n_ctx` efetivo), `/v1/models` e um teste real de geracao: o modelo precisa
devolver um alerta JSON que passe pelo validador de schema dentro do prazo. Com o
servidor fora do ar, `llama_health` falha uma vez e o resto e pulado.

## Fase 2 - coleta continua

```bash
cripto-monitor backfill            # popula o historico por REST
cripto-monitor collect             # fica coletando; Ctrl-C encerra limpo
cripto-monitor status              # o historico esta continuo?
cripto-monitor replay --resumo     # o que foi gravado, em ordem
```

O que o coletor garante:

- **Conexao viva de verdade.** Ping/pong da biblioteca mais um watchdog proprio:
  passado `stale_after_s` sem nenhuma mensagem, a conexao e considerada morta
  mesmo "aberta", derrubada e refeita. Reconexao usa backoff exponencial com
  jitter e teto.
- **Idempotencia.** A chave e (fonte, simbolo, timeframe, abertura). Reentrega
  apos reconexao, backfill sobreposto e replay atualizam a vela, nunca duplicam.
  O callback de vela fechada dispara uma unica vez por vela — e disso que a
  Fase 4 dependera para nao repetir alerta.
- **Lacunas.** Uma vela nova cuja anterior nao existe denuncia periodo perdido:
  o evento e registrado e o REST preenche o buraco. `status` mostra o que sobrou.
- **Vela em formacao nunca vira fato.** Somente velas fechadas entram no banco.
- **Nada derruba a coleta.** JSON quebrado, vela impossivel, REST fora do ar e
  queda de rede viram evento no diario e a coleta segue.

Onde os dados ficam (padrao `$XDG_DATA_HOME/cripto-monitor`):

```
state.sqlite                        velas fechadas, diario de eventos, estado
raw/binance/ETHUSDT/AAAA-MM-DD.jsonl.gz   mensagem original, para auditoria
```

### Rodar como servico

```bash
mkdir -p ~/.config/systemd/user
cp systemd/cripto-monitor.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now cripto-monitor
journalctl --user -u cripto-monitor -f
loginctl enable-linger $USER    # para seguir rodando apos o logout
```

O servico nao depende do Hyprland. Reiniciar nao perde estado nem reemite vela
ja gravada. O bloco de endurecimento da unit deixa o processo com acesso de
escrita apenas ao diretorio de dados.

## Configuracao

Precedencia: padroes do codigo < arquivo TOML < variaveis de ambiente.

```bash
mkdir -p ~/.config/cripto-monitor
cp config/monitor.example.toml ~/.config/cripto-monitor/monitor.toml
```

Variaveis seguem `CRIPTO_MONITOR_<SECAO>_<CHAVE>`:

```bash
CRIPTO_MONITOR_COLLECTOR_STALE_AFTER_S=30 cripto-monitor collect
```

Chaves ou secoes desconhecidas sao erro explicito, e nao silencio.

## Testes

```bash
make test    # ou: PYTHONPATH=src:tests python3 -m unittest discover -s tests -v
```

A suite sobe servidores HTTP e WebSocket reais em processo e nao depende de rede
externa. Cobre o codec de frames, a normalizacao de velas, o validador de schema,
a idempotencia do armazenamento e os cenarios ruins que importam: conexao viva
porem muda, queda no meio do stream, lacuna de vela, JSON corrompido, REST fora
do ar e reentrega apos reconexao.

## Arquitetura

```
src/cripto_monitor/
  cli.py                  interface de terminal
  config.py               configuracao em camadas (TOML + ambiente)
  candles.py              modelo de vela, grade de timeframes, lacunas
  storage.py              SQLite (estado) e JSONL comprimido (bruto)
  collector.py            laco de coleta, watchdog, backoff, backfill
  replay.py               reentrega ordenada do historico
  schema.py               contrato do alerta e validador
  term.py                 formatacao de terminal, sem dependencias
  exchanges/base.py       interface abstrata de exchange
  exchanges/binance.py    adaptador Binance (somente dados publicos)
  net/http.py             cliente HTTP minimo (nunca via proxy para o llama-server)
  net/wsprobe.py          cliente WebSocket RFC 6455 para sondagem
  diagnostics/            ambiente, mercado, modelo, relatorio
```

Toda conversa com a exchange passa pelo adaptador. O armazenamento, o replay e a
estrategia nao conhecem Binance: adicionar Coinbase ou Kraken e escrever um
adaptador novo.

## Roadmap

| Fase | Entrega | Estado |
|---|---|---|
| 1 | Diagnostico somente leitura do ambiente, da fonte e do modelo | **pronto** |
| 2 | Coletor WebSocket persistente, candles 5m, storage e replay | **pronto** |
| 3 | Indicadores deterministicos, eventos de mercado e dashboard sem LLM | a fazer |
| 4 | Contexto compacto, cliente LLM, alertas idempotentes e explicaveis | a fazer |
| 5 | Backtesting, walk-forward e testes de falha | a fazer |

## Limites que nao mudam

Sem execucao de ordens, sem integracao com saldo, sem alavancagem, sem saque.
Vela em formacao nunca confirma decisao. Tudo em UTC. Se o modelo cair, ficar
lento ou responder fora do schema, a coleta continua e nenhum alerta direcional
novo e emitido.
