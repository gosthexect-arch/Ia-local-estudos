# cripto-monitor

Assistente local de monitoramento continuo de criptomoedas, para terminal.

**Escopo: observacao e analise.** O programa nao envia, altera ou cancela ordens,
nao pede chave de negociacao e nao tem permissao de saque. Um alerta e uma leitura
de cenario rastreavel ate os dados que a originaram, nunca uma ordem nem promessa
de resultado. A decisao financeira e sempre do usuario.

Documentos de origem do projeto: [`docs/00-conceito.md`](docs/00-conceito.md) e
[`docs/01-especificacao.md`](docs/01-especificacao.md).

## Estado atual: Fase 1 (diagnostico somente leitura)

O que existe hoje e o comando `doctor`: ele mede o ambiente antes de qualquer
coisa ser construida em cima dele. Nao instala pacotes, nao usa `sudo`, nao
escreve em disco e nao altera configuracao do sistema.

Roda **apenas com a biblioteca padrao do Python 3.11+** — nada para instalar.

```bash
git clone <este-repo> && cd Ia-local-estudos
PYTHONPATH=src python3 -m cripto_monitor doctor
```

Ou, instalando no ambiente do usuario para ganhar o comando `cripto-monitor`:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
cripto-monitor doctor
```

### Comandos

| Comando | O que faz |
|---|---|
| `cripto-monitor doctor` | diagnostico completo: ambiente, dados de mercado e modelo local |
| `cripto-monitor doctor --offline` | so o ambiente local; nao toca a rede |
| `cripto-monitor doctor --skip modelo` | pula o bloco do llama-server (util com o servidor desligado) |
| `cripto-monitor doctor --json` | mesmo relatorio em JSON, para automacao |
| `cripto-monitor doctor -v` | inclui os detalhes brutos de cada verificacao |
| `cripto-monitor config` | mostra a configuracao efetiva e de onde ela veio |

Codigo de saida: `0` quando nada falhou (avisos nao reprovam), `1` com qualquer
falha, `2` para erro de uso ou de configuracao.

### O que o `doctor` verifica

**Ambiente local** — versao do Python, host/sessao, CPU e load, RAM e swap, GPU e
driver (`amdgpu` esperado para a RX 580), relogio em UTC, e se o diretorio de
dados e gravavel com espaco livre.

**Dados de mercado** — resolucao DNS, `ping` REST, deriva do relogio local contra
a exchange, tres velas historicas do timeframe principal (validando ordem e o
estado `closed`/`forming`) e uma sondagem curta do stream de velas por WebSocket,
com validacao do handshake e do formato do evento.

**Modelo local** — presenca do arquivo GGUF, `/health` e `/props` do
`llama-server` (inclusive o `n_ctx` efetivo), `/v1/models`, e um teste real de
geracao: o modelo precisa devolver um alerta em JSON que passe pelo validador de
schema dentro do prazo configurado.

O teste de JSON responde diretamente a pergunta 5 da especificacao — se o modelo
ja fecha o contrato estruturado. Falhar ali nao bloqueia as fases 2 e 3, que sao
deterministicas e independem do LLM.

## Configuracao

Precedencia: padroes do codigo < arquivo TOML < variaveis de ambiente.

```bash
mkdir -p ~/.config/cripto-monitor
cp config/monitor.example.toml ~/.config/cripto-monitor/monitor.toml
```

Variaveis seguem o formato `CRIPTO_MONITOR_<SECAO>_<CHAVE>`:

```bash
CRIPTO_MONITOR_EXCHANGE_SYMBOL=BTCUSDT cripto-monitor doctor --skip modelo
```

Chaves ou secoes desconhecidas sao erro explicito, e nao silencio.

## Testes

```bash
make test        # ou: PYTHONPATH=src:tests python3 -m unittest discover -s tests -v
```

A suite sobe um servidor WebSocket e um servidor HTTP falsos em processo, entao
exercita o handshake, o codec de frames, a normalizacao de velas, o validador de
schema e os caminhos de falha (servidor fora do ar, prosa no lugar de JSON,
conexao viva porem sem dados) sem depender de rede externa.

## Arquitetura

```
src/cripto_monitor/
  cli.py                  interface de terminal
  config.py               configuracao em camadas (TOML + ambiente)
  schema.py               contrato do alerta e validador
  term.py                 formatacao de terminal, sem dependencias
  net/http.py             cliente HTTP minimo (nunca via proxy para o llama-server)
  net/wsprobe.py          cliente WebSocket RFC 6455 para sondagem
  diagnostics/system.py   ambiente local
  diagnostics/exchange.py dados publicos de mercado
  diagnostics/model.py    llama-server e contrato JSON
  diagnostics/report.py   relatorio em texto e em JSON
```

A interface da exchange fica isolada em `diagnostics/exchange.py` e `config.py`:
trocar Binance por Coinbase ou Kraken nao deve alcancar a estrategia nem o
armazenamento.

## Roadmap

| Fase | Entrega | Estado |
|---|---|---|
| 1 | Diagnostico somente leitura do ambiente, da fonte de dados e do modelo | **pronto** |
| 2 | Coletor WebSocket persistente, candles de 5m, storage e replay | a fazer |
| 3 | Indicadores deterministicos, eventos de mercado e dashboard sem LLM | a fazer |
| 4 | Contexto compacto, cliente LLM, alertas idempotentes e explicaveis | a fazer |
| 5 | Backtesting, walk-forward e testes de falha | a fazer |

A Fase 2 e a primeira que grava em disco e a primeira que precisara de
dependencias externas (`websockets`), o que sera pedido antes de instalar.

## Limites que nao mudam

Sem execucao de ordens, sem integracao com saldo, sem alavancagem, sem saque.
Vela em formacao nunca confirma decisao. Tudo em UTC. Se o modelo cair, ficar
lento ou responder fora do schema, a coleta continua e nenhum alerta direcional
novo e emitido.
