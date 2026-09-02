# Prompt para Claude Code — Assistente local de monitoramento cripto

## Papel

Você é o Claude Code responsável por analisar, planejar e desenvolver um software local de monitoramento contínuo de criptomoedas. Antes de modificar arquivos, instalar dependências ou alterar configurações do sistema, faça diagnóstico somente leitura, explique o plano e peça confirmação.

## O que quero construir

Quero construir um assistente local que acompanhe constantemente gráficos e dados de criptomoedas enquanto estiver em operação. O sistema deverá monitorar principalmente ETH/USDT usando uma conexão WebSocket pública e gratuita, sem executar ordens e sem possuir chaves privadas de negociação.

A unidade principal de análise será a vela de cinco minutos. O programa deverá receber atualizações contínuas da vela, identificar quando ela estiver fechada, calcular indicadores e enviar ao modelo local um contexto atualizado. O modelo deverá gerar alertas explicáveis, como `BUY_BIAS`, `SELL_BIAS` ou `HOLD`, mas esses alertas não serão ordens e não deverão movimentar dinheiro.

“Ver constantemente” significa manter a conexão de dados ativa, detectar alterações, reconectar quando necessário e atualizar o estado local. Não significa enviar screenshots a cada segundo ao modelo. O sistema deve usar dados estruturados e somente solicitar uma nova análise completa a cada fechamento de candle de cinco minutos ou quando ocorrer um evento importante, como mudança de regime, rompimento, volume anormal ou alteração relevante do livro.

## Ambiente de desenvolvimento

O sistema-alvo é Arch Linux, em boot único, com Hyprland, DankMaterialShell e Wayland. O usuário é `ght`, o hostname é `Ghosth`, e os terminais disponíveis incluem Ghostty, WezTerm e Kitty.

Hardware conhecido:

- Intel Xeon E5-2650 v4.
- Aproximadamente 15 GiB de RAM.
- AMD Radeon RX 580 2048SP com `amdgpu` e Vulkan RADV.
- Aproximadamente 23 GiB de swap.

O modelo local está em:

```text
/home/ght/models/gemma4/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
```

O `llama-server` local normalmente responde em:

```text
http://127.0.0.1:8080/v1
```

A configuração estável conhecida é:

```bash
llama-server \
  -m /home/ght/models/gemma4/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
  -ngl 99 \
  -c 65536 \
  -fa on \
  -ctk q4_0 \
  -ctv q4_0 \
  --no-kv-offload \
  --jinja \
  --host 127.0.0.1 \
  --port 8080
```

Use a RX 580 para a inferência local e o Xeon para coleta, cálculos, armazenamento, replay, backtesting e dashboard. Não altere essa configuração sem medir desempenho, VRAM, RAM, CPU e swap.

## Fonte de dados

Comece avaliando a Binance WebSocket pública para ETH/USDT, especialmente o stream de candles de cinco minutos, trades, ticker e profundidade do livro. Não use API de envio de ordens. Coinbase Advanced Trade WebSocket público e Kraken Spot WebSocket v2 público podem ser implementados como adaptadores futuros.

A conexão deverá ser constante, com heartbeat, timeout, backoff exponencial, jitter, reconexão, re-subscription e recuperação de estado. A aplicação deve detectar conexão aparentemente viva, mas sem mensagens recentes.

A coleta deve ser desacoplada da análise. Se o modelo estiver ocupado ou indisponível, o coletor não pode parar de registrar dados.

## Indicadores iniciais

Calcule fora do LLM:

- EMA 9 e EMA 21.
- EMA 200 no timeframe de 1 hora.
- RSI.
- MACD.
- VWAP diário.
- ADX.
- ATR.
- Volume relativo.
- Máxima e mínima do dia anterior.
- Pivôs, suportes, resistências e rompimentos.
- Spread, bid/ask e desequilíbrio do livro quando os dados estiverem disponíveis.

Use candles confirmados. Candles em formação devem ser marcados como `forming` e não podem produzir uma decisão baseada em fechamento.

## Contexto do modelo

O modelo local deve receber um snapshot compacto com:

1. Estado atual do mercado.
2. Últimos candles fechados de 5m, 15m e 1h.
3. Indicadores calculados e sua versão.
4. Regime de mercado.
5. Qualidade, idade e fonte dos dados.
6. Alertas anteriores relevantes.
7. Condições de invalidação.
8. Identificadores dos registros completos armazenados no disco.

Não envie todo o histórico ao prompt. Mantenha o histórico completo no armazenamento local e use resumos versionados no contexto. Não envie screenshots continuamente. Só use imagens se for confirmado que o modelo GGUF e o projector suportam visão multimodal.

## Saída obrigatória do modelo

Exija JSON validável por schema:

```json
{
  "decision": "HOLD",
  "confidence": 0.0,
  "regime": "RANGE",
  "timeframe_alignment": {
    "1h": "NEUTRAL",
    "15m": "NEUTRAL",
    "5m": "NEUTRAL"
  },
  "thesis": "Sem confirmação suficiente.",
  "invalidation": "A hipótese perde validade se os níveis estruturais mudarem.",
  "evidence_ids": [],
  "data_quality": "GOOD",
  "generated_at": "2026-01-01T00:00:00Z"
}
```

Valores permitidos para `decision`: `BUY_BIAS`, `SELL_BIAS`, `HOLD` e `DATA_UNAVAILABLE`. O sistema deve rejeitar JSON inválido, campos desconhecidos críticos, timestamps inválidos, confiança fora do intervalo ou evidências inexistentes.

A confiança não deve ser tratada como probabilidade antes de ser calibrada em dados fora da amostra.

## Requisitos operacionais

O sistema deve conter persistência, reconexão de WebSockets, idempotência, relógio e fusos, tratamento de candles incompletos, autenticação somente quando indispensável para dados privados futuros, testes, observabilidade, controle de exposição conceitual, falha do modelo e proteção contra alertas duplicados.

Mesmo não havendo execução de ordens, idempotência é necessária: o mesmo fechamento de vela não pode gerar múltiplos alertas após reconexão ou reinicialização. Use uma chave composta por fonte, símbolo, timeframe, timestamp de fechamento e versão da estratégia.

Armazene tudo em UTC. Detecte clock drift, dados atrasados, gaps, duplicatas, volume inválido, preço inválido e candles fora de ordem. Em caso de dúvida, use `HOLD` ou `DATA_UNAVAILABLE`.

Use SQLite para estado operacional, alertas e metadados. Use Parquet ou formato equivalente para histórico volumoso. Salve o payload de entrada, hash do snapshot, versão do prompt, resposta bruta, resposta normalizada, latência e resultado posterior.

Implemente logs estruturados sem dados sensíveis, métricas de latência, idade do dado, reconexões, mensagens descartadas, erros do modelo, consumo de CPU/RAM/VRAM e estado de saúde. Crie um endpoint ou comando local para verificar se o sistema está saudável.

Se o `llama-server` cair, ficar lento ou retornar resposta inválida, o monitoramento deve continuar coletando e armazenando dados, mas não deve gerar um alerta direcional novo.

## Arquitetura esperada

Separe os seguintes componentes:

```text
collector
websocket_manager
candle_builder
feature_engine
context_manager
llm_client
schema_validator
alert_engine
storage
health_monitor
local_dashboard
```

A interface da exchange deve ser abstrata. A estratégia e o armazenamento não podem depender de detalhes específicos da Binance.

O serviço deverá ser executável de forma independente do Hyprland, preferencialmente como serviço de usuário do `systemd`, com reinício seguro e sem perder estado.

## O que não fazer

Não implementar envio, cancelamento ou alteração de ordens. Não pedir API key de trading. Não incluir permissões de saque. Não interpretar alertas como garantia de lucro. Não construir um sistema que dependa de screenshots contínuos. Não usar candle aberto como confirmação. Não deixar o modelo decidir sozinho quais dados são verdadeiros. Não instalar pacotes, usar `sudo` ou alterar o sistema sem solicitar autorização.

## Perguntas que devem ser respondidas antes da implementação

Antes de começar a codificar, faça apenas as perguntas que realmente alteram a arquitetura:

1. A fonte inicial deverá ser Binance, Coinbase ou Kraken, considerando a disponibilidade do serviço para minha região?
2. O símbolo inicial será exclusivamente ETH/USDT ou também devemos acompanhar BTC/USDT e ETH/BTC desde o MVP?
3. O alerta será exibido apenas em um dashboard local ou também deverá usar notificações do desktop, terminal ou outro canal?
4. O histórico será coletado a partir do momento da instalação ou já existe uma base local para backtesting?
5. O modelo local responde atualmente a chat completions com JSON estruturado e possui suporte multimodal, ou devemos começar somente com texto e dados numéricos?
6. Qual deve ser o tempo máximo aceitável entre o fechamento da vela e a geração do alerta?

Se alguma resposta não for necessária para iniciar o MVP, assuma a opção mais segura: Binance pública, ETH/USDT, dashboard local, dados numéricos, paper/observação e análise após fechamento de cada vela de cinco minutos.

## Ordem de desenvolvimento

Comece com diagnóstico somente leitura do ambiente e do endpoint do llama-server. Depois implemente a conexão pública, armazenamento e candles. Em seguida implemente indicadores e replay. Só depois integre o modelo e o dashboard.

Cada etapa deve incluir testes e uma forma de verificar o resultado antes de avançar. O primeiro marco é um monitor que permanece conectado, armazena candles de 5m e reconecta corretamente. O segundo é um motor que reproduz indicadores por replay. O terceiro é um assistente que gera JSON explicável sem duplicar alertas. O quarto é um dashboard local com estado, evidências e saúde do sistema.

## Critérios de aceitação

O MVP deve conseguir permanecer em execução, receber dados públicos, construir candles fechados de cinco minutos, identificar eventos atrasados, reconectar, preservar o estado, impedir alertas duplicados, registrar toda decisão e continuar seguro quando o modelo local falhar.

Nenhuma capacidade de operar financeiramente deve existir no MVP. O software será considerado pronto para avaliação somente quando os resultados puderem ser reproduzidos por replay e cada alerta puder ser rastreado até os dados que o originaram.
