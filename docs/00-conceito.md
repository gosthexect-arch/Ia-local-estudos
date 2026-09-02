# Assistente Local de Monitoramento Contínuo de Criptomoedas

## Documento de conceito, fundamentos e ambiente do projeto

**Versão:** 2.0  
**Autor:** Manus AI  
**Escopo atual:** monitoramento e análise em tempo real, sem execução de ordens.

> **Aviso financeiro:** este documento descreve uma arquitetura de software e não constitui recomendação de investimento, promessa de rentabilidade ou aconselhamento financeiro. Criptoativos são voláteis e podem causar perdas significativas. O sistema deve começar em modo de observação e simulação, sem chaves de operação e sem capacidade de enviar ordens.

## 1. Visão geral

A proposta é desenvolver um assistente local capaz de acompanhar continuamente o mercado de criptomoedas, especialmente Ethereum, receber dados em tempo real, calcular indicadores técnicos, interpretar o estado atual do mercado e produzir alertas explicáveis para o usuário.

O assistente não será um robô de execução financeira. Ele não comprará, venderá, cancelará ordens, movimentará fundos nem terá acesso a permissões de negociação. Sua função será observar, organizar evidências, avaliar cenários e informar possíveis condições de mercado, sempre deixando a decisão final com o usuário.

A expressão “ver constantemente os gráficos” deve ser implementada como uma conexão contínua a dados de mercado, não como o envio incessante de screenshots ao modelo. O sistema receberá eventos de WebSocket, construirá candles e indicadores localmente, manterá um estado atualizado e acionará o modelo em intervalos relevantes, principalmente no fechamento de cada vela de cinco minutos ou quando ocorrer uma mudança significativa no mercado.

## 2. Ideia central

O projeto combina três componentes que normalmente aparecem separados: aquisição contínua de dados, análise determinística de mercado e interpretação por um modelo de linguagem local.

A aquisição fornece os fatos. O motor quantitativo calcula os indicadores de forma reproduzível. O modelo local interpreta a combinação de fatores, explica conflitos entre timeframes e transforma o estado técnico em um alerta estruturado. Essa separação evita que o modelo invente valores, interprete pixels de maneira inconsistente ou tome decisões com base em candles ainda incompletos.

O sistema deverá responder perguntas como:

> O mercado está em tendência ou em consolidação? O ETH está alinhado com BTC e ETH/BTC? O movimento atual possui confirmação de volume e momentum? Quais níveis invalidariam a hipótese? Há dados suficientes para emitir um alerta ou o comportamento correto é aguardar?

A resposta não deverá ser uma ordem financeira. Deverá ser um relatório compacto, versionado e auditável, com classificação como `BUY_BIAS`, `SELL_BIAS` ou `HOLD`, nível de confiança calibrado posteriormente, evidências utilizadas, riscos observados e condições de invalidação.

## 3. Fundamentos técnicos

### 3.1 Dados de mercado em tempo real

A primeira fonte será uma API pública de dados da exchange por WebSocket. Não serão necessárias chaves privadas para monitorar candles, trades, ticker e livro de ofertas. A Binance WebSocket pública é a primeira candidata para o MVP, desde que esteja disponível para o usuário; Coinbase Advanced Trade e Kraken Spot WebSocket são alternativas.

O foco inicial será ETH/USDT com candles de cinco minutos. A conexão deverá permanecer ativa e possuir heartbeat, timeout, reconexão com backoff, detecção de lacunas, validação de sequência e recuperação por REST quando necessário. O sistema deve distinguir a atualização de uma vela em formação do evento de vela fechada.

Dados adicionais poderão incluir BTC/USDT, ETH/BTC, trades recentes, bid/ask, spread e profundidade do livro. DXY e TOTAL3 serão fontes complementares futuras, pois exigem uma estratégia de aquisição e normalização diferente da utilizada para uma exchange cripto.

### 3.2 Indicadores e estrutura de mercado

Os indicadores serão calculados no próprio computador, fora do modelo de linguagem. A base inicial inclui EMA 9 e EMA 21 para direção de curto prazo, EMA 200 no timeframe de uma hora para viés superior, RSI e divergências para momentum, MACD para confirmação, VWAP para preço justo intradiário, ADX para regime, ATR para volatilidade, volume relativo e níveis estruturais.

A análise deverá considerar candles de 5 minutos como unidade principal, além de 15 minutos e 1 hora para alinhamento entre períodos. O sistema poderá acompanhar 1 minuto para microestrutura, mas não deve transformar cada oscilação em alerta. O fechamento de vela e os eventos de regime devem ser os principais gatilhos.

Suportes, resistências, máxima e mínima do dia anterior, pivôs, rompimentos, rejeições, liquidity sweeps e zonas de interesse devem ser registrados como hipóteses calculáveis. Termos como “order block” precisam ter uma definição operacional no código para não depender de interpretação subjetiva do modelo.

### 3.3 Modelo local

A análise será realizada pelo `llama-server` local, usando o modelo GGUF já disponível. O servidor atende em `http://127.0.0.1:8080/v1` e possui uma configuração Vulkan validada na RX 580.

O modelo receberá dados compactados e estruturados, não um histórico ilimitado. O contexto terá quatro camadas: estado atual, janela recente, resumo histórico e referências para a auditoria completa armazenada no disco. O modelo deverá responder em JSON validável por schema.

A inferência não ocorrerá necessariamente a cada tick. O coletor e o motor de indicadores trabalharão continuamente, mas o modelo será acionado no fechamento de uma vela de cinco minutos, em mudança de regime, rompimento estrutural, anomalia de volume, divergência relevante ou alteração importante do estado do mercado.

## 4. Diferencial do projeto

O diferencial principal é a combinação entre **monitoramento persistente, processamento local, contexto controlado e auditabilidade**.

A maioria das soluções baseadas em IA tenta enviar uma imagem de gráfico para um modelo e pedir uma previsão. Este projeto trata o gráfico como uma representação visual de dados estruturados. O sistema guarda os eventos originais, calcula os indicadores de forma determinística e usa o modelo apenas para interpretar relações e explicar cenários.

Outro diferencial é a privacidade. O processamento pode ocorrer localmente, usando a RX 580 para inferência, sem enviar o histórico de mercado, as hipóteses ou as configurações do usuário para um serviço externo. Como não haverá chave de operação, o sistema também reduz drasticamente o impacto de um erro do modelo ou de uma falha de segurança.

O terceiro diferencial é o contexto evolutivo. O assistente não receberá somente o último candle; ele terá uma memória operacional resumida, com hipóteses anteriores, invalidações, mudanças de regime e resultados posteriores. Essa memória será rastreável, e não uma acumulação invisível de texto.

O quarto diferencial é a separação entre interpretação e autoridade. O modelo pode sugerir um viés ou explicar um conflito, mas não poderá alterar as regras do sistema, inventar dados ausentes ou executar operações. Essa limitação torna o projeto mais seguro e mais fácil de testar.

## 5. Ambiente computacional

O desenvolvimento será feito em Arch Linux, em boot único, com Hyprland, DankMaterialShell, Wayland, Ghostty, WezTerm, Kitty e Wofi. O projeto não deve depender da interface gráfica para funcionar.

O hardware conhecido é:

| Componente | Uso no projeto |
|---|---|
| Intel Xeon E5-2650 v4 | Coleta assíncrona, cálculos, persistência, backtesting, replay e dashboard |
| AMD Radeon RX 580 2048SP com amdgpu/RADV/Vulkan | Inferência local do modelo GGUF via llama.cpp |
| Aproximadamente 15 GiB de RAM | Buffers, banco local, contexto e serviços auxiliares |
| Aproximadamente 23 GiB de swap | Contingência; não deve ser tratado como memória de alto desempenho |
| Hyprland/Wayland | Interface local opcional e notificações |

O modelo está em:

```text
/home/ght/models/gemma4/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
```

A configuração de referência do servidor local é:

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

Essa configuração deve ser considerada um baseline. Antes de qualquer alteração, é preciso medir latência, uso de VRAM, RAM, CPU e swap. O contexto de 65.536 tokens não deve ser preenchido artificialmente; uma janela menor e bem selecionada tende a ser mais útil do que um histórico enorme e ruidoso.

## 6. Arquitetura do ambiente a desenvolver

```text
API pública WebSocket da exchange
          │
          ▼
Coletor contínuo ── reconexão ── heartbeat ── validação
          │
          ▼
Candles confirmados e armazenamento bruto
          │
          ▼
Motor determinístico de indicadores e eventos
          │
          ▼
Gerenciador de contexto e memória operacional
          │
          ├──────────────► Banco local e auditoria
          │
          ▼
Cliente do llama-server local
          │
          ▼
Validador de JSON e classificador de alerta
          │
          ▼
Dashboard, logs, notificações e histórico de decisões
```

O projeto deve ter módulos independentes para coletor, candles, indicadores, contexto, cliente LLM, schema, persistência, alertas e interface. O código de exchange não deve ficar misturado à estratégia. Assim, uma nova fonte de dados poderá ser adicionada sem reescrever o restante do sistema.

## 7. Persistência, confiabilidade e segurança

O sistema deverá persistir candles, eventos relevantes, snapshots de features, respostas do modelo, alertas, erros, métricas e estado de saúde. SQLite é adequado para o estado operacional; Parquet ou formato equivalente pode armazenar históricos maiores.

Cada evento deve conter identificador, timestamp UTC, símbolo, timeframe, fonte, versão do algoritmo e qualidade do dado. Toda resposta do modelo deve guardar o hash do snapshot de entrada e a versão do prompt/schema usados.

A conexão WebSocket deve detectar desconexões silenciosas, atraso, mensagens duplicadas e gaps. Após reconectar, o sistema deverá recuperar dados por REST, reconstruir candles e somente então liberar novas análises.

Idempotência é necessária mesmo sem execução de ordens. Um evento ou fechamento de vela não pode gerar o mesmo alerta diversas vezes por causa de reconexão, reprocessamento ou reinício do serviço. O identificador deve ser composto por fonte, símbolo, timeframe e instante de fechamento, com controle de versão.

Os horários serão armazenados em UTC. Candles em formação serão marcados como `forming`; somente candles `closed` poderão alimentar decisões de fechamento. Dados atrasados, inválidos ou inconsistentes devem congelar o alerta e aparecer no motivo do estado `HOLD`.

O programa deverá operar sem credenciais. Qualquer variável de ambiente futura deve ser mantida fora do Git, dos prompts e dos logs. Não haverá permissões de saque ou negociação porque o escopo não inclui operação financeira.

## 8. Tratamento de falhas do modelo

O modelo pode estar indisponível, ocupado, lento ou retornar JSON inválido. Em todos esses casos, o sistema deverá registrar o incidente e gerar um estado explícito como `MODEL_UNAVAILABLE`, `INVALID_RESPONSE` ou `STALE_ANALYSIS`. O comportamento seguro será não emitir novo alerta operacional até que o sistema esteja consistente.

O modelo não pode criar indicadores, substituir valores ausentes por suposições ou transformar texto livre em comando. O cliente deve validar tipos, campos obrigatórios, limites numéricos e enumerações antes de aceitar a resposta.

## 9. Alertas e interface

A interface local deverá mostrar o preço atual, a idade do último dado, o estado da conexão, o candle atual, indicadores, regime, último alerta, confiança calibrada, evidências, invalidação, falhas recentes e utilização do hardware.

Os alertas podem aparecer no dashboard, no terminal, por notificação local do desktop e, futuramente, em um canal externo. Cada alerta deve informar quando foi gerado, qual candle estava fechado, quais dados foram utilizados e por que o sistema escolheu `HOLD`, `BUY_BIAS` ou `SELL_BIAS`.

A interface não deve sugerir que o alerta é uma garantia ou ordem. Deve exibir claramente que se trata de análise automatizada e que a decisão financeira pertence ao usuário.

## 10. Desenvolvimento por fases

A primeira fase deve apenas diagnosticar o ambiente, testar o endpoint local do llama-server e confirmar a capacidade de conexão com a API pública escolhida. Nenhuma instalação, alteração do sistema ou configuração importante deve ocorrer sem autorização.

A segunda fase implementará o coletor WebSocket para ETH/USDT, armazenamento, reconexão, heartbeat e candles de cinco minutos. O resultado deverá ser reproduzível por replay.

A terceira fase implementará indicadores, eventos de mercado, estado de qualidade e dashboard mínimo sem LLM. O sistema deve primeiro funcionar como monitor quantitativo determinístico.

A quarta fase integrará o modelo local, o contexto compacto, o schema JSON, a memória operacional e os alertas explicáveis.

A quinta fase implementará backtesting, walk-forward, replay em tempo real e testes de falhas. O projeto deverá comparar a análise do modelo com regras simples e registrar falsos positivos, falsos negativos, atraso e estabilidade.

## 11. Critérios de sucesso

O projeto será considerado tecnicamente sólido quando permanecer conectado por longos períodos, recuperar-se de falhas, não duplicar alertas, rejeitar candles incompletos, identificar dados atrasados, reproduzir uma sessão histórica e explicar cada alerta a partir de registros persistidos.

A qualidade não deverá ser medida somente pela quantidade de alertas ou por uma suposta taxa de acerto. Também serão avaliados atraso, estabilidade, cobertura de dados, qualidade das evidências, comportamento em períodos de alta volatilidade, falsos sinais e capacidade de permanecer em `HOLD` quando não houver confirmação.

## 12. Limites explícitos do escopo

Este projeto não terá execução de ordens, integração com saldo, cancelamento de ordens, gestão de posição real, acesso a saque, alavancagem ou automação financeira. A API será usada apenas para monitorar o mercado.

O software também não deverá apresentar previsões como certezas. A saída será uma avaliação de cenário baseada nos dados disponíveis naquele momento, com limitações e condições de invalidação.

## Referências técnicas

[1]: https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/ws-streams/~ "Binance — Spot WebSocket Market Streams"

[2]: https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/websocket/websocket-overview "Coinbase — Advanced Trade WebSocket Overview"

[3]: https://docs.kraken.com/exchange/api-reference/spot-websocket-v2/ticker "Kraken — Spot WebSocket v2 Ticker"

[4]: https://www.tradingview.com/support/solutions/43000529348-how-to-configure-webhook-alerts/ "TradingView — Webhook Alerts"

[5]: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md "llama.cpp — LLaMA.cpp HTTP Server"
