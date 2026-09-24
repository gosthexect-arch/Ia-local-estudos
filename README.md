# IA Local — Chat com VLM (llama.cpp) + agente com Tavily e Terminal

Aplicativo de chat **100% local** que roda modelos de visão (VLM) quantizados em GGUF via
`llama-cpp-python`, com uma interface web limpa e um **agente autônomo** que:

- 🌐 **pesquisa na web sozinho** (API do Tavily) quando não tem informação atual ou tem qualquer incerteza;
- 💻 **usa o terminal do seu computador** para ler qualquer formato de arquivo que o Python não leia nativamente;
- 🖼️ **analisa imagens** enviadas no chat (o modelo é obrigatoriamente um VLM).

Tudo começa por um único arquivo: **`start.bat`**.

---

## Início rápido (Windows 10/11)

1. **Baixe um modelo de visão** (dois arquivos GGUF: o modelo e o `mmproj`) e coloque na pasta `models/`.
   Veja sugestões em [`models/LEIA-ME.txt`](models/LEIA-ME.txt) — ex.: `Qwen3-VL-4B-Instruct-Q4_K_M.gguf` +
   `mmproj-Qwen3-VL-4B-Instruct-F16.gguf`.
2. Crie uma chave gratuita do Tavily em <https://app.tavily.com>.
3. Dê dois cliques em **`start.bat`**. Na primeira vez ele pede a chave do Tavily, instala tudo e abre o
   navegador em `http://127.0.0.1:7860`. Nas próximas vezes, só abre o app.

> Requisitos: internet na primeira execução. Se não houver Python 3.10–3.13, o `start.bat` tenta instalar o
> Python 3.12 via `winget`. Não é preciso instalar CUDA Toolkit nem Visual Studio.

## O que o `start.bat` faz

| Etapa | Detalhe |
|---|---|
| Chave Tavily | Pedida **somente na primeira instalação** e salva em `.env` (`TAVILY_API_KEY=...`). |
| Python + `.venv` | Encontra Python 3.10–3.13 (ou instala via winget) e cria o ambiente virtual isolado. |
| Scan de hardware | `python -m core.hardware` detecta GPU (NVIDIA via `nvidia-smi`; AMD/Intel via registro do Windows), VRAM, versão de CUDA do driver, *compute capability*, núcleos físicos/lógicos e núcleos P/E de CPUs híbridas. |
| Build do llama.cpp | Escolhe e testa a wheel pré-compilada certa: **CUDA 12.5/12.4/11.8** (NVIDIA, a partir da GTX 10xx) → **Vulkan** (AMD/Intel/NVIDIA antigas) → **CPU**. Se o CUDA Toolkit não estiver instalado, instala só o runtime (`cudart` + `cuBLAS`) via pip. A build só é aceita depois de verificar que o llama.cpp enxerga a GPU. |
| Dependências | `pip install -r requirements.txt` (repetido só quando o arquivo muda). |
| Threads e camadas | Calcula `n_threads` ideal e o máximo de `n_gpu_layers` para cada GGUF em `models/` (gravado em `hardware_profile.json`) e abre a interface. |

## A interface

- **Chat central** com streaming; o raciocínio do modelo (`<think>`) e cada uso de ferramenta aparecem em
  blocos recolhíveis (💭 Raciocínio, 🌐 Pesquisa, 💻 Terminal, 📄 Arquivo).
- **📎 Anexos no próprio campo de mensagem**: imagens vão direto para o modelo de visão; outros arquivos são
  salvos em `workspace/uploads/` e têm o texto extraído (PDF, DOCX, XLSX, PPTX, ODT, EPUB, HTML, CSV, código…).
  Se o Python não conseguir ler o formato, a IA é instruída a usar o terminal.
- **Menu lateral**: seletor de modelo (troca em tempo real), **contexto** (slider — recarrega ao soltar, com
  estimativa ao vivo de VRAM e até quanto contexto cabe 100% na GPU), status do modelo e **Avançado**
  (temperatura, máx. tokens, camadas na GPU manual, precisão do KV cache, visão na GPU, liberar/bloquear o
  terminal, máx. passos do agente).
- Botão **parar** durante a geração e **Nova conversa**.

## Ferramentas do agente

| Ferramenta | O que faz |
|---|---|
| `tavily_search` | Busca na web via Tavily (única API de busca usada). Só a IA chama — não há comando manual. |
| `run_terminal_command` | Executa comandos no PowerShell/cmd (Windows) ou bash, com tempo limite, saída truncada e encerramento da árvore de processos. |
| `read_file` | Lê arquivos com leitores nativos do Python, com paginação (`offset`) para arquivos grandes. |

O system prompt é montado assim: **`personality.txt`** (edite à vontade) + contexto do sistema (data,
SO, pasta de trabalho) + as regras obrigatórias. Entre elas:

> *Sempre que você não tiver informações atualizadas sobre o que o usuário perguntou, ou se tiver qualquer
> nível de incerteza/alucinação, você DEVE executar a função de busca do Tavily antes de responder.*

**Formatos de *tool calling* suportados:** o app usa o chat template do próprio GGUF. Se ele suporta ferramentas
(Qwen2.5/3-VL, Mistral Small, Gemma 4, InternVL…), elas vão no formato nativo do modelo; se não (Gemma 3,
LLaVA, SmolVLM…), são descritas no prompt. A saída é interpretada nos formatos Hermes/Qwen (`<tool_call>`),
XML do Qwen3.5, Mistral (`[TOOL_CALLS]`), Gemma 4, LFM2 (pythonic), Llama 3 e JSON puro.

### Segurança do terminal

A IA executa comandos reais com as permissões do seu usuário. Proteções incluídas:

- bloqueio de comandos destrutivos (formatar/particionar disco, apagar a raiz ou pastas do sistema
  recursivamente, desligar o PC, apagar registro/backups, baixar-e-executar scripts, desativar o antivírus…);
- sem entrada interativa (stdin fechado), tempo limite (padrão 90 s) com *kill* de toda a árvore de processos,
  saída truncada para não estourar a memória nem o contexto;
- cada comando aparece na conversa; a opção **"Permitir que a IA use o terminal"** (menu Avançado) desliga tudo.

A lista de bloqueio é uma proteção extra, **não um sandbox**. Não peça ações arriscadas ao modelo.

## Como o app busca o máximo de tokens/s

- **Offload calculado pelo GGUF real** (`core/offload.py`): bytes exatos de cada camada + KV cache (considera
  GQA, *sliding window* da Gemma, camadas recorrentes de modelos híbridos) + buffer de computação + encoder de
  visão + reserva do driver. Estratégia: tenta **100% na GPU** com KV f16 → com **KV q8_0** (metade da
  memória) → movendo o encoder de visão para a CPU; só então faz offload parcial com o máximo de camadas.
  Se mesmo assim faltar memória na carga, tenta de novo com menos camadas automaticamente.
- **VRAM livre medida na hora** pelo próprio ggml (CUDA/Vulkan/Metal) antes de carregar.
- **Flash Attention automático** (o llama.cpp ativa quando a GPU suporta).
- **Threads = núcleos físicos de performance** (hyper-threading e núcleos E deixam a geração mais lenta).
- **Cache de prefixo multimodal**: entre os passos do agente e entre mensagens, só os tokens novos são
  processados — imagens já codificadas não passam de novo pelo encoder de visão.
- Imagens são reduzidas (padrão 1280 px) antes de ir ao modelo: menos tokens e menos VRAM.

## Estrutura

```
start.bat              # único ponto de entrada (instalação + scan + execução)
requirements.txt       # dependências Python
personality.txt        # personalidade (System Prompt)
app.py                 # interface Gradio
core/
  hardware.py          # scan de hardware, escolha da build, DLLs CUDA (só stdlib)
  gguf_info.py         # leitor de metadados GGUF (só stdlib)
  offload.py           # planejador de n_gpu_layers / KV cache (só stdlib)
  models.py            # descoberta de modelos e pareamento com o mmproj
  engine.py            # llama-cpp-python + handler multimodal com cache de prefixo
  agent.py             # loop agêntico (pensar → ferramenta → observar → responder)
  tool_parser.py       # interpreta tool calls em vários formatos + streaming
  tools.py             # terminal, read_file, Tavily
  file_reader.py       # leitura nativa de arquivos e preparo de imagens
  prompts.py           # system prompt
models/                # seus GGUFs (modelo + mmproj)
workspace/             # pasta de trabalho da IA (anexos em workspace/uploads)
tests/                 # testes (pytest), incluindo integração real com llama.cpp
```

Os arquivos antigos na raiz (`main_agent.py`, `tools.py`, `memory_manager.py` e os `.md` de fases) são os
estudos anteriores com Ollama e não são usados pelo app.

## Configuração avançada (`.env`)

| Variável | Uso |
|---|---|
| `TAVILY_API_KEY` | Chave do Tavily (pedida pelo start.bat). |
| `TAVILY_SEARCH_DEPTH` | `advanced` (padrão, 2 créditos/busca) ou `basic` (1 crédito). |
| `APP_PORT` / `APP_HOST` | Porta/endereço da interface (padrão `7860` / `127.0.0.1`). |
| `IA_BACKEND` | Força a build do llama.cpp, ex.: `cu124`, `vulkan` ou `cpu`. Apague `.venv\llama_backend.txt` para reinstalar. |

As preferências da interface ficam em `settings.json`; logs em `logs/app.log`.

## Solução de problemas

- **"build preferida falhou" / ficou em CPU**: veja as mensagens do `start.bat`. Atualize o driver da GPU e
  apague `.venv\llama_backend.txt` para tentar de novo (ou force com `IA_BACKEND`).
- **Lento / pouca VRAM**: reduza o contexto no slider (a estimativa mostra até onde cabe 100% na GPU), use uma
  quantização menor (Q4_K_M) ou deixe o KV cache em `auto`/`q8_0`.
- **Modelo não aparece**: ele precisa do arquivo `mmproj` na mesma pasta (o app exige VLM). Clique em ↻.
- **Busca web indisponível**: confira `TAVILY_API_KEY` no `.env`.
- Para reinstalar tudo do zero, apague a pasta `.venv` e rode o `start.bat`.

## Testes

```
.venv\Scripts\python -m pip install pytest
.venv\Scripts\python -m pytest tests
```

Os testes de integração usam um Qwen2-VL minúsculo de pesos aleatórios (`tests/fixtures`, gerado com o
conversor oficial do llama.cpp) para validar carga, visão, M-RoPE e que o cache de prefixo produz exatamente
os mesmos logits que um processamento do zero.
