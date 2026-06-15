# Fase 2: Instalação do Modelo Qwen3 e Configuração do Ollama para Múltiplos Modelos

Nesta fase, você instalará o modelo **Qwen3 4B Instruct** no Ollama, que será o "cérebro" para automação e tarefas gerais. Também abordaremos como o Ollama gerencia múltiplos modelos, permitindo que você alterne entre o **Phi-3.5-medium** (para pesquisa aprofundada) e o **Qwen3 4B** (para automação e tarefas rápidas) conforme a necessidade.

## 1. Instalação do Modelo Qwen3 4B Instruct

Como o Ollama já está instalado (conforme a Fase 1), o processo para baixar o Qwen3 4B é direto:

1.  **Certifique-se de que o servidor Ollama está rodando.** Se você o fechou, abra um novo terminal ou prompt de comando e execute:
    ```bash
    ollama serve
    ```
    Mantenha este terminal aberto em segundo plano, pois ele é o servidor que hospeda seus modelos.

2.  **Abra um *novo* terminal ou prompt de comando** (não o que está rodando `ollama serve`).

3.  **Baixe o modelo Qwen3 4B Instruct**:
    ```bash
    ollama pull qwen3:4b
    ```
    Este comando fará o download da versão quantizada do Qwen3 4B Instruct. Ele é otimizado para rodar em GPUs com VRAM limitada, como a sua GTX 1060, oferecendo um bom equilíbrio entre inteligência e velocidade para tarefas de automação.

    *   **Verificação**: Após o download, você pode listar os modelos instalados com:
        ```bash
        ollama list
        ```
        Você deverá ver `phi3.5:medium` e `qwen3:4b` na lista.

## 2. Gerenciamento de Múltiplos Modelos no Ollama

O Ollama permite que você tenha vários modelos instalados e alterne entre eles facilmente no seu código Python. O servidor Ollama gerencia o carregamento e descarregamento dos modelos na VRAM (ou RAM, se a VRAM for insuficiente) conforme eles são solicitados.

### 2.1. Como o Ollama Lida com a VRAM/RAM

*   Quando você solicita um modelo (ex: `qwen3:4b`) através da API do Ollama (que o `langchain-ollama` utiliza), o Ollama tenta carregá-lo na VRAM da sua GPU.
*   Se a VRAM for insuficiente (como no caso do Phi-3.5-medium, que é maior que 6GB), o Ollama fará o *offloading* para a RAM do sistema e usará a CPU para inferência. Isso é automático e transparente para você, mas resulta em menor velocidade.
*   Para o **Qwen3 4B**, que tem cerca de 2.5GB, ele deve caber confortavelmente na sua GTX 1060 de 6GB, garantindo inferência rápida na GPU.

### 2.2. Alternando Modelos no Código Python

No seu script Python, você simplesmente especifica qual modelo deseja usar ao inicializar o `ChatOllama`:

*   **Para o "Pesquisador Profissional" (Phi-3.5-medium na CPU/RAM):**
    ```python
    from langchain_ollama import ChatOllama

    llm_pesquisador = ChatOllama(
        model="phi3.5:medium",
        num_ctx=32000, # Contexto do modelo, ajuste conforme necessário
    )
    ```

*   **Para o "Automador Rápido" (Qwen3 4B na GPU/VRAM):**
    ```python
    from langchain_ollama import ChatOllama

    llm_automacao = ChatOllama(
        model="qwen3:4b",
        num_ctx=256000, # Contexto do modelo, ajuste conforme necessário
    )
    ```

Você pode ter instâncias diferentes de `ChatOllama` no seu código, cada uma apontando para um modelo diferente. O Ollama se encarregará de carregar e descarregar os modelos conforme eles são chamados, embora possa haver um pequeno atraso na primeira vez que um modelo é carregado ou quando há uma troca de modelo que exige descarregar um e carregar outro.

Com o Qwen3 4B instalado, você terá o modelo de automação pronto para ser integrado ao seu agente na próxima fase.
