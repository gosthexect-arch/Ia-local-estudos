# Plano de Ação para Construção da sua IA Local Avançada

Este plano de ação detalha os passos práticos para implementar sua IA local, incorporando os insights do relatório de pesquisa aprofundada e otimizando para seu hardware (GTX 1060 6GB VRAM, 16GB RAM). O foco é em uma arquitetura modular, com modelos especializados e ferramentas determinísticas para automação, manipulação de arquivos, leitura de PDFs, criação de PDFs, cálculos e memória persistente.

## 1. Visão Geral da Arquitetura Recomendada

A estratégia central é um **loop `planner-executor-observer`** onde um LLM (modelo de linguagem grande) planeja as ações, mas a execução é delegada a ferramentas Python determinísticas. Isso garante controle, segurança e eficiência, aproveitando ao máximo seus recursos de hardware [1].

### Componentes Chave:

*   **Servidor de Modelos**: Ollama (para hospedar LLMs localmente).
*   **Orquestrador**: Python (com LangChain/Browser-Use).
*   **Automação de Navegador**: Playwright.
*   **Manipulação de Documentos**: PyPDF2, fpdf2, PyMuPDF, pypdf, pdfplumber, OCRmyPDF + Tesseract (para PDFs escaneados).
*   **Memória/RAG**: `PersistentMemory` (para histórico de conversas) e, opcionalmente, Chroma/Qdrant (para recuperação de informações em documentos).

## 2. Modelos LLM Recomendados para sua GTX 1060 (6GB VRAM)

Para otimizar o uso da sua VRAM, a recomendação é utilizar um conjunto de modelos pequenos e especializados, carregando-os conforme a necessidade [1].

| Papel no sistema | Modelo Sugerido (Ollama) | Tamanho (GB) | Ponto Forte | Quando Usar |
|---|---|---:|---|---|
| **Principal (Planner)** | `Qwen3 4B Instruct` | ~2.5 | Equilíbrio entre qualidade, `tool use`, raciocínio, multilíngue. | Para planejar tarefas complexas, interpretar instruções, orquestrar ferramentas. |
| **Visão (Opcional)** | `Gemma 3 4B` | ~3.3 | Entrada de imagem + texto. | Para analisar screenshots de páginas web, elementos visuais, gráficos. |
| **Código (Opcional)** | `Qwen2.5-Coder 3B` | ~1.9 | Geração e correção de código (Python, JS, etc.). | Para gerar scripts de automação, `locators` Playwright, regexes. |
| **Embeddings (RAG)** | `Qwen3-Embedding 0.6B` | ~0.6 | Leve para indexação e busca semântica. | Para criar representações vetoriais de documentos e links. |

**Ação**: Baixe o `Qwen3 4B Instruct` no Ollama como seu modelo principal. Considere baixar os outros modelos (`Gemma 3 4B`, `Qwen2.5-Coder 3B`, `Qwen3-Embedding 0.6B`) conforme a necessidade de funcionalidades específicas [1].

## 3. Plano de Implementação Detalhado

### Passo 3.1: Preparação do Ambiente Base

1.  **Instalar Ollama**: Baixe e instale o Ollama do site oficial [https://ollama.com/](https://ollama.com/).
2.  **Baixar Modelos LLM**: Abra o terminal e baixe o modelo principal:
    ```bash
    ollama pull qwen3:4b
    # Opcional: ollama pull gemma:4b (para visão), ollama pull qwen2.5-coder:3b (para código)
    ```
3.  **Configurar Ambiente Python**: Crie e ative um ambiente virtual:
    ```bash
    python -m venv ia_local_env
    # No Windows:
    .\ia_local_env\Scripts\activate
    # No Linux/macOS:
    source ia_local_env/bin/activate
    ```
4.  **Instalar Bibliotecas Python Essenciais**: Instale as dependências principais:
    ```bash
    pip install browser-use langchain-ollama playwright PyPDF2 fpdf2 pandas pymupdf pypdf pdfplumber ocrmypdf unstructured
    playwright install
    # Opcional para OCR: pip install ocrmypdf && sudo apt-get install tesseract-ocr (Linux) ou instalar Tesseract para Windows
    ```

### Passo 3.2: Implementação das Ferramentas Personalizadas

Crie um arquivo Python (ex: `tools.py`) para centralizar suas ferramentas. As ferramentas devem ser funções Python decoradas com `@tools.action` do `browser_use`. Lembre-se de que o `browser_use` injeta objetos como `browser_session` e `file_system` nas suas ferramentas [4].

1.  **Ferramenta de Criação de PDF (`create_pdf_file`)**: Utilize `fpdf2` para gerar PDFs a partir de texto.
    ```python
    from fpdf import FPDF
    from browser_use import Tools, ActionResult

    tools = Tools()

    @tools.action(description='Cria um arquivo PDF com o conteúdo fornecido.')
    def create_pdf_file(file_name: str, content: str) -> ActionResult:
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', size=12)
            pdf.multi_cell(0, 10, content.encode('latin-1', 'replace').decode('latin-1'))
            pdf.output(file_name)
            return ActionResult(extracted_content=f'PDF {file_name} criado com sucesso.')
        except Exception as e:
            return ActionResult(extracted_content=f'Erro ao criar PDF {file_name}: {e}')
    ```

2.  **Ferramenta de Cálculos (`calculate`)**: Uma função simples para operações matemáticas.
    ```python
    @tools.action(description='Realiza uma operação matemática e retorna o resultado.')
    def calculate(expression: str) -> ActionResult:
        try:
            result = eval(expression) # Cuidado com eval em produção, mas para uso local é aceitável
            return ActionResult(extracted_content=f'O resultado de {expression} é {result}.')
        except Exception as e:
            return ActionResult(extracted_content=f'Erro ao calcular {expression}: {e}')
    ```

3.  **Ferramenta de Leitura e Análise de PDFs (`read_pdf_content`)**: Use `PyPDF2` para extrair texto. Para PDFs mais complexos ou escaneados, considere integrar `PyMuPDF`, `pdfplumber` e `ocrmypdf` como ferramentas adicionais [1].
    ```python
    import PyPDF2

    @tools.action(description='Lê o conteúdo de um arquivo PDF e retorna o texto.')
    def read_pdf_content(file_path: str) -> ActionResult:
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() or ''
                return ActionResult(extracted_content=f'Conteúdo do PDF {file_path}:\n{text}')
        except Exception as e:
            return ActionResult(extracted_content=f'Erro ao ler PDF {file_path}: {e}')
    ```

### Passo 3.3: Implementação da Memória Persistente

Utilize a classe `PersistentMemory` para gerenciar o histórico de conversas, salvando-o em um arquivo JSON. Isso permite que o agente mantenha o contexto entre as interações [1].

```python
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import json
import os

class PersistentMemory:
    def __init__(self, history_file="agent_history.json"):
        self.history_file = history_file
        self.history = ChatMessageHistory()
        self._load_history()
        self.memory = ConversationBufferMemory(chat_memory=self.history, return_messages=True)

    def _load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for message in data:
                    if message['type'] == 'human':
                        self.history.add_user_message(message['content'])
                    elif message['type'] == 'ai':
                        self.history.add_ai_message(message['content'])

    def save_history(self):
        data = []
        for message in self.history.messages:
            if isinstance(message, HumanMessage):
                data.append({'type': 'human', 'content': message.content})
            elif isinstance(message, AIMessage):
                data.append({'type': 'ai', 'content': message.content})
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def get_memory(self):
        return self.memory
```

### Passo 3.4: Construção do Agente Principal

Integre o Ollama, as ferramentas personalizadas e a memória persistente no seu agente `browser-use`. O `Agent` do `browser-use` aceita um LLM e uma lista de ferramentas [3].

```python
import asyncio
from browser_use import Agent
from langchain_ollama import ChatOllama
# Importe suas ferramentas e a classe PersistentMemory aqui
from tools import tools # Assumindo que suas ferramentas estão em tools.py
from memory_manager import PersistentMemory # Assumindo que PersistentMemory está em memory_manager.py

async def run_advanced_agent(task_description: str, persistent_memory: PersistentMemory) -> str:
    llm = ChatOllama(
        model="qwen3:4b", # Use o modelo principal recomendado
        num_ctx=256000,    # Contexto do modelo, ajuste conforme o modelo e sua VRAM
    )

    agent = Agent(
        task=task_description,
        llm=llm,
        tools=tools, # Passa as ferramentas personalizadas
        memory=persistent_memory.get_memory(), # Passa a memória
        max_actions_per_step=5,
        tool_call_in_content=False,
    )
    print(f"Executando tarefa: {task_description}")
    result = await agent.run(max_steps=20)
    persistent_memory.save_history() # Salva o histórico após cada execução
    return result

async def main():
    persistent_memory = PersistentMemory()

    # Exemplo de uso com as novas funcionalidades
    task1 = "Calcule 123 + 456 e crie um PDF chamado 'calculo_simples.pdf' com o resultado."
    result1 = await run_advanced_agent(task1, persistent_memory)
    print(f"\nResultado da Tarefa 1: {result1}")

    task2 = "Leia o conteúdo do arquivo 'documento_exemplo.pdf' e depois pesquise no Google por 'tendências de IA em 2024' e me diga o título do primeiro resultado."
    # Certifique-se de ter um 'documento_exemplo.pdf' na mesma pasta para este teste
    result2 = await run_advanced_agent(task2, persistent_memory)
    print(f"\nResultado da Tarefa 2: {result2}")

    task3 = "Visite os links 'https://www.openai.com' e 'https://www.anthropic.com'. Para cada um, me diga o título da página e um breve resumo do que se trata."
    result3 = await run_advanced_agent(task3, persistent_memory)
    print(f"\nResultado da Tarefa 3: {result3}")

    task4 = "Qual foi o resultado do cálculo que fizemos anteriormente?"
    result4 = await run_advanced_agent(task4, persistent_memory)
    print(f"\nResultado da Tarefa 4 (Memória): {result4}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Passo 3.5: Teste e Refinamento

*   **Teste Iterativo**: Comece com tarefas simples e aumente a complexidade gradualmente.
*   **Depuração**: Utilize o `Trace Viewer` do Playwright para entender as ações do navegador e depurar problemas.
*   **Otimização de Prompts**: Refine as descrições das suas ferramentas e as instruções para o LLM para melhorar a precisão e a eficiência.

## 4. Próximos Passos e Expansão

*   **Análise Avançada de PDFs**: Integre `PyMuPDF` e `pdfplumber` para extração de dados mais estruturados de PDFs. Para PDFs escaneados, configure `ocrmypdf`.
*   **RAG Completo**: Se a necessidade de consultar grandes volumes de documentos for alta, implemente um sistema RAG com `Chroma` ou `Qdrant Local Mode` para indexar seus documentos e permitir que o LLM os consulte eficientemente.
*   **Interface de Usuário**: Desenvolva uma interface simples com `Gradio` ou `Streamlit` para interagir com seu agente de forma mais amigável.
*   **Guardrails**: Implemente mecanismos para aprovação humana em ações críticas ou destrutivas, conforme sugerido na arquitetura [1].

Este plano fornece um roteiro claro para construir uma IA local poderosa e adaptada às suas necessidades e recursos de hardware.

## 5. Referências

[1] Relatório analítico para desenvolver uma IA local no seu desktop (deep-research-report.md), fornecido pelo usuário.
[2] Reddit. (2025, January 12). *What can i do with 6GB of VRAM ? : r/LocalLLaMA*. Disponível em: [https://www.reddit.com/r/LocalLLaMA/comments/1hzi6ed/what_can_i_do_with_6gb_of_vram/](https://www.reddit.com/r/LocalLLaMA/comments/1hzi6ed/what_can_i_do_with_6gb_of_vram/)
[3] Aleksandar Haber. (2025, January 16). *Install and Run “Browser Use” AI Agents Locally using Ollama*. Disponível em: [https://aleksandarhaber.com/install-and-run-browser-use-ai-agents-locally-using-ollama/](https://aleksandarhaber.com/install-and-run-browser-use-ai-agents-locally-using-ollama/)
[4] Browser-Use Docs. *Add Tools*. Disponível em: [https://docs.browser-use.com/open-source/customize/tools/add](https://docs.browser-use.com/open-source/customize/tools/add)
