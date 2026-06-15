# Guia para Construção de uma IA Local para Automação de Tarefas e Navegação (Atualizado)

## 1. Introdução

Este guia detalhado tem como objetivo auxiliar na construção de um sistema de Inteligência Artificial (IA) local, capaz de automatizar tarefas, manipular arquivos, ler PDFs e interagir com navegadores web. A solução proposta foca em ferramentas de código aberto e gratuitas, otimizadas para operar em hardware com recursos limitados, como o seu sistema com 16GB de RAM e uma NVIDIA GeForce GTX 1060 de 6GB de VRAM.

## 2. Considerações de Hardware

Seu sistema, equipado com um processador Intel Xeon E5-2650 v4, 16GB de RAM e uma NVIDIA GeForce GTX 1060 (6GB VRAM), é adequado para executar modelos de linguagem grandes (LLMs) de menor porte localmente. A limitação de 6GB de VRAM é um fator crucial, direcionando a escolha para modelos quantizados e eficientes. Ferramentas como **Ollama** e a biblioteca **Browser-Use** em conjunto com **Playwright** são ideais para este cenário, pois permitem a execução de LLMs e a automação de navegador de forma eficiente.

## 3. Componentes Essenciais

Para construir sua IA local, utilizaremos os seguintes componentes:

### 3.1. Ollama: Execução de LLMs Localmente

**Ollama** é uma plataforma que facilita a execução de modelos de linguagem grandes (LLMs) em sua máquina local. Ele simplifica o processo de download, configuração e execução de diversos modelos de código aberto. Para sua GTX 1060 de 6GB, modelos como `qwen2.5:7b` ou `phi-3-mini` são boas opções, pois são projetados para serem executados em GPUs com VRAM limitada [1] [2].

**Instalação do Ollama:**
1.  Acesse o site oficial do Ollama: [https://ollama.com/](https://ollama.com/)
2.  Baixe o instalador para o seu sistema operacional (Windows, Linux, macOS).
3.  Siga as instruções de instalação. É um processo direto de "próximo, próximo, finalizar".

**Download de um Modelo (Exemplo: Qwen2.5:7b):**
Após a instalação do Ollama, abra o terminal ou prompt de comando e execute:
```bash
ollama pull qwen2.5:7b
```
Este comando fará o download do modelo `qwen2.5:7b`. Você pode explorar outros modelos disponíveis no site do Ollama que se encaixem na sua VRAM.

### 3.2. Browser-Use: Automação de Navegador com IA

A biblioteca **Browser-Use** permite que agentes de IA interajam com navegadores web, realizando tarefas como navegação, preenchimento de formulários, extração de informações e cliques em elementos. Ela se integra com LLMs (como os executados via Ollama) para interpretar instruções em linguagem natural e traduzi-las em ações no navegador [3].

### 3.3. Playwright: O Motor de Automação do Navegador

**Playwright** é uma biblioteca Python que permite controlar navegadores (Chromium, Firefox e WebKit) de forma programática. O Browser-Use utiliza o Playwright nos bastidores para executar as ações no navegador. Ele será instalado como uma dependência do Browser-Use [4].

### 3.4. Ambiente Python e Instalação de Bibliotecas

É altamente recomendável criar um ambiente virtual Python para gerenciar as dependências do seu projeto.

**Configuração do Ambiente Python:**
1.  **Crie um ambiente virtual:**
    ```bash
    python -m venv ia_local_env
    ```
2.  **Ative o ambiente virtual:**
    *   No Windows:
        ```bash
        .\ia_local_env\Scripts\activate
        ```
    *   No Linux/macOS:
        ```bash
        source ia_local_env/bin/activate
        ```
3.  **Instale as bibliotecas necessárias:**
    ```bash
    pip install browser-use langchain-ollama playwright PyPDF2 fpdf2 pandas
    playwright install
    ```
    *   `browser-use`: A biblioteca principal para automação com IA.
    *   `langchain-ollama`: Permite a integração do LangChain (usado pelo Browser-Use) com o Ollama.
    *   `playwright`: A biblioteca de automação de navegador. O comando `playwright install` baixa os drivers dos navegadores.
    *   `PyPDF2`: Para leitura de PDFs.
    *   `fpdf2`: Para criação de PDFs.
    *   `pandas`: Para manipulação de dados, útil para análise de conteúdo.

## 4. Funcionalidades Avançadas: Ferramentas Personalizadas e Memória

Para que sua IA possa criar PDFs, analisar múltiplos links, fazer contas e ter memória persistente, precisaremos estender o agente do `browser-use` com **ferramentas personalizadas** e um mecanismo de **memória**.

### 4.1. Ferramentas Personalizadas

O `browser-use` permite adicionar funções Python como ferramentas que o LLM pode chamar. Isso é feito usando o decorador `@tools.action` [5].

#### 4.1.1. Ferramenta para Criação de PDFs

Vamos criar uma ferramenta que gera um arquivo PDF a partir de um texto fornecido.

```python
from fpdf import FPDF
from browser_use import Tools, ActionResult

tools = Tools()

@tools.action(description='Cria um arquivo PDF com o conteúdo fornecido.')
def create_pdf_file(file_name: str, content: str) -> ActionResult:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    pdf.multi_cell(0, 10, content.encode('latin-1', 'replace').decode('latin-1')) # Encoding para caracteres especiais
    pdf.output(file_name)
    return ActionResult(extracted_content=f'PDF {file_name} criado com sucesso.')
```

#### 4.1.2. Ferramenta para Cálculos

Uma ferramenta simples para realizar operações matemáticas.

```python
@tools.action(description='Realiza uma operação matemática e retorna o resultado.')
def calculate(expression: str) -> ActionResult:
    try:
        result = eval(expression) # Cuidado com eval em produção, mas para uso local é aceitável
        return ActionResult(extracted_content=f'O resultado de {expression} é {result}.')
    except Exception as e:
        return ActionResult(extracted_content=f'Erro ao calcular {expression}: {e}')
```

#### 4.1.3. Ferramenta para Leitura e Análise de PDFs

Esta ferramenta lerá o conteúdo de um PDF e o retornará para o LLM analisar.

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

### 4.2. Memória Persistente para o Agente

Para que o agente tenha "memória" e possa lembrar de conversas e informações passadas, podemos integrar módulos de memória do LangChain. Uma abordagem comum é usar `ConversationBufferMemory` ou `ConversationSummaryBufferMemory` com um `ChatMessageHistory` que pode ser salvo e carregado de um arquivo.

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

# Exemplo de uso:
# persistent_memory = PersistentMemory()
# memory = persistent_memory.get_memory()
# # Passe 'memory' para o seu agente LangChain
# # Após cada interação, chame:
# # persistent_memory.save_history()
```

## 5. Exemplo de Código Atualizado: Agente com Ferramentas e Memória

Agora, vamos integrar as ferramentas personalizadas e a memória ao nosso agente `browser-use`.

Crie um arquivo Python (ex: `agente_avancado.py`) com o seguinte conteúdo:

```python
import os
import asyncio
from browser_use import Agent, Tools, ActionResult
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import json
from fpdf import FPDF
import PyPDF2

# --- Ferramentas Personalizadas ---

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

@tools.action(description='Realiza uma operação matemática e retorna o resultado.')
def calculate(expression: str) -> ActionResult:
    try:
        result = eval(expression)
        return ActionResult(extracted_content=f'O resultado de {expression} é {result}.')
    except Exception as e:
        return ActionResult(extracted_content=f'Erro ao calcular {expression}: {e}')

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

# --- Memória Persistente ---

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

# --- Função Principal do Agente ---

async def run_advanced_agent(task_description: str, persistent_memory: PersistentMemory) -> str:
    # Certifique-se de que o servidor Ollama esteja rodando (ollama serve)
    # e que o modelo qwen2.5:7b foi baixado (ollama pull qwen2.5:7b)
    
    # Opcional: Configura o host do Ollama se não estiver no padrão
    # os.environ["OLLAMA_HOST"] = "http://localhost:11434"

    llm = ChatOllama(
        model="qwen2.5:7b", # Use o modelo que você baixou
        num_ctx=32000,     # Contexto do modelo, ajuste conforme necessário
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

    # Exemplo 1: Fazer uma conta e criar um PDF com o resultado
    task1 = "Calcule 150 * 25 e crie um PDF chamado 'resultado_calculo.pdf' com o resultado."
    result1 = await run_advanced_agent(task1, persistent_memory)
    print(f"\nResultado da Tarefa 1: {result1}")

    # Exemplo 2: Ler um PDF e depois pesquisar algo relacionado no navegador
    # Crie um PDF de teste chamado 'documento_teste.pdf' na mesma pasta para este exemplo
    # Ex: Crie um arquivo 'documento_teste.pdf' com o texto 'Inteligência Artificial é o futuro.'
    task2 = "Leia o conteúdo do arquivo 'documento_teste.pdf' e depois pesquise no Google por 'aplicações de inteligência artificial' e me diga o título do primeiro resultado."
    result2 = await run_advanced_agent(task2, persistent_memory)
    print(f"\nResultado da Tarefa 2: {result2}")

    # Exemplo 3: Analisar múltiplos links (o agente usará o browser-use para navegar)
    task3 = "Visite os links 'https://www.google.com' e 'https://www.wikipedia.org'. Me diga o título de cada página e um breve resumo do que elas são."
    result3 = await run_advanced_agent(task3, persistent_memory)
    print(f"\nResultado da Tarefa 3: {result3}")

    # Exemplo 4: Perguntar algo que o agente deveria lembrar de interações anteriores
    task4 = "Qual foi o resultado do cálculo que fizemos anteriormente?"
    result4 = await run_advanced_agent(task4, persistent_memory)
    print(f"\nResultado da Tarefa 4 (Memória): {result4}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Para executar este código:**
1.  Certifique-se de que o Ollama está rodando em segundo plano. Abra um terminal e execute `ollama serve`.
2.  No terminal onde seu ambiente virtual está ativado, execute:
    ```bash
    python agente_avancado.py
    ```
3.  Para o Exemplo 2, crie um arquivo PDF simples chamado `documento_teste.pdf` na mesma pasta do script Python com algum texto para que a ferramenta `read_pdf_content` possa lê-lo.

## 6. Próximos Passos e Customização

Com esta base, você pode expandir as capacidades da sua IA:

*   **Integração com Ferramentas:** Continue adicionando mais ferramentas personalizadas para qualquer funcionalidade que você precise (ex: enviar e-mails, interagir com APIs específicas, etc.).
*   **Agentes Mais Complexos:** Explore frameworks como `CrewAI` ou `LangChain Agents` para construir agentes mais sofisticados que podem planejar e executar múltiplas etapas para atingir um objetivo complexo, utilizando as ferramentas que você definiu.
*   **Modelos Diferentes:** Experimente outros modelos no Ollama, sempre verificando a compatibilidade com sua VRAM. Modelos como `phi-3-mini` ou `gemma:2b` podem ser ainda mais leves.
*   **Interface de Usuário:** Considere adicionar uma interface de usuário simples (com `Gradio` ou `Streamlit`) para interagir com seu agente de forma mais amigável.
*   **Análise de Conteúdo de PDFs Aprofundada:** Para análise mais complexa de PDFs (extração de tabelas, imagens, etc.), você pode explorar bibliotecas como `pdfplumber` ou `camelot-py` e integrá-las como novas ferramentas.

## 7. Referências

[1] Reddit. (2025, January 12). *What can i do with 6GB of VRAM ? : r/LocalLLaMA*. Disponível em: [https://www.reddit.com/r/LocalLLaMA/comments/1hzi6ed/what_can_i_do_with_6gb_of_vram/](https://www.reddit.com/r/LocalLLaMA/comments/1hzi6ed/what_can_i_do_with_6gb_of_vram/)
[2] Medium. (2026, April 2). *Running Local LLMs on a 6GB GPU Laptop — What Actually Works ...*. Disponível em: [https://medium.com/@kundansinghsorout/running-local-llms-on-a-6gb-gpu-laptop-what-actually-works-in-2026-and-what-doesnt-487fda2a604e](https://medium.com/@kundansinghsorout/running-local-llms-on-a-6gb-gpu-laptop-what_actually_works_in_2026_and_what_doesnt_487fda2a604e)
[3] Aleksandar Haber. (2025, January 16). *Install and Run “Browser Use” AI Agents Locally using Ollama*. Disponível em: [https://aleksandarhaber.com/install-and-run-browser-use-ai-agents-locally-using-ollama/](https://aleksandarhaber.com/install-and-run-browser-use-ai-agents-locally-using-ollama/)
[4] Browser-Use Docs. *Supported Models*. Disponível em: [https://docs.browser-use.com/open-source/supported-models](https://docs.browser-use.com/open-source/supported-models)
[5] Browser-Use Docs. *Add Tools*. Disponível em: [https://docs.browser-use.com/open-source/customize/tools/add](https://docs.browser-use.com/open-source/customize/tools/add)
