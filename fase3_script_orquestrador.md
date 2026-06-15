# Fase 3: Criação do Script Orquestrador que Utiliza Ambos os Modelos

Nesta fase, você construirá o script Python principal que orquestra a interação entre os dois modelos LLM (Phi-3.5-medium para pesquisa e Qwen3 4B para automação), as ferramentas personalizadas e a memória persistente. Este script será o "cérebro" do seu projeto, decidindo qual modelo usar e quais ferramentas acionar com base na tarefa fornecida.

## 1. Estrutura do Projeto

Para manter o projeto organizado, sugere-se a seguinte estrutura de arquivos:

```
/seu_projeto/
├── main_agent.py         # Script principal que orquestra os modelos e tarefas
├── tools.py              # Definição das ferramentas personalizadas (PDF, cálculo, etc.)
├── memory_manager.py     # Classe para gerenciar a memória persistente do agente
├── agent_history.json    # Arquivo onde a memória do agente será salva (criado automaticamente)
└── documento_teste.pdf   # Exemplo de PDF para testes (crie este arquivo manualmente)
```

## 2. Implementação das Ferramentas Personalizadas (`tools.py`)

Este arquivo contém as funções Python que seu agente poderá chamar para realizar ações específicas. Elas são decoradas com `@tools.action` do `browser_use`.

Crie um arquivo chamado `tools.py` na raiz do seu projeto com o seguinte conteúdo:

```python
from fpdf import FPDF
from browser_use import Tools, ActionResult, BrowserSession
import PyPDF2
import os

tools = Tools()

@tools.action(description=\'Cria um arquivo PDF com o conteúdo fornecido. Requer o nome do arquivo e o conteúdo.\')
def create_pdf_file(file_name: str, content: str) -> ActionResult:
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font(\'Arial\', size=12)
        # Certifique-se de que o conteúdo é uma string e lida com caracteres especiais
        pdf.multi_cell(0, 10, content.encode(\'latin-1\', \'replace\').decode(\'latin-1\'))
        pdf.output(file_name)
        return ActionResult(extracted_content=f\'PDF {file_name} criado com sucesso.\')
    except Exception as e:
        return ActionResult(extracted_content=f\'Erro ao criar PDF {file_name}: {e}\')

@tools.action(description=\'Realiza uma operação matemática e retorna o resultado. Requer a expressão matemática como string.\')
def calculate(expression: str) -> ActionResult:
    try:
        result = eval(expression) # Cuidado com eval em produção, mas para uso local é aceitável
        return ActionResult(extracted_content=f\'O resultado de {expression} é {result}.\')
    except Exception as e:
        return ActionResult(extracted_content=f\'Erro ao calcular {expression}: {e}\')

@tools.action(description=\'Lê o conteúdo de um arquivo PDF e retorna o texto. Requer o caminho completo do arquivo PDF.\')
def read_pdf_content(file_path: str) -> ActionResult:
    try:
        if not os.path.exists(file_path):
            return ActionResult(extracted_content=f\'Erro: Arquivo PDF não encontrado em {file_path}\' )
        with open(file_path, \'rb\') as file:
            reader = PyPDF2.PdfReader(file)
            text = \'\'
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() or \'\'
            return ActionResult(extracted_content=f\'Conteúdo do PDF {file_path}:\\n{text}\' )
    except Exception as e:
        return ActionResult(extracted_content=f\'Erro ao ler PDF {file_path}: {e}\' )

# Exemplo de ferramenta para navegar e extrair informações (usando browser_session)
@tools.action(description=\'Navega para uma URL e extrai o título da página. Requer a URL.\')
async def navigate_and_get_title(browser_session: BrowserSession, url: str) -> ActionResult:
    try:
        page = await browser_session.must_get_current_page()
        await page.goto(url)
        title = await page.title()
        return ActionResult(extracted_content=f\'Navegou para {url}. Título da página: {title}\' )
    except Exception as e:
        return ActionResult(extracted_content=f\'Erro ao navegar ou obter título de {url}: {e}\' )
```

## 3. Implementação do Gerenciador de Memória Persistente (`memory_manager.py`)

Este arquivo contém a classe `PersistentMemory` que permite ao seu agente lembrar de interações passadas, salvando e carregando o histórico de conversas em um arquivo JSON.

Crie um arquivo chamado `memory_manager.py` na raiz do seu projeto com o seguinte conteúdo:

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
            with open(self.history_file, \'r\', encoding=\'utf-8\') as f:
                data = json.load(f)
                for message in data:
                    if message[\'type\'] == \'human\':
                        self.history.add_user_message(message[\'content\'])
                    elif message[\'type\'] == \'ai\':
                        self.history.add_ai_message(message[\'content\'])

    def save_history(self):
        data = []
        for message in self.history.messages:
            if isinstance(message, HumanMessage):
                data.append({\'type\': \'human\', \'content\': message.content})
            elif isinstance(message, AIMessage):
                data.append({\'type\': \'ai\', \'content\': message.content})
        with open(self.history_file, \'w\', encoding=\'utf-8\') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def get_memory(self):
        return self.memory
```

## 4. Criação do Script Principal (`main_agent.py`)

Este é o coração do seu projeto. Ele define as instâncias dos dois LLMs (Phi-3.5 e Qwen3), carrega as ferramentas e a memória, e contém a lógica para orquestrar as tarefas.

Crie um arquivo chamado `main_agent.py` na raiz do seu projeto com o seguinte conteúdo:

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

# --- Importar Ferramentas Personalizadas ---
# Certifique-se de que o arquivo tools.py está na mesma pasta
from tools import tools

# --- Importar Gerenciador de Memória Persistente ---
# Certifique-se de que o arquivo memory_manager.py está na mesma pasta
from memory_manager import PersistentMemory

# --- Configuração dos Modelos LLM --- 
# Instâncias de ChatOllama para cada modelo

# Modelo para Pesquisa Aprofundada (Phi-3.5-medium na CPU/RAM)
# Ideal para tarefas que exigem raciocínio complexo e análise de conteúdo extenso.
llm_pesquisador = ChatOllama(
    model="phi3.5:medium", 
    num_ctx=32000, # Contexto do modelo, Phi-3.5 suporta contextos grandes
    # Ollama automaticamente fará o offloading para RAM se a VRAM for insuficiente
)

# Modelo para Automação Rápida (Qwen3 4B na GPU/VRAM)
# Ideal para tarefas que exigem respostas rápidas e execução de ferramentas.
llm_automacao = ChatOllama(
    model="qwen3:4b", 
    num_ctx=256000, # Contexto do modelo, Qwen3 4B tem contexto grande
    # Este modelo deve caber na sua GTX 1060 (6GB VRAM) para inferência rápida
)

# --- Função Principal do Agente Orquestrador ---

async def run_orchestrator_agent(task_description: str, use_llm_model: str, persistent_memory: PersistentMemory) -> str:
    # Certifique-se de que o servidor Ollama está rodando (ollama serve)

    current_llm = None
    if use_llm_model == "pesquisador":
        current_llm = llm_pesquisador
        print(f"\\n--- Usando LLM Pesquisador (phi3.5:medium) para a tarefa: {task_description} ---")
    elif use_llm_model == "automacao":
        current_llm = llm_automacao
        print(f"\\n--- Usando LLM Automação (qwen3:4b) para a tarefa: {task_description} ---")
    else:
        return "Erro: Modelo LLM especificado inválido. Use \"pesquisador\" ou \"automacao\"."

    agent = Agent(
        task=task_description,
        llm=current_llm,
        tools=tools, # Passa as ferramentas personalizadas importadas de tools.py
        memory=persistent_memory.get_memory(), # Passa a memória persistente
        max_actions_per_step=10, # Aumentado para permitir mais passos em tarefas complexas
        tool_call_in_content=False,
    )
    
    result = await agent.run(max_steps=30) # Aumentado para permitir mais passos
    persistent_memory.save_history() # Salva o histórico após cada execução
    return result

async def main():
    persistent_memory = PersistentMemory()

    print("Iniciando o agente orquestrador...")
    print("Certifique-se de que o servidor Ollama está rodando em segundo plano (ollama serve).")

    # --- Exemplo de Uso: Tarefa de Pesquisa (usando Phi-3.5-medium) ---
    # O Phi-3.5-medium é ideal para entender e sintetizar informações complexas.
    research_task = "Explique em detalhes o conceito de \"Memória de GPU Compartilhada\" e suas implicações para o desempenho de LLMs em GPUs com VRAM limitada. Use o navegador para pesquisar se necessário."
    research_result = await run_orchestrator_agent(research_task, "pesquisador", persistent_memory)
    print(f"\\nResultado da Tarefa de Pesquisa: {research_result}")

    # --- Exemplo de Uso: Tarefa de Automação (usando Qwen3 4B) ---
    # O Qwen3 4B é rápido e eficiente para executar ferramentas e automação de navegador.
    automation_task = "Vá para o Google, pesquise por \"melhores práticas de IA local 2024\", e crie um PDF chamado \"melhores_praticas_ia_local.pdf\" com o título e o URL do primeiro resultado da pesquisa. Depois, calcule 150 * 25 e adicione o resultado ao PDF."
    automation_result = await run_orchestrator_agent(automation_task, "automacao", persistent_memory)
    print(f"\\nResultado da Tarefa de Automação: {automation_result}")

    # --- Exemplo de Uso: Leitura de PDF e Memória (usando Phi-3.5-medium para análise) ---
    # Primeiro, crie um PDF de teste para este exemplo (ex: documento_teste.pdf com conteúdo relevante)
    # O agente usará a ferramenta read_pdf_content e o LLM pesquisador para analisar.
    pdf_analysis_task = "Leia o conteúdo do arquivo PDF chamado \"melhores_praticas_ia_local.pdf\" que você criou e me diga qual foi o resultado do cálculo que fizemos anteriormente. Depois, resuma o conteúdo do PDF em 3 frases."
    pdf_analysis_result = await run_orchestrator_agent(pdf_analysis_task, "pesquisador", persistent_memory)
    print(f"\\nResultado da Análise de PDF e Memória: {pdf_analysis_result}")

    # --- Exemplo de Uso: Tarefa de Navegação Simples (usando Qwen3 4B) ---
    navigation_task = "Navegue para https://www.wikipedia.org/ e me diga o título da página."
    navigation_result = await run_orchestrator_agent(navigation_task, "automacao", persistent_memory)
    print(f"\\nResultado da Navegação Simples: {navigation_result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 5. Próximos Passos

Com todos os arquivos criados, a próxima fase será a execução e teste do seu agente orquestrador. Você precisará:

1.  **Garantir que o servidor Ollama esteja ativo** (`ollama serve`).
2.  **Criar um arquivo PDF de teste** (`documento_teste.pdf`) na mesma pasta do projeto para testar a funcionalidade de leitura de PDF.
3.  **Executar o `main_agent.py`** e observar o comportamento do seu agente.
