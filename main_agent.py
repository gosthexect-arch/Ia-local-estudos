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
        print(f"\n--- Usando LLM Pesquisador (phi3.5:medium) para a tarefa: {task_description} ---")
    elif use_llm_model == "automacao":
        current_llm = llm_automacao
        print(f"\n--- Usando LLM Automação (qwen3:4b) para a tarefa: {task_description} ---")
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
    print(f"\nResultado da Tarefa de Pesquisa: {research_result}")

    # --- Exemplo de Uso: Tarefa de Automação (usando Qwen3 4B) ---
    # O Qwen3 4B é rápido e eficiente para executar ferramentas e automação de navegador.
    automation_task = "Vá para o Google, pesquise por \"melhores práticas de IA local 2024\", e crie um PDF chamado \"melhores_praticas_ia_local.pdf\" com o título e o URL do primeiro resultado da pesquisa. Depois, calcule 150 * 25 e adicione o resultado ao PDF."
    automation_result = await run_orchestrator_agent(automation_task, "automacao", persistent_memory)
    print(f"\nResultado da Tarefa de Automação: {automation_result}")

    # --- Exemplo de Uso: Leitura de PDF e Memória (usando Phi-3.5-medium para análise) ---
    # Primeiro, crie um PDF de teste para este exemplo (ex: documento_teste.pdf com conteúdo relevante)
    # O agente usará a ferramenta read_pdf_content e o LLM pesquisador para analisar.
    pdf_analysis_task = "Leia o conteúdo do arquivo PDF chamado \"melhores_praticas_ia_local.pdf\" que você criou e me diga qual foi o resultado do cálculo que fizemos anteriormente. Depois, resuma o conteúdo do PDF em 3 frases."
    pdf_analysis_result = await run_orchestrator_agent(pdf_analysis_task, "pesquisador", persistent_memory)
    print(f"\nResultado da Análise de PDF e Memória: {pdf_analysis_result}")

    # --- Exemplo de Uso: Tarefa de Navegação Simples (usando Qwen3 4B) ---
    navigation_task = "Navegue para https://www.wikipedia.org/ e me diga o título da página."
    navigation_result = await run_orchestrator_agent(navigation_task, "automacao", persistent_memory)
    print(f"\nResultado da Navegação Simples: {navigation_result}")

if __name__ == "__main__":
    asyncio.run(main())
