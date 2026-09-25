"""System prompt: personality.txt + contexto do sistema + regras das ferramentas."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from core.config import read_personality
from core.tools import default_shell, os_label

WEEKDAYS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]

RULES = """\
## Regras obrigatórias de uso das ferramentas
1. BUSCA NA WEB (Tavily): Sempre que você não tiver informações atualizadas sobre o que o usuário perguntou, ou se tiver qualquer nível de incerteza/alucinação, você DEVE executar a função de busca do Tavily (`tavily_search`) antes de responder. Isso inclui notícias, preços, cotações, versões de software, leis, eventos e qualquer fato posterior ao seu treinamento ou que mude com o tempo. Depois de receber os resultados, responda com base neles e cite as fontes (URLs). Nunca invente fontes.
2. ARQUIVOS: Quando o usuário anexar arquivos, o texto extraído aparece na mensagem dele. Se o Python não conseguiu ler o arquivo nativamente, você DEVE usar `run_terminal_command` para extrair o conteúdo (conversores de linha de comando, PowerShell, etc.) antes de responder. Use `read_file` com `offset` para ler partes que não couberam na mensagem.
3. TERMINAL: Prefira comandos de leitura. Nunca apague, formate, desinstale ou altere configurações do sistema. Se um comando falhar, leia o erro e tente uma alternativa diferente (não repita o mesmo comando).
4. IMAGENS: Você é um modelo de visão. Imagens anexadas pelo usuário chegam junto da mensagem; descreva e analise o que vê com precisão.
5. Nunca invente resultados de ferramentas. Quando já tiver as informações necessárias, pare de chamar ferramentas e responda ao usuário de forma clara."""

PROMPTED_TOOLS = """\
## Ferramentas disponíveis
Você pode chamar as ferramentas abaixo (assinaturas em JSON Schema):
<tools>
{tools}
</tools>

Para chamar uma ferramenta, escreva SOMENTE um ou mais blocos exatamente neste formato e pare:
<tool_call>
{{"name": "nome_da_ferramenta", "arguments": {{"parametro": "valor"}}}}
</tool_call>
O resultado chegará na próxima mensagem dentro de <tool_response></tool_response>. Só então escreva a resposta final ao usuário (sem blocos <tool_call>)."""


def system_context(workspace: Path) -> str:
    now = time.localtime()
    date = f"{WEEKDAYS[now.tm_wday]}, {time.strftime('%d/%m/%Y', now)}"
    return (
        "## Contexto do sistema\n"
        f"- Data de hoje: {date} (início da conversa às {time.strftime('%H:%M', now)}).\n"
        f"- Computador do usuário: {os_label()}; shell padrão do terminal: {default_shell()}.\n"
        f"- Pasta de trabalho (workspace): {workspace}\n"
        f"- Arquivos enviados pelo usuário são salvos em: {workspace / 'uploads'}"
    )


def build_system_prompt(workspace: Path, tools: list[dict[str, Any]], native_tools: bool) -> str:
    parts = [read_personality(), system_context(workspace), RULES]
    if not native_tools:
        schema = "\n".join(json.dumps(t["function"], ensure_ascii=False) for t in tools)
        parts.append(PROMPTED_TOOLS.format(tools=schema))
    return "\n\n".join(parts)
