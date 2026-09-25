"""Ferramentas que a IA pode chamar (function calling).

1. run_terminal_command — executa comandos no terminal do sistema (leitura
   universal de arquivos via conversores de linha de comando etc.).
2. read_file            — leitura nativa em Python, com paginação.
3. tavily_search        — busca na web via API do Tavily (só a IA chama).

Nenhuma ferramenta lança exceção para o agente: erros viram texto que o modelo
lê e usa para corrigir a próxima tentativa.
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.config import WORKSPACE_DIR, tavily_api_key
from core.file_reader import UnreadableFile, extract_text, human_size

IS_WINDOWS = sys.platform == "win32"
_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0
MAX_COMMAND_CHARS = 8000
_HEAD_BYTES, _TAIL_BYTES = 256 * 1024, 64 * 1024


@dataclass
class ToolContext:
    terminal_enabled: bool = True
    terminal_timeout: int = 90
    max_output_chars: int = 12000
    workspace: Path = WORKSPACE_DIR
    cancel: threading.Event = field(default_factory=threading.Event)


@dataclass
class ToolResult:
    ok: bool
    content: str          # texto devolvido ao modelo
    summary: str          # linha curta para a UI
    duration: float = 0.0


def default_shell() -> str:
    return "powershell" if IS_WINDOWS else "bash"


def os_label() -> str:
    if IS_WINDOWS:
        build = sys.getwindowsversion().build  # type: ignore[attr-defined]
        return f"Windows {'11' if build >= 22000 else '10'}"
    return "macOS" if sys.platform == "darwin" else "Linux"


def tool_schemas() -> list[dict[str, Any]]:
    shells = ["powershell", "cmd"] if IS_WINDOWS else ["bash", "sh"]
    return [
        {
            "type": "function",
            "function": {
                "name": "tavily_search",
                "description": (
                    "Busca na internet em tempo real (API Tavily). Use SEMPRE que a pergunta envolver informação "
                    "recente/atual, dados que mudam com o tempo (preços, versões, notícias, eventos, pessoas, "
                    "estatísticas) ou quando houver QUALQUER incerteza ou risco de inventar fatos. Retorna um resumo "
                    "e as fontes (título, URL, trecho)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Consulta de busca objetiva e específica."},
                        "topic": {"type": "string", "enum": ["general", "news", "finance"],
                                  "description": "Use 'news' para notícias/acontecimentos recentes."},
                        "time_range": {"type": "string", "enum": ["day", "week", "month", "year"],
                                       "description": "Limita os resultados ao período (opcional)."},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 10,
                                        "description": "Quantidade de fontes (padrão 5)."},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_terminal_command",
                "description": (
                    f"Executa um comando no terminal do computador do usuário ({os_label()}; shell padrão: "
                    f"{default_shell()}) e retorna a saída (stdout/stderr) e o código de saída. Use para ler/extrair "
                    "o conteúdo de arquivos que o Python não leu nativamente (ex.: pdftotext, conversores, "
                    "PowerShell/COM do Office, tar, ffprobe), listar pastas e inspecionar o sistema. Comandos não "
                    "podem pedir interação (não há teclado). Nunca execute ações destrutivas."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Comando completo a executar."},
                        "shell": {"type": "string", "enum": shells, "description": f"Padrão: {default_shell()}."},
                        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600,
                                            "description": "Tempo máximo de execução (padrão 90)."},
                        "working_directory": {"type": "string",
                                              "description": "Pasta onde executar (padrão: pasta workspace)."},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Lê o texto de um arquivo local com leitores nativos do Python (txt, código, csv, json, xml, "
                    "html, pdf com texto, docx, xlsx, pptx, odt, epub, rtf, eml, ipynb, lista de zip/tar). Para "
                    "arquivos longos use 'offset' para ler a continuação. Se o formato não for suportado, a "
                    "resposta sugere comandos para run_terminal_command."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Caminho do arquivo (absoluto ou relativo ao workspace)."},
                        "offset": {"type": "integer", "minimum": 0, "description": "Caractere inicial (padrão 0)."},
                        "max_chars": {"type": "integer", "minimum": 200, "description": "Máximo de caracteres a retornar."},
                    },
                    "required": ["path"],
                },
            },
        },
    ]


TOOL_NAMES = {"tavily_search", "run_terminal_command", "read_file"}


def truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n\n[... {omitted} caracteres omitidos ...]\n\n{text[-tail:]}"


def _as_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- terminal
_DANGEROUS: list[tuple[str, str]] = [
    (r"\bformat(\.com)?\s+[a-z]:", "formatar uma unidade de disco"),
    (r"\b(format-volume|clear-disk|initialize-disk|remove-partition|diskpart)\b", "operações de particionamento/disco"),
    (r"\brm\s+(-[^\s]*\s+)*-[^\s]*[rR][^\s]*\s+(-[^\s]+\s+)*(/|~|\*|/\*|[a-z]:[\\/]?)(\s|$)",
     "remoção recursiva da raiz ou da pasta pessoal"),
    (r"\b(rd|rmdir)\b[^\n]*\s/s\b[^\n]*\s['\"]?[a-z]:\\?['\"]?(\s|$)", "remover uma unidade inteira"),
    (r"\bdel\b[^\n]*\s/[sq]\b[^\n]*\s['\"]?[a-z]:\\(\*(\.\*)?)?['\"]?(\s|$)", "apagar a raiz de uma unidade"),
    (r"remove-item\b[^\n]*-recurse[^\n]*\s['\"]?([a-z]:\\?|\\|/|~|\$env:(systemroot|windir|userprofile|"
     r"programfiles|programdata|appdata|localappdata))['\"]?(\s|$)", "remoção recursiva de pasta do sistema/usuário"),
    (r"\b(shutdown|stop-computer|restart-computer|logoff)\b", "desligar/reiniciar o computador"),
    (r"\b(bcdedit|bootrec|bcdboot|systemreset|reagentc)\b", "alterar boot/restaurar o sistema"),
    (r"\breg(\.exe)?\s+delete\s+hk(lm|ey_local_machine)|remove-item\b[^\n]*hklm:", "apagar chaves do registro"),
    (r"\bcipher\s+/w", "apagar espaço livre do disco"),
    (r"\b(vssadmin|wbadmin)\b[^\n]*\bdelete\b", "apagar backups/cópias de sombra"),
    (r"\bmkfs(\.\w+)?\b|\bdd\s+[^\n]*\bof=/dev/", "formatar/sobrescrever disco"),
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "fork bomb"),
    (r"\b(takeown|icacls)\b[^\n]*[a-z]:\\windows", "alterar permissões do Windows"),
    (r"\bnet\s+user\b[^\n]*\s/(add|delete)\b|\bnet\s+localgroup\s+administra", "alterar contas de usuário"),
    (r"\b(iex|invoke-expression)\b[^\n]*\b(iwr|irm|invoke-webrequest|invoke-restmethod|net\.webclient|downloadstring)\b",
     "executar script baixado da internet"),
    (r"\b(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(ba|z|da)?sh\b", "executar script baixado da internet"),
    (r"\bset-mppreference\b[^\n]*-disable|\bsc(\.exe)?\s+(delete|stop)\s+windefend", "desativar o antivírus"),
]


def blocked_reason(command: str) -> str | None:
    low = command.lower()
    for pattern, reason in _DANGEROUS:
        if re.search(pattern, low, re.IGNORECASE):
            return reason
    return None


class _Capture(threading.Thread):
    """Lê um pipe sem limite de tempo, guardando só o começo e o fim (evita estourar a RAM)."""

    def __init__(self, stream: Any):
        super().__init__(daemon=True)
        self.stream = stream
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def run(self) -> None:
        read = getattr(self.stream, "read1", self.stream.read)
        try:
            while True:
                chunk = read(65536)
                if not chunk:
                    break
                self.total += len(chunk)
                room = _HEAD_BYTES - len(self.head)
                if room > 0:
                    self.head += chunk[:room]
                    chunk = chunk[room:]
                if chunk:
                    self.tail += chunk
                    if len(self.tail) > _TAIL_BYTES:
                        del self.tail[: len(self.tail) - _TAIL_BYTES]
        except (OSError, ValueError):
            pass

    def data(self) -> tuple[bytes, int]:
        omitted = self.total - len(self.head) - len(self.tail)
        return bytes(self.head) + bytes(self.tail), max(0, omitted)


def _decode_output(data: bytes) -> str:
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if IS_WINDOWS:
        try:
            import ctypes

            return data.decode(f"cp{ctypes.windll.kernel32.GetOEMCP()}", errors="replace")
        except (LookupError, OSError, AttributeError):
            pass
    return data.decode("latin-1", errors="replace")


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True,
                           timeout=15, creationflags=_NO_WINDOW)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError, ProcessLookupError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _build_command(command: str, shell: str) -> tuple[list[str] | str, str]:
    if IS_WINDOWS:
        if shell == "cmd":
            # /s + aspas externas: o cmd executa o texto literalmente (sem reescapar aspas).
            return f'cmd.exe /d /s /c "chcp 65001 >nul & {command}"', "cmd"
        exe = shutil.which("powershell") or shutil.which("pwsh") or "powershell.exe"
        script = (
            "$ProgressPreference='SilentlyContinue'; $ErrorActionPreference='Continue'; "
            "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; $OutputEncoding=[System.Text.Encoding]::UTF8\n"
            f"{command}\n"
            "if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
        )
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        return [exe, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                "-EncodedCommand", encoded], "powershell"
    exe = shutil.which("bash") if shell != "sh" else None
    return [exe or "/bin/sh", "-c", command], ("bash" if exe else "sh")


def run_terminal_command(ctx: ToolContext, command: str = "", shell: str | None = None,
                         timeout_seconds: Any = None, working_directory: str | None = None,
                         **_ignored: Any) -> ToolResult:
    if not ctx.terminal_enabled:
        return ToolResult(False, "ERRO: o acesso ao terminal está desativado nas configurações do app.",
                          "terminal desativado")
    command = (command or "").strip()
    if not command:
        return ToolResult(False, "ERRO: parâmetro 'command' vazio.", "comando vazio")
    if len(command) > MAX_COMMAND_CHARS:
        return ToolResult(False, f"ERRO: comando longo demais (máx. {MAX_COMMAND_CHARS} caracteres). "
                          "Grave um script em arquivo e execute-o.", "comando longo demais")
    reason = blocked_reason(command)
    if reason:
        return ToolResult(False, f"BLOQUEADO por segurança ({reason}). Este app não permite esse tipo de comando; "
                          "use uma alternativa somente de leitura.", f"bloqueado: {reason}")
    shell = (shell or default_shell()).lower().strip()
    if shell not in (("powershell", "cmd") if IS_WINDOWS else ("bash", "sh")):
        shell = default_shell()
    timeout = _as_int(timeout_seconds, ctx.terminal_timeout, 1, 600)
    cwd = ctx.workspace
    if working_directory:
        cand = Path(os.path.expandvars(os.path.expanduser(working_directory)))
        cand = cand if cand.is_absolute() else ctx.workspace / cand
        if not cand.is_dir():
            return ToolResult(False, f"ERRO: pasta de trabalho inexistente: {cand}", "pasta inexistente")
        cwd = cand
    cwd.mkdir(parents=True, exist_ok=True)

    argv, shell = _build_command(command, shell)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")  # 'python' = Python do app
    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(
            argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, creationflags=_NO_WINDOW, start_new_session=not IS_WINDOWS,
        )
    except (OSError, ValueError) as e:
        return ToolResult(False, f"ERRO ao iniciar o shell '{shell}': {e}", f"falha ao iniciar {shell}")

    out_cap, err_cap = _Capture(proc.stdout), _Capture(proc.stderr)
    out_cap.start()
    err_cap.start()
    status = ""
    while True:
        try:
            proc.wait(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            if ctx.cancel.is_set():
                status = "interrompido pelo usuário"
            elif time.perf_counter() - t0 > timeout:
                status = f"tempo limite de {timeout}s excedido (processo encerrado)"
            if status:
                _kill_tree(proc)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                break
    out_cap.join(timeout=3)
    err_cap.join(timeout=3)
    for pipe in (proc.stdout, proc.stderr):
        try:
            pipe.close()
        except (OSError, AttributeError):
            pass
    elapsed = time.perf_counter() - t0
    out_raw, out_omitted = out_cap.data()
    err_raw, err_omitted = err_cap.data()
    stdout = truncate_middle(_decode_output(out_raw), ctx.max_output_chars)
    stderr = truncate_middle(_decode_output(err_raw), max(1500, ctx.max_output_chars // 3))
    if out_omitted:
        stdout += f"\n[... saída muito grande: {out_omitted} bytes omitidos ...]"
    code = proc.returncode
    ok = not status and code == 0
    lines = [f"Comando ({shell}, pasta {cwd}): {command}",
             f"Código de saída: {code if code is not None else 'n/d'} | duração {elapsed:.1f}s"]
    if status:
        lines.append(f"Status: {status}")
    lines.append("--- STDOUT ---\n" + (stdout.rstrip() or "(vazio)"))
    if stderr.strip():
        lines.append("--- STDERR ---\n" + stderr.rstrip() + (f"\n[... {err_omitted} bytes omitidos ...]"
                                                             if err_omitted else ""))
    summary = status or (f"código {code}" + ("" if ok else " (erro)"))
    return ToolResult(ok, "\n".join(lines), summary, elapsed)


# --------------------------------------------------------------------------- read_file
def resolve_path(path: str, workspace: Path) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(path.strip().strip('"').strip("'"))))
    if not p.is_absolute():
        for base in (workspace, workspace / "uploads"):
            if (base / p).exists():
                return base / p
        return workspace / p
    return p


def read_file(ctx: ToolContext, path: str = "", offset: Any = 0, max_chars: Any = None, **_ignored: Any) -> ToolResult:
    if not path:
        return ToolResult(False, "ERRO: parâmetro 'path' vazio.", "caminho vazio")
    p = resolve_path(str(path), ctx.workspace)
    t0 = time.perf_counter()
    try:
        text = extract_text(p)
    except UnreadableFile as e:
        msg = (f"Não foi possível ler '{p}' nativamente em Python: {e.reason}.\n"
               f"Use a ferramenta run_terminal_command para extrair o conteúdo. {e.hint}")
        return ToolResult(False, msg, e.reason, time.perf_counter() - t0)
    except (OSError, ValueError) as e:
        return ToolResult(False, f"ERRO ao ler '{p}': {e}", str(e), time.perf_counter() - t0)
    start = _as_int(offset, 0, 0, max(0, len(text)))
    limit = _as_int(max_chars, ctx.max_output_chars, 200, ctx.max_output_chars)
    chunk = text[start:start + limit]
    end = start + len(chunk)
    try:
        size = human_size(p.stat().st_size)
    except OSError:
        size = "?"
    header = f"Arquivo: {p} ({size}) — caracteres {start + 1}–{end} de {len(text)}"
    if end < len(text):
        header += f"\n(Há mais conteúdo: chame read_file com offset={end} para continuar.)"
    return ToolResult(True, f"{header}\n\n{chunk}", f"{end - start} caracteres lidos", time.perf_counter() - t0)


# --------------------------------------------------------------------------- tavily
def tavily_search(ctx: ToolContext, query: str = "", topic: str | None = None, time_range: str | None = None,
                  max_results: Any = None, **_ignored: Any) -> ToolResult:
    query = (query or "").strip()
    if not query:
        return ToolResult(False, "ERRO: parâmetro 'query' vazio.", "consulta vazia")
    key = tavily_api_key()
    if not key:
        return ToolResult(False, "ERRO: chave TAVILY_API_KEY ausente no arquivo .env. Informe ao usuário que a "
                          "busca na web está indisponível e responda com cautela.", "sem chave Tavily")
    try:
        from tavily import TavilyClient  # type: ignore
    except ImportError:
        return ToolResult(False, "ERRO: pacote tavily-python não instalado.", "tavily-python ausente")
    kwargs: dict[str, Any] = {
        "query": query[:400],
        "search_depth": os.environ.get("TAVILY_SEARCH_DEPTH", "advanced"),
        "topic": topic if topic in ("general", "news", "finance") else "general",
        "max_results": _as_int(max_results, 5, 1, 10),
        "include_answer": True,
        "timeout": 45,
    }
    if time_range in ("day", "week", "month", "year"):
        kwargs["time_range"] = time_range
    t0 = time.perf_counter()
    try:
        resp = TavilyClient(api_key=key).search(**kwargs)
    except Exception as e:  # noqa: BLE001 - chave inválida, limite, rede...
        name = type(e).__name__
        hint = {"InvalidAPIKeyError": "chave do Tavily inválida (verifique o .env)",
                "UsageLimitExceededError": "limite de uso da conta Tavily atingido",
                "TimeoutError": "tempo esgotado na busca"}.get(name, str(e)[:300])
        return ToolResult(False, f"ERRO na busca Tavily ({name}): {hint}. Informe ao usuário se não puder "
                          "confirmar a informação.", hint, time.perf_counter() - t0)
    results = resp.get("results") or []
    per_source = max(400, min(1500, ctx.max_output_chars // max(1, len(results) + 1)))
    lines = [f'Resultados da busca Tavily para: "{query}" (consultado em {time.strftime("%d/%m/%Y %H:%M")})']
    if resp.get("answer"):
        lines.append(f"\nResumo automático: {resp['answer']}")
    for i, r in enumerate(results, 1):
        content = re.sub(r"\s+", " ", str(r.get("content") or "")).strip()
        date = f" | publicado: {r['published_date']}" if r.get("published_date") else ""
        lines.append(f"\n[{i}] {r.get('title') or '(sem título)'}\nURL: {r.get('url')}{date}\n"
                     f"Trecho: {content[:per_source]}")
    if not results:
        lines.append("\nNenhum resultado encontrado. Tente reformular a consulta.")
    lines.append("\nUse estas fontes na resposta e cite as URLs relevantes.")
    return ToolResult(bool(results), truncate_middle("\n".join(lines), ctx.max_output_chars),
                      f"{len(results)} fontes", time.perf_counter() - t0)


# --------------------------------------------------------------------------- despacho
_REGISTRY: dict[str, Callable[..., ToolResult]] = {
    "run_terminal_command": run_terminal_command,
    "read_file": read_file,
    "tavily_search": tavily_search,
}

_ALIASES = {
    "terminal": "run_terminal_command", "run_command": "run_terminal_command", "execute_command": "run_terminal_command",
    "shell": "run_terminal_command", "bash": "run_terminal_command", "powershell": "run_terminal_command",
    "web_search": "tavily_search", "search": "tavily_search", "tavily": "tavily_search", "search_web": "tavily_search",
    "read": "read_file", "open_file": "read_file", "file_read": "read_file",
}

_PARAM_ALIASES = {
    "tavily_search": {"q": "query", "search_query": "query", "consulta": "query"},
    "run_terminal_command": {"cmd": "command", "comando": "command", "timeout": "timeout_seconds", "cwd": "working_directory"},
    "read_file": {"file": "path", "file_path": "path", "filename": "path", "caminho": "path"},
}


def execute_tool(name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    real = name if name in _REGISTRY else _ALIASES.get(name.lower().strip(), "")
    if not real:
        return ToolResult(False, f"ERRO: ferramenta '{name}' não existe. Ferramentas disponíveis: "
                          f"{', '.join(sorted(_REGISTRY))}.", "ferramenta desconhecida")
    fixed = {_PARAM_ALIASES.get(real, {}).get(k, k): v for k, v in (args or {}).items()}
    t0 = time.perf_counter()
    try:
        result = _REGISTRY[real](ctx, **fixed)
    except Exception as e:  # noqa: BLE001 - nunca derruba o agente
        result = ToolResult(False, f"ERRO interno ao executar {real}: {type(e).__name__}: {e}", str(e))
    if not result.duration:
        result.duration = time.perf_counter() - t0
    return result
