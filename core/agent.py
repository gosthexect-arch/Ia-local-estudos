"""Loop agêntico: o modelo pensa -> chama ferramentas -> observa -> responde.

Suporta dois modos de tool calling, escolhidos pelo chat template do GGUF:
  * native   — o template do modelo sabe renderizar `tools` (Qwen-VL, Mistral, Gemma 4...):
               as ferramentas vão no formato em que o modelo foi treinado.
  * prompted — templates sem suporte (Gemma 3, LLaVA, SmolVLM...): as ferramentas são
               descritas no system prompt e as respostas voltam como <tool_response>.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from core.config import UPLOADS_DIR, WORKSPACE_DIR, Settings
from core.engine import ContextOverflowError, EngineError, LlamaEngine
from core.file_reader import UnreadableFile, extract_text, human_size, image_to_data_uri, is_image, sniff_type
from core.prompts import build_system_prompt
from core.tool_parser import ParsedOutput, StreamFilter, ToolCall, parse_model_output
from core.tools import TOOL_NAMES, ToolContext, ToolResult, execute_tool, tool_schemas, truncate_middle

CHARS_PER_TOKEN = 3.0
IMAGE_PLACEHOLDER = "[imagem enviada anteriormente — removida do contexto para economizar memória]"


@dataclass
class AgentEvent:
    kind: str                     # status | thinking | text | tool_call | tool_result | stats | notice | error | final
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    messages: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    system_key: tuple = ()
    mode: str = ""                # "native" | "prompted"
    prompted_only: str = ""       # modelo cujo template falhou com ferramentas nativas

    def reset(self) -> None:
        self.messages.clear()
        self.system_prompt, self.system_key, self.mode, self.prompted_only = "", (), "", ""


def output_budget(n_ctx: int) -> int:
    return int(min(24000, max(2500, n_ctx * CHARS_PER_TOKEN * 0.30)))


def attachment_budget(n_ctx: int) -> int:
    return int(min(90000, max(3000, n_ctx * CHARS_PER_TOKEN * 0.45)))


def _safe_name(name: str) -> str:
    name = re.sub(r"[^\w.\-() ]+", "_", Path(name).name).strip() or "arquivo"
    return name[-120:]


def save_upload(src: str | Path, original_name: str | None = None) -> Path:
    """Copia o anexo para workspace/uploads (caminho estável, acessível ao terminal)."""
    src = Path(src)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{_safe_name(original_name or src.name)}"
    i = 1
    while dest.exists():
        dest = dest.with_name(f"{dest.stem}_{i}{dest.suffix}")
        i += 1
    shutil.copy2(src, dest)
    return dest


def _text_of(msg: dict[str, Any]) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


class Agent:
    def __init__(self, engine: LlamaEngine):
        self.engine = engine

    # ------------------------------------------------------------ mensagem do usuário
    def build_user_message(self, text: str, files: list[tuple[str, str]], settings: Settings,
                           n_ctx: int) -> tuple[dict[str, Any], list[str]]:
        """files: [(caminho_temporário, nome_original)]. Retorna (mensagem, avisos)."""
        notices: list[str] = []
        images: list[str] = []
        blocks: list[str] = []
        docs: list[tuple[Path, str]] = []
        for tmp, original in files:
            try:
                saved = save_upload(tmp, original)
            except OSError as e:
                notices.append(f"Não foi possível salvar '{original}': {e}")
                continue
            if is_image(saved):
                try:
                    images.append(image_to_data_uri(saved, settings.image_max_side))
                    blocks.append(f"[Imagem anexada: {original} — salva em {saved}]")
                    continue
                except Exception as e:  # noqa: BLE001 - formato que o Pillow não abre
                    notices.append(f"'{original}' não pôde ser aberta como imagem ({e}); tratada como arquivo.")
            docs.append((saved, original))

        budget = attachment_budget(n_ctx)
        per_doc = budget // max(1, len(docs))
        for saved, original in docs:
            size = human_size(saved.stat().st_size)
            header = f"[Arquivo anexado: {original} ({size}, {sniff_type(saved)}) — salvo em {saved}]"
            try:
                content = extract_text(saved)
            except UnreadableFile as e:
                blocks.append(f"{header}\nO Python não conseguiu ler este arquivo nativamente: {e.reason}.\n"
                              f"Use run_terminal_command para extrair o conteúdo antes de responder. {e.hint}")
                notices.append(f"'{original}': leitura nativa falhou ({e.reason}) — a IA usará o terminal.")
                continue
            if len(content) > per_doc:
                shown = content[:per_doc]
                blocks.append(f"{header}\n<conteúdo caracteres 1–{per_doc} de {len(content)}; use read_file com "
                              f"offset={per_doc} para ler o restante>\n{shown}\n</conteúdo>")
            else:
                blocks.append(f"{header}\n<conteúdo>\n{content}\n</conteúdo>")

        body = text.strip() or ("Analise o(s) arquivo(s) anexado(s)." if files else "")
        full_text = body + ("\n\n" + "\n\n".join(blocks) if blocks else "")
        msg: dict[str, Any]
        if images:
            parts: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": u}} for u in images]
            parts.append({"type": "text", "text": full_text})
            msg = {"role": "user", "content": parts}
        else:
            msg = {"role": "user", "content": full_text}
        msg["_turn"] = True  # início de um turno do usuário (não enviado ao modelo)
        return msg, notices

    # ------------------------------------------------------------ loop principal
    def run(self, conv: Conversation, text: str, files: list[tuple[str, str]], settings: Settings,
            cancel: threading.Event) -> Iterator[AgentEvent]:
        loaded = self.engine.loaded
        if loaded is None:
            yield AgentEvent("error", "Nenhum modelo carregado. Escolha um modelo no menu lateral.")
            return
        snapshot = copy.deepcopy(conv.messages)
        n_ctx = loaded.n_ctx
        tools = tool_schemas()
        wanted = "native" if loaded.native_tools and conv.prompted_only != loaded.entry.key else "prompted"
        if conv.mode != wanted:
            if conv.mode == "native":
                self._convert_history_to_prompted(conv)
            conv.mode = wanted
        self._ensure_system_prompt(conv, loaded.entry.key, tools)

        user_msg, notices = self.build_user_message(text, files, settings, n_ctx)
        for n in notices:
            yield AgentEvent("notice", n)
        conv.messages.append(user_msg)
        ctx = ToolContext(terminal_enabled=settings.terminal_enabled, terminal_timeout=settings.terminal_timeout,
                          max_output_chars=output_budget(n_ctx), cancel=cancel)
        executed: dict[str, ToolResult] = {}
        max_steps = max(1, int(settings.max_agent_steps))
        step, overflow_retries, mode_fallback = 0, 0, False

        while step <= max_steps:
            if cancel.is_set() and step > 0:  # parado entre ferramentas: encerra sem nova geração
                conv.messages.append({"role": "assistant", "content": "(interrompido pelo usuário)"})
                yield AgentEvent("final", "(interrompido pelo usuário)", {"cancelled": True})
                return
            final_step = step == max_steps
            request = self._request_messages(conv, final_step)
            use_tools = tools if (conv.mode == "native" and not final_step) else None
            yield AgentEvent("status", "Processando…")
            try:
                raw = yield from self._generate(request, use_tools, settings, cancel)
            except ContextOverflowError as e:
                overflow_retries += 1
                if overflow_retries <= 12 and self._trim(conv):
                    yield AgentEvent("notice", "Contexto cheio: partes antigas da conversa foram resumidas/removidas.")
                    continue
                conv.messages[:] = snapshot
                yield AgentEvent("error", f"A mensagem não cabe na janela de contexto ({e.needed} > {e.n_ctx} "
                                          "tokens). Aumente o contexto no menu lateral ou envie menos conteúdo.")
                return
            except EngineError as e:
                conv.messages[:] = snapshot
                yield AgentEvent("error", str(e))
                return
            except Exception as e:  # noqa: BLE001 - p.ex. template que não aceita tools/role tool
                template_error = not isinstance(e, (RuntimeError, OSError, MemoryError))
                if conv.mode == "native" and not mode_fallback and template_error:
                    mode_fallback = True
                    conv.mode, conv.prompted_only = "prompted", loaded.entry.key
                    self._convert_history_to_prompted(conv)
                    self._ensure_system_prompt(conv, loaded.entry.key, tools, force=True)
                    yield AgentEvent("notice", f"Template do modelo não aceitou ferramentas nativas ({e}); "
                                               "usando o modo de ferramentas via prompt.")
                    continue
                conv.messages[:] = snapshot
                yield AgentEvent("error", f"Erro durante a geração: {type(e).__name__}: {e}")
                return

            stats = self.engine.last_stats
            yield AgentEvent("stats", stats.describe(), {"tps": stats.tokens_per_second})
            parsed = parse_model_output(raw, TOOL_NAMES, self.engine.starts_in_thinking)

            if cancel.is_set():
                partial = parsed.content or "(resposta interrompida)"
                conv.messages.append({"role": "assistant", "content": partial})
                yield AgentEvent("final", partial, {"thinking": parsed.thinking, "cancelled": True})
                return

            if parsed.tool_calls and not final_step:
                conv.messages.append(self._assistant_tool_message(conv, parsed))
                results: list[tuple[ToolCall, ToolResult]] = []
                for call in parsed.tool_calls:
                    yield AgentEvent("tool_call", call.name, {"id": call.id, "arguments": call.arguments,
                                                              "content": parsed.content})
                    result = self._run_tool(call, ctx, executed)
                    results.append((call, result))
                    yield AgentEvent("tool_result", call.name, {
                        "id": call.id, "ok": result.ok, "summary": result.summary, "duration": result.duration,
                        "content": result.content, "arguments": call.arguments})
                    if cancel.is_set():
                        break
                conv.messages.extend(self._tool_result_messages(conv, results))
                step += 1
                continue

            answer = parsed.content
            if not answer and parsed.tool_calls:
                answer = "Não consegui concluir a tarefa dentro do limite de passos de ferramentas."
            if not answer:
                answer = "(o modelo não gerou resposta)"
            if stats.finish_reason == "length":
                answer += "\n\n_(resposta cortada: limite de tokens de saída atingido)_"
            conv.messages.append({"role": "assistant", "content": answer})
            yield AgentEvent("final", answer, {"thinking": parsed.thinking})
            return

    # ------------------------------------------------------------ geração com streaming
    def _generate(self, messages: list[dict[str, Any]], tools: list[dict] | None, settings: Settings,
                  cancel: threading.Event) -> Iterator[AgentEvent]:
        chunks: list[str] = []
        filt: StreamFilter | None = None
        for delta in self.engine.chat_stream(messages, tools=tools, temperature=settings.temperature,
                                             max_tokens=settings.max_tokens, cancel=cancel):
            if filt is None:
                filt = StreamFilter(self.engine.starts_in_thinking)
            chunks.append(delta)
            for kind, piece in filt.feed(delta):
                if kind == "text":
                    yield AgentEvent("text", piece)
                elif kind == "think":
                    yield AgentEvent("thinking", piece)
                elif kind == "tool_start":
                    yield AgentEvent("status", "Preparando chamada de ferramenta…")
        if filt is not None:
            for kind, piece in filt.flush():
                if kind == "text":
                    yield AgentEvent("text", piece)
                elif kind == "think":
                    yield AgentEvent("thinking", piece)
        return "".join(chunks)

    # ------------------------------------------------------------ ferramentas
    @staticmethod
    def _run_tool(call: ToolCall, ctx: ToolContext, executed: dict[str, ToolResult]) -> ToolResult:
        if call.error:
            return ToolResult(False, f"ERRO: {call.error}. Reenvie a chamada no formato correto.", call.error)
        key = call.name + json.dumps(call.arguments, sort_keys=True, ensure_ascii=False, default=str)
        if key in executed:
            prev = executed[key]
            return ToolResult(prev.ok, prev.content + "\n\n[Aviso: esta chamada é idêntica a uma anterior. Use o "
                              "resultado acima e responda ao usuário, ou tente algo diferente.]", "chamada repetida")
        result = execute_tool(call.name, call.arguments, ctx)
        executed[key] = result
        return result

    @staticmethod
    def _assistant_tool_message(conv: Conversation, parsed: ParsedOutput) -> dict[str, Any]:
        if conv.mode == "native":
            return {"role": "assistant", "content": parsed.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in parsed.tool_calls]}
        return {"role": "assistant", "content": parsed.raw}

    def _tool_result_messages(self, conv: Conversation, results: list[tuple[ToolCall, ToolResult]]
                              ) -> list[dict[str, Any]]:
        loaded = self.engine.loaded
        if conv.mode == "native" and loaded is not None and loaded.tool_role:
            return [{"role": "tool", "tool_call_id": c.id, "name": c.name, "content": r.content} for c, r in results]
        body = "\n".join(f"<tool_response>\n[{c.name}]\n{r.content}\n</tool_response>" for c, r in results)
        return [{"role": "user", "content": body, "_tool": True}]

    # ------------------------------------------------------------ prompt / contexto
    def _ensure_system_prompt(self, conv: Conversation, model_key: str, tools: list[dict[str, Any]],
                              force: bool = False) -> None:
        # Mantido estável durante o dia: mudar o system prompt invalida o cache de prefixo.
        key = (model_key, conv.mode, time.strftime("%Y-%m-%d"))
        if force or key != conv.system_key or not conv.system_prompt:
            conv.system_prompt = build_system_prompt(WORKSPACE_DIR, tools, native_tools=conv.mode == "native")
            conv.system_key = key

    @staticmethod
    def _request_messages(conv: Conversation, final_step: bool) -> list[dict[str, Any]]:
        msgs = [{"role": "system", "content": conv.system_prompt}]
        for m in conv.messages:
            msgs.append({k: v for k, v in m.items() if not k.startswith("_")})
        if final_step:
            nudge = ("\n\n(Limite de chamadas de ferramentas atingido. Responda agora ao usuário com as "
                     "informações que já tem, sem chamar ferramentas.)")
            last = msgs[-1]
            if isinstance(last.get("content"), str) and last["role"] in ("user", "tool"):
                last["content"] = last["content"] + nudge
            else:
                msgs.append({"role": "user", "content": nudge.strip()})
        return msgs

    @staticmethod
    def _convert_history_to_prompted(conv: Conversation) -> None:
        """Reescreve tool calls nativas como texto (<tool_call>/<tool_response>)."""
        out: list[dict[str, Any]] = []
        for m in conv.messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                calls = "\n".join("<tool_call>\n" + json.dumps({"name": c["function"]["name"],
                                                                 "arguments": c["function"]["arguments"]},
                                                                ensure_ascii=False) + "\n</tool_call>"
                                  for c in m["tool_calls"])
                out.append({"role": "assistant", "content": ((m.get("content") or "") + "\n" + calls).strip()})
            elif m.get("role") == "tool":
                block = f"<tool_response>\n[{m.get('name', '')}]\n{m.get('content', '')}\n</tool_response>"
                if out and out[-1].get("_tool"):
                    out[-1]["content"] += "\n" + block
                else:
                    out.append({"role": "user", "content": block, "_tool": True})
            else:
                out.append(m)
        conv.messages[:] = out

    @staticmethod
    def _trim(conv: Conversation) -> bool:
        """Libera contexto: imagens antigas -> resultados antigos -> turnos antigos -> anexos atuais."""
        msgs = conv.messages
        starts = [i for i, m in enumerate(msgs) if m.get("_turn")]
        current = starts[-1] if starts else 0
        for m in msgs[:current]:
            if isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"]):
                m["content"] = IMAGE_PLACEHOLDER + "\n" + _text_of(m)
                return True
        for m in msgs[:current]:
            if (m.get("role") == "tool" or m.get("_tool")) and len(_text_of(m)) > 1500:
                m["content"] = truncate_middle(_text_of(m), 1200)
                return True
        if len(starts) >= 2:
            del msgs[: starts[1]]
            return True
        tool_msgs = [m for m in msgs[current + 1:-1] if (m.get("role") == "tool" or m.get("_tool"))
                     and len(_text_of(m)) > 2000]
        if tool_msgs:
            m = tool_msgs[0]
            m["content"] = truncate_middle(_text_of(m), len(_text_of(m)) // 2)
            return True
        if msgs and msgs[current].get("_turn"):
            m = msgs[current]
            text = _text_of(m)
            if len(text) > 2000:
                new_text = truncate_middle(text, len(text) // 2)
                if isinstance(m["content"], list):
                    for p in m["content"]:
                        if p.get("type") == "text":
                            p["text"] = new_text
                else:
                    m["content"] = new_text
                return True
        return False
