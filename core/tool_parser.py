"""Interpreta a saída do modelo: pensamento (<think>), texto e chamadas de ferramenta.

Modelos diferentes emitem tool calls em formatos diferentes; todos são aceitos:
  * Hermes / Qwen 2.5-3 / InternVL / MiniCPM:  <tool_call>{"name": ..., "arguments": {...}}</tool_call>
  * Qwen3-Coder / Qwen3.5 (XML):              <tool_call><function=nome><parameter=x>v</parameter></function></tool_call>
  * Mistral / Pixtral:                         [TOOL_CALLS][{"name":..., "arguments":...}]  ou  [TOOL_CALLS]nome[ARGS]{...}
  * Gemma 4:                                   <|tool_call>call:nome{chave:<|"|>valor<|"|>}<tool_call|>
  * LFM2-VL (pythonic):                        <|tool_call_start|>[nome(chave="valor")]<|tool_call_end|>
  * Llama 3.x:                                 <|python_tag|>{"name": ..., "parameters": {...}}
  * JSON puro (resposta inteira ou bloco ```json) com "name" + "arguments"/"parameters".
"""

from __future__ import annotations

import ast
import json
import random
import re
import string
from dataclasses import dataclass, field
from typing import Any

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

# (marcador de início, marcador de fim ou None = até o fim da mensagem)
TOOL_MARKERS: list[tuple[str, str | None]] = [
    ("<tool_call>", "</tool_call>"),
    ("<|tool_call>", "<tool_call|>"),
    ("<|tool_call_start|>", "<|tool_call_end|>"),
    ("<function=", "</function>"),
    ("[TOOL_CALLS]", None),
    ("<|python_tag|>", None),
]


def new_call_id() -> str:
    # 9 caracteres alfanuméricos: formato exigido pelos templates da Mistral, aceito pelos demais.
    return "".join(random.choices(string.ascii_letters + string.digits, k=9))


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=new_call_id)
    error: str = ""  # argumentos ilegíveis: o agente devolve o erro ao modelo


@dataclass
class ParsedOutput:
    content: str
    thinking: str
    tool_calls: list[ToolCall]
    raw: str  # saída sem o bloco de pensamento (usada no histórico do modo "prompted")


# --------------------------------------------------------------------------- JSON tolerante
def loads_lenient(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    fixed = re.sub(r",\s*([}\]])", r"\1", text)  # vírgula sobrando
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass
    try:  # aspas simples / True / None (estilo Python)
        return ast.literal_eval(re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False",
                                                                  re.sub(r"\bnull\b", "None", fixed))))
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        pass
    # Primeiro objeto JSON completo dentro do texto.
    start = text.find("{")
    if start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    raise ValueError("JSON inválido")


def _coerce_args(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        value = loads_lenient(value)
    if not isinstance(value, dict):
        raise ValueError("os argumentos devem ser um objeto JSON")
    return value


def _call_from_obj(obj: Any) -> list[ToolCall]:
    if isinstance(obj, list):
        out: list[ToolCall] = []
        for item in obj:
            out.extend(_call_from_obj(item))
        return out
    if not isinstance(obj, dict):
        raise ValueError("chamada de ferramenta não é um objeto")
    if "function" in obj and isinstance(obj["function"], dict):  # formato OpenAI
        obj = obj["function"]
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError('campo "name" ausente')
    args_raw = obj.get("arguments", obj.get("parameters", obj.get("args", obj.get("input"))))
    try:
        return [ToolCall(name.strip(), _coerce_args(args_raw))]
    except ValueError as e:
        return [ToolCall(name.strip(), {}, error=f"argumentos inválidos: {e}")]


def _parse_xml_function(body: str) -> list[ToolCall]:
    calls = []
    for m in re.finditer(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", body, re.S):
        args: dict[str, Any] = {}
        for p in re.finditer(r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", m.group(2), re.S):
            raw = p.group(2).strip("\n")
            try:
                val = json.loads(raw)
                args[p.group(1)] = val if not isinstance(val, str) else raw.strip()
            except (json.JSONDecodeError, ValueError):
                args[p.group(1)] = raw.strip()
        calls.append(ToolCall(m.group(1).strip(), args))
    return calls


def _gemma_to_json(text: str) -> str:
    strings: list[str] = []

    def capture(m: re.Match[str]) -> str:
        strings.append(m.group(1))
        return f"\x00{len(strings) - 1}\x00"

    text = re.sub(r'<\|"\|>(.*?)<\|"\|>', capture, text, flags=re.S)
    text = re.sub(r"(?<=[{,])\s*(\w+)\s*:", r'"\1":', text)
    for i, s in enumerate(strings):
        text = text.replace(f"\x00{i}\x00", json.dumps(s, ensure_ascii=False))
    return text


def _parse_gemma(body: str) -> list[ToolCall]:
    calls = []
    for m in re.finditer(r"call:([\w.-]+)\s*(\{.*\})", body, re.S):
        try:
            calls.append(ToolCall(m.group(1), _coerce_args(_gemma_to_json(m.group(2)))))
        except ValueError as e:
            calls.append(ToolCall(m.group(1), {}, error=f"argumentos inválidos: {e}"))
    return calls


def _parse_pythonic(body: str) -> list[ToolCall]:
    body = body.strip()
    if not body.startswith("["):
        body = f"[{body}]"
    tree = ast.parse(body, mode="eval")
    calls = []
    for node in getattr(tree.body, "elts", []):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
            calls.append(ToolCall(name, args))
    return calls


def _parse_mistral(body: str) -> list[ToolCall]:
    body = body.strip()
    if body.startswith(("[", "{")):
        return _call_from_obj(loads_lenient(body))
    calls = []
    for m in re.finditer(r"([\w.-]+)\s*(?:\[CALL_ID\][\w-]+)?\s*\[ARGS\]\s*(\{.*?\})(?=\s*(?:\[TOOL_CALLS\]|[\w.-]+\s*\[ARGS\]|$))",
                         body, re.S):
        try:
            calls.append(ToolCall(m.group(1), _coerce_args(m.group(2))))
        except ValueError as e:
            calls.append(ToolCall(m.group(1), {}, error=f"argumentos inválidos: {e}"))
    return calls


def _parse_block(marker: str, body: str) -> list[ToolCall]:
    try:
        if marker == "<tool_call>":
            if "<function=" in body:
                return _parse_xml_function(body)
            return _call_from_obj(loads_lenient(body))
        if marker == "<function=":
            return _parse_xml_function("<function=" + body)
        if marker == "<|tool_call>":
            return _parse_gemma(body)
        if marker == "<|tool_call_start|>":
            return _parse_pythonic(body)
        if marker == "[TOOL_CALLS]":
            return _parse_mistral(body)
        if marker == "<|python_tag|>":
            return _call_from_obj(loads_lenient(body))
    except (ValueError, SyntaxError) as e:
        return [ToolCall("?", {}, error=f"não foi possível interpretar a chamada ({e}): {body[:200]}")]
    return []


# --------------------------------------------------------------------------- parse final
def split_thinking(text: str, starts_in_thinking: bool = False) -> tuple[str, str]:
    """Retorna (pensamento, restante)."""
    if starts_in_thinking and THINK_CLOSE in text and THINK_OPEN not in text.split(THINK_CLOSE, 1)[0]:
        think, rest = text.split(THINK_CLOSE, 1)
        return think.strip(), rest
    thoughts: list[str] = []

    def grab(m: re.Match[str]) -> str:
        thoughts.append(m.group(1).strip())
        return ""

    rest = re.sub(r"<think>(.*?)(?:</think>|$)", grab, text, flags=re.S)
    if not thoughts and THINK_CLOSE in rest:  # template já abriu o <think>
        think, rest = rest.split(THINK_CLOSE, 1)
        thoughts.append(think.strip())
    if starts_in_thinking and not thoughts and THINK_CLOSE not in text:
        return text.strip(), ""  # modelo ainda pensando quando parou
    return "\n\n".join(t for t in thoughts if t), rest


def parse_model_output(text: str, known_tools: set[str] | None = None,
                       starts_in_thinking: bool = False) -> ParsedOutput:
    thinking, rest = split_thinking(text, starts_in_thinking)
    calls: list[ToolCall] = []
    visible_parts: list[str] = []
    pos = 0
    while pos < len(rest):
        best: tuple[int, str, str | None] | None = None
        for start, end in TOOL_MARKERS:
            i = rest.find(start, pos)
            if i >= 0 and (best is None or i < best[0]):
                best = (i, start, end)
        if best is None:
            visible_parts.append(rest[pos:])
            break
        i, start, end = best
        visible_parts.append(rest[pos:i])
        body_start = i + len(start)
        if end is None:
            body, pos = rest[body_start:], len(rest)
        else:
            j = rest.find(end, body_start)
            body, pos = (rest[body_start:j], j + len(end)) if j >= 0 else (rest[body_start:], len(rest))
        calls.extend(_parse_block(start, body))

    content = "".join(visible_parts).strip()
    if not calls and known_tools:
        calls = _bare_json_calls(content, known_tools)
        if calls:
            content = ""
    return ParsedOutput(content=content, thinking=thinking, tool_calls=calls, raw=rest.strip())


def _bare_json_calls(content: str, known_tools: set[str]) -> list[ToolCall]:
    """Resposta que é SÓ um JSON de chamada (sem marcadores) para uma ferramenta conhecida."""
    stripped = content.strip()
    m = re.fullmatch(r"```(?:json|tool_call)?\s*(.*?)\s*```", stripped, re.S)
    if m:
        stripped = m.group(1).strip()
    if not stripped.startswith(("{", "[")):
        return []
    try:
        calls = _call_from_obj(loads_lenient(stripped))
    except ValueError:
        return []
    return calls if calls and all(c.name in known_tools for c in calls) else []


# --------------------------------------------------------------------------- streaming
class StreamFilter:
    """Separa em tempo real texto visível, pensamento e chamadas de ferramenta (ocultas).

    feed() devolve eventos ("text" | "think" | "tool", trecho). Um marcador que pode
    estar cortado entre dois tokens fica retido até ser resolvido.
    """

    def __init__(self, starts_in_thinking: bool = False):
        self.mode = "think" if starts_in_thinking else "text"
        self.buf = ""
        self.tool_end: str | None = None

    def _markers(self) -> list[str]:
        if self.mode == "text":
            return [THINK_OPEN, THINK_CLOSE] + [s for s, _ in TOOL_MARKERS]
        if self.mode == "think":
            return [THINK_CLOSE]
        return [self.tool_end] if self.tool_end else []

    def feed(self, delta: str) -> list[tuple[str, str]]:
        self.buf += delta
        events: list[tuple[str, str]] = []
        while True:
            markers = self._markers()
            hit = min(((self.buf.find(m), m) for m in markers if m and m in self.buf), default=None)
            if hit is None:
                keep = _partial_suffix(self.buf, markers)
                emit, self.buf = self.buf[: len(self.buf) - keep], self.buf[len(self.buf) - keep:]
                if emit:
                    events.append((self.mode if self.mode != "tool" else "tool", emit))
                return events
            i, marker = hit
            if i:
                events.append((self.mode, self.buf[:i]))
            self.buf = self.buf[i + len(marker):]
            if self.mode == "text":
                if marker == THINK_OPEN:
                    self.mode = "think"
                elif marker == THINK_CLOSE:
                    pass  # fechamento solto: ignora
                else:
                    self.mode = "tool"
                    self.tool_end = dict(TOOL_MARKERS)[marker]
                    events.append(("tool_start", marker))
            elif self.mode == "think":
                self.mode = "text"
            else:  # fim da chamada de ferramenta
                self.mode, self.tool_end = "text", None

    def flush(self) -> list[tuple[str, str]]:
        out = [(self.mode, self.buf)] if self.buf else []
        self.buf = ""
        return out


def _partial_suffix(buf: str, markers: list[str | None]) -> int:
    """Tamanho do maior sufixo de buf que é prefixo de algum marcador."""
    best = 0
    for m in markers:
        if not m:
            continue
        for k in range(min(len(m) - 1, len(buf)), 0, -1):
            if buf.endswith(m[:k]):
                best = max(best, k)
                break
    return best
