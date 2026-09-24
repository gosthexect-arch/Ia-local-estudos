"""Loop agêntico com um motor falso roteirizado (sem llama.cpp)."""

import copy
import json
import threading
from types import SimpleNamespace

import pytest

from core import agent as agent_mod
from core import tools as tools_mod
from core.agent import Agent, Conversation
from core.config import Settings
from core.engine import ContextOverflowError, GenerationStats
from core.tools import ToolResult


class FakeEngine:
    def __init__(self, script, native=True, tool_role=True, n_ctx=4096, max_prompt_chars=None):
        self.loaded = SimpleNamespace(n_ctx=n_ctx, native_tools=native, tool_role=tool_role,
                                      entry=SimpleNamespace(key="fake.gguf"))
        self.script = list(script)
        self.calls = []
        self.last_stats = GenerationStats()
        self.max_prompt_chars = max_prompt_chars
        self.starts_in_thinking = False

    def chat_stream(self, messages, tools=None, temperature=0.6, max_tokens=2048, cancel=None):
        self.calls.append((copy.deepcopy(messages), tools))
        if self.max_prompt_chars:
            total = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
            if total > self.max_prompt_chars:
                raise ContextOverflowError(total, self.max_prompt_chars)
        out = self.script.pop(0)
        if isinstance(out, BaseException):
            raise out
        for i in range(0, len(out), 5):
            if cancel is not None and cancel.is_set():
                return
            yield out[i:i + 5]


@pytest.fixture(autouse=True)
def fake_tavily(monkeypatch, tmp_path):
    calls = []

    def fake(ctx, query="", **kw):
        calls.append(query)
        return ToolResult(True, f"RESULTADO para {query}: https://fonte.example", "1 fonte")

    monkeypatch.setitem(tools_mod._REGISTRY, "tavily_search", fake)
    monkeypatch.setattr(agent_mod, "UPLOADS_DIR", tmp_path / "uploads")
    return calls


def run(agent, conv, text, files=(), settings=None, cancel=None):
    settings = settings or Settings()
    return list(agent.run(conv, text, list(files), settings, cancel or threading.Event()))


CALL = '<tool_call>\n{"name": "tavily_search", "arguments": {"query": "notícia de hoje"}}\n</tool_call>'


def test_native_tool_round_trip(fake_tavily):
    eng = FakeEngine([CALL, "Resposta final com [fonte](https://fonte.example)."])
    conv = Conversation()
    events = run(Agent(eng), conv, "O que aconteceu hoje?")
    kinds = [e.kind for e in events]
    assert "tool_call" in kinds and "tool_result" in kinds and kinds[-1] == "final"
    assert events[-1].text.startswith("Resposta final")
    assert fake_tavily == ["notícia de hoje"]
    roles = [m["role"] for m in conv.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    call_id = conv.messages[1]["tool_calls"][0]["id"]
    assert conv.messages[2]["tool_call_id"] == call_id
    first_msgs, first_tools = eng.calls[0]
    assert first_tools and first_msgs[0]["role"] == "system"
    assert "DEVE executar a função de busca do Tavily" in first_msgs[0]["content"]
    assert "_turn" not in first_msgs[1]


def test_prompted_mode_keeps_roles_alternating():
    eng = FakeEngine([CALL, "Pronto."], native=False)
    conv = Conversation()
    run(Agent(eng), conv, "pergunta")
    sys_prompt = eng.calls[0][0][0]["content"]
    assert "<tools>" in sys_prompt and eng.calls[0][1] is None
    roles = [m["role"] for m in conv.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert "<tool_response>" in conv.messages[2]["content"]
    assert "<tool_call>" in conv.messages[1]["content"]


def test_repeated_call_is_flagged(fake_tavily):
    eng = FakeEngine([CALL, CALL, "fim"])
    events = run(Agent(eng), Conversation(), "x")
    results = [e for e in events if e.kind == "tool_result"]
    assert len(fake_tavily) == 1 and "idêntica" in results[1].data["content"]


def test_step_limit_forces_answer():
    eng = FakeEngine([CALL.replace("hoje", str(i)) for i in range(2)] + ["Resposta sem ferramentas."])
    s = Settings(max_agent_steps=2)
    events = run(Agent(eng), Conversation(), "x", settings=s)
    assert events[-1].text == "Resposta sem ferramentas."
    last_msgs, last_tools = eng.calls[-1]
    assert last_tools is None and "Limite de chamadas" in json.dumps(last_msgs, ensure_ascii=False)
    # modelo que ignora o limite: o agente encerra com uma resposta explicativa
    eng = FakeEngine([CALL.replace("hoje", str(i)) for i in range(3)])
    events = run(Agent(eng), Conversation(), "x", settings=s)
    assert events[-1].kind == "final" and "limite" in events[-1].text


def test_context_overflow_trims_old_turns():
    eng = FakeEngine(["r1", "r2"], max_prompt_chars=None)
    conv = Conversation()
    agent = Agent(eng)
    run(agent, conv, "primeira pergunta " + "a" * 3000)
    eng.script = ["r3"]
    eng.max_prompt_chars = len(conv.system_prompt) + 2500
    events = run(agent, conv, "segunda pergunta")
    assert events[-1].kind == "final" and events[-1].text == "r3"
    assert any(e.kind == "notice" for e in events)
    assert all("primeira" not in json.dumps(m, ensure_ascii=False) for m in conv.messages)


def test_context_overflow_unrecoverable_rolls_back():
    eng = FakeEngine(["nunca"], max_prompt_chars=10)
    conv = Conversation()
    events = run(Agent(eng), conv, "oi")
    assert events[-1].kind == "error" and conv.messages == []


def test_template_error_falls_back_to_prompted():
    eng = FakeEngine([ValueError("template não suporta tools"), "ok"], native=True)
    conv = Conversation()
    events = run(Agent(eng), conv, "oi")
    assert events[-1].text == "ok" and conv.mode == "prompted" and conv.prompted_only == "fake.gguf"
    assert "<tools>" in eng.calls[1][0][0]["content"]


def test_attachments_image_and_text(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    img = tmp_path / "foto.png"
    Image.new("RGB", (64, 32), (0, 128, 255)).save(img)
    doc = tmp_path / "notas.txt"
    doc.write_text("linha secreta 42", encoding="utf-8")
    eng = FakeEngine(["Vi a imagem."])
    conv = Conversation()
    run(Agent(eng), conv, "o que tem aqui?", files=[(str(img), "foto.png"), (str(doc), "notas.txt")])
    content = eng.calls[0][0][1]["content"]
    assert content[0]["type"] == "image_url" and content[0]["image_url"]["url"].startswith("data:image/jpeg")
    text = content[-1]["text"]
    assert "linha secreta 42" in text and "notas.txt" in text and "foto.png" in text


def test_unreadable_attachment_tells_model_to_use_terminal(tmp_path):
    bad = tmp_path / "planilha.xls"
    bad.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + bytes(range(256)) * 10)
    eng = FakeEngine(["ok"])
    events = run(Agent(eng), Conversation(), "", files=[(str(bad), "planilha.xls")])
    text = eng.calls[0][0][1]["content"]
    assert "run_terminal_command" in text and "Analise" in text
    assert any(e.kind == "notice" for e in events)


def test_cancel_keeps_history_consistent():
    cancel = threading.Event()
    eng = FakeEngine(["resposta longa " * 50])
    conv = Conversation()
    agent = Agent(eng)
    gen = agent.run(conv, "oi", [], Settings(), cancel)
    for ev in gen:
        if ev.kind == "text":
            cancel.set()
    assert conv.messages[-1]["role"] == "assistant"


def test_thinking_is_not_stored_in_history():
    eng = FakeEngine(["<think>passos internos</think>Resposta limpa."])
    conv = Conversation()
    events = run(Agent(eng), conv, "oi")
    assert "".join(e.text for e in events if e.kind == "thinking") == "passos internos"
    assert conv.messages[-1]["content"] == "Resposta limpa."


def test_cancel_between_tool_steps_does_not_generate_again(fake_tavily):
    cancel = threading.Event()
    eng = FakeEngine([CALL, "nunca deveria ser gerado"])
    conv = Conversation()
    events = []
    for ev in Agent(eng).run(conv, "x", [], Settings(), cancel):
        events.append(ev)
        if ev.kind == "tool_result":
            cancel.set()
    assert len(eng.calls) == 1 and events[-1].data.get("cancelled")
    assert [m["role"] for m in conv.messages] == ["user", "assistant", "tool", "assistant"]


def test_runtime_errors_do_not_switch_tool_mode():
    eng = FakeEngine([RuntimeError("llama_decode returned -3")])
    conv = Conversation()
    events = run(Agent(eng), conv, "oi")
    assert events[-1].kind == "error" and conv.mode == "native" and conv.messages == []
