"""IA Local — chat com modelo de visão (VLM) rodando no llama.cpp + agente com ferramentas.

Inicie pelo start.bat: ele instala as dependências, pede a chave do Tavily na
primeira execução, escaneia o hardware e abre esta interface no navegador.
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # 100% local: sem telemetria
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import json
import logging
import socket
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import gradio as gr

from core.agent import Agent, Conversation
from core.config import LOGS_DIR, MODELS_DIR, Settings, ensure_dirs, load_env, tavily_api_key
from core.engine import EngineError, LlamaEngine
from core.models import ModelEntry, list_models, mmproj_search_url
from core.offload import max_context_for_full_offload

ensure_dirs()
load_env()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), RotatingFileHandler(LOGS_DIR / "app.log", maxBytes=2_000_000,
                                                           backupCount=2, encoding="utf-8")],
)
log = logging.getLogger("ia_local")

CTX_MIN, CTX_STEP, CTX_CAP = 2048, 1024, 262144


class App:
    def __init__(self) -> None:
        self.settings = Settings.load()
        self.engine = LlamaEngine()
        self.agent = Agent(self.engine)
        self.models: dict[str, ModelEntry] = {}
        self.orphans: list[Path] = []
        self.cancel = threading.Event()
        self.status = "Nenhum modelo carregado."
        self.loading = False
        self._full_ctx: dict[tuple, int] = {}
        self.refresh_models()

    def refresh_models(self) -> None:
        ready, self.orphans = list_models()
        self.models = {m.key: m for m in ready}

    def choices(self) -> list[tuple[str, str]]:
        return [(m.label, m.key) for m in self.models.values()]

    def save(self) -> None:
        self.settings.save()


APP = App()


# =============================================================================
# Textos da barra lateral
# =============================================================================
def hardware_md() -> str:
    p = APP.engine.profile
    if not p:
        return "<small>Perfil de hardware ausente — inicie pelo <b>start.bat</b>.</small>"
    cpu, rec = p.get("cpu", {}), p.get("recommended", {})
    gpus = [g for g in p.get("gpus", []) if not g.get("integrated")]
    gpu = " + ".join(f"{g['name']} ({g['vram_total_mb'] / 1024:.0f} GB)" for g in gpus) or "nenhuma (CPU)"
    backend = p.get("installed_backend") or (p.get("backend_candidates") or ["?"])[0]
    return (f"<small>🖥️ <b>GPU:</b> {gpu}<br>⚙️ <b>CPU:</b> {cpu.get('physical_cores', '?')} núcleos · "
            f"{rec.get('n_threads', '?')} threads no llama.cpp<br>🔌 <b>Backend:</b> {backend}</small>")


def models_note() -> str:
    parts = []
    if APP.orphans:
        items = "".join(f"<br>• <b>{p.name}</b> — <a href='{mmproj_search_url(p)}' target='_blank'>procurar mmproj</a>"
                        for p in APP.orphans[:4])
        more = f"<br>… e mais {len(APP.orphans) - 4}" if len(APP.orphans) > 4 else ""
        parts.append("⚠️ <b>Falta o arquivo mmproj</b> (projetor de visão) destes modelos, por isso eles não "
                     f"aparecem na lista:{items}{more}<br>Baixe o <code>mmproj-*.gguf</code> do mesmo repositório "
                     "do modelo, coloque na mesma pasta e clique em ↻ Atualizar lista.")
    elif not APP.models:
        parts.append("Nenhum modelo encontrado. Coloque o <b>.gguf do modelo</b> e o <b>mmproj .gguf</b> na "
                     f"pasta <code>{MODELS_DIR}</code> e clique em ↻ Atualizar lista.")
    return "<small>" + "<br><br>".join(parts) + "</small>" if parts else ""


def plan_md(model_key: str | None, n_ctx: float | None) -> str:
    entry = APP.models.get(model_key or "")
    if entry is None:
        return ""
    try:
        plan = APP.engine.preview_plan(entry, int(n_ctx or APP.settings.n_ctx), APP.settings)
        key = (entry.key, APP.settings.kv_cache)
        if key not in APP._full_ctx:
            info = APP.engine.gguf(entry.path)
            mm = APP.engine.gguf(entry.mmproj) if entry.mmproj else None
            APP._full_ctx[key] = max_context_for_full_offload(info, mm, APP.engine.profile,
                                                              kv_pref=APP.settings.kv_cache)
    except Exception as e:  # noqa: BLE001
        return f"<small>Estimativa indisponível: {e}</small>"
    full = APP._full_ctx.get(key, 0)
    hint = f"<br>100% na GPU até ~{full // 1024}k tokens de contexto." if full else ""
    return f"<small>📊 Estimativa: {plan.describe()}{hint}</small>"


def status_md() -> str:
    lm = APP.engine.loaded
    if lm is None:
        return f"<small>{APP.status}</small>"
    tools = "nativo do modelo" if lm.native_tools else "via prompt"
    notes = "".join(f"<br>ℹ️ {n}" for n in lm.notes)
    if lm.plan.backend == "cpu" and (why := APP.engine.gpu_note()):
        notes += f"<br>⚠️ <b>GPU não está em uso.</b> {why}"
    return (f"<small>✅ <b>{lm.entry.path.name}</b> pronto ({lm.load_seconds:.1f}s)<br>"
            f"{lm.plan.describe()}<br>contexto {lm.n_ctx} · {lm.n_threads} threads · ferramentas: {tools}"
            f"{notes}</small>")


def ctx_slider_update(model_key: str | None, value: float | None = None) -> dict:
    entry = APP.models.get(model_key or "")
    maximum = CTX_CAP
    if entry is not None:
        try:
            maximum = min(CTX_CAP, APP.engine.gguf(entry.path).n_ctx_train or CTX_CAP)
        except Exception:  # noqa: BLE001
            pass
    maximum = max(CTX_MIN, maximum // CTX_STEP * CTX_STEP)
    val = int(value or APP.settings.n_ctx)
    val = max(CTX_MIN, min(maximum, val // CTX_STEP * CTX_STEP))
    return gr.update(maximum=maximum, value=val)


# =============================================================================
# Carga de modelo
# =============================================================================
def load_model(model_key: str | None, n_ctx: float | None):
    """Carrega (ou recarrega) o modelo; gera atualizações de status para a UI."""
    entry = APP.models.get(model_key or "")
    if entry is None:
        yield status_md(), plan_md(model_key, n_ctx)
        return
    n_ctx = int(n_ctx or APP.settings.n_ctx)
    APP.settings.model, APP.settings.n_ctx = entry.key, n_ctx
    APP.save()
    ok, err = APP.engine.available()
    if not ok:
        APP.status = f"❌ {err}"
        yield status_md(), plan_md(model_key, n_ctx)
        return
    APP.status = f"⏳ Carregando <b>{entry.path.name}</b> (contexto {n_ctx})…"
    APP.loading = True
    yield status_md(), plan_md(model_key, n_ctx)
    try:
        APP.engine.load(entry, n_ctx, APP.settings)
        APP.status = "Pronto."
    except (EngineError, Exception) as e:  # noqa: BLE001
        log.exception("falha ao carregar modelo")
        APP.status = f"❌ Falha ao carregar: {e}"
    finally:
        APP.loading = False
    yield status_md(), plan_md(model_key, n_ctx)


def on_model_change(model_key: str | None, n_ctx: float | None):
    slider = ctx_slider_update(model_key, n_ctx)
    for status, plan in load_model(model_key, slider.get("value", n_ctx)):
        yield status, plan, slider


def initial_load():
    """Na abertura da página: carrega o último modelo usado (uma vez só)."""
    key = APP.settings.model if APP.settings.model in APP.models else next(iter(APP.models), None)
    slider = ctx_slider_update(key, APP.settings.n_ctx)
    dd = gr.update(choices=APP.choices(), value=key)
    if APP.engine.loaded is not None or key is None or APP.loading:
        yield dd, slider, status_md(), plan_md(key, slider["value"]), models_note()
        return
    for status, plan in load_model(key, slider["value"]):
        yield dd, slider, status, plan, models_note()


def on_refresh(model_key: str | None):
    APP.refresh_models()
    value = model_key if model_key in APP.models else next(iter(APP.models), None)
    return gr.update(choices=APP.choices(), value=value), models_note()


def on_setting(name: str):
    def handler(value: Any) -> None:
        setattr(APP.settings, name, type(getattr(APP.settings, name))(value))
        APP.save()
    return handler


def on_load_setting(name: str):
    """Configurações que exigem recarregar o modelo (camadas, KV, visão)."""
    def handler(value: Any, model_key: str | None, n_ctx: float | None):
        setattr(APP.settings, name, type(getattr(APP.settings, name))(value))
        APP.save()
        APP._full_ctx.clear()
        if APP.engine.loaded is None:
            yield status_md(), plan_md(model_key, n_ctx)
            return
        yield from load_model(model_key, n_ctx)
    return handler


# =============================================================================
# Chat
# =============================================================================
def tool_title(name: str, args: dict[str, Any]) -> str:
    if name == "tavily_search":
        return f"🌐 Pesquisando na web: “{str(args.get('query', ''))[:90]}”"
    if name == "run_terminal_command":
        cmd = " ".join(str(args.get("command", "")).split())
        return f"💻 Terminal: {cmd[:90]}{'…' if len(cmd) > 90 else ''}"
    if name == "read_file":
        return f"📄 Lendo arquivo: {Path(str(args.get('path', ''))).name}"
    return f"🔧 {name}"


def _fence(text: str, lang: str = "text") -> str:
    fence = "````" if "```" in text else "```"
    return f"{fence}{lang}\n{text}\n{fence}"


def add_user_message(message: dict | None, history: list[dict]):
    message = message or {}
    text = (message.get("text") or "").strip()
    files = []
    for f in message.get("files") or []:
        path = f if isinstance(f, str) else (f.get("path") if isinstance(f, dict) else getattr(f, "path", None))
        if path:
            files.append(path)
    if not text and not files:
        return gr.update(), history, None
    history = list(history or [])
    for path in files:
        history.append({"role": "user", "content": {"path": path}})
    if text:
        history.append({"role": "user", "content": text})
    pending = {"text": text, "files": [(p, Path(p).name) for p in files]}
    return gr.MultimodalTextbox(value=None, interactive=False, submit_btn=False, stop_btn=True), history, pending


def respond(history: list[dict], pending: dict | None, conv: Conversation):
    history = list(history or [])
    if not pending:
        yield history, gr.update(), conv
        return
    APP.cancel.clear()
    stats = ""
    placeholder: int | None = None
    think_idx: int | None = None
    answer_idx: int | None = None
    tool_idx: dict[str, int] = {}
    last = 0.0

    def add(msg: dict) -> int:
        nonlocal placeholder
        if placeholder is not None:
            history.pop(placeholder)
            placeholder = None
        history.append(msg)
        return len(history) - 1

    def close_thinking() -> None:
        nonlocal think_idx
        if think_idx is not None:
            history[think_idx]["metadata"]["status"] = "done"
            if not history[think_idx]["content"].strip():
                history[think_idx]["content"] = "…"
            think_idx = None

    try:
        for ev in APP.agent.run(conv, pending["text"], pending["files"], APP.settings, APP.cancel):
            k = ev.kind
            if k == "status":
                if placeholder is None and think_idx is None and answer_idx is None:
                    history.append({"role": "assistant", "content": "",
                                    "metadata": {"title": f"⏳ {ev.text}", "status": "pending"}})
                    placeholder = len(history) - 1
            elif k == "thinking":
                if think_idx is None:
                    think_idx = add({"role": "assistant", "content": "",
                                     "metadata": {"title": "💭 Raciocínio", "status": "pending"}})
                history[think_idx]["content"] += ev.text
            elif k == "text":
                close_thinking()
                if answer_idx is None:
                    answer_idx = add({"role": "assistant", "content": ""})
                history[answer_idx]["content"] += ev.text
            elif k == "tool_call":
                close_thinking()
                if answer_idx is not None and not history[answer_idx]["content"].strip():
                    history.pop(answer_idx)
                answer_idx = None
                args = ev.data.get("arguments") or {}
                tool_idx[ev.data["id"]] = add({
                    "role": "assistant",
                    "content": _fence(json.dumps(args, ensure_ascii=False, indent=2), "json"),
                    "metadata": {"title": tool_title(ev.text, args), "status": "pending"},
                })
            elif k == "tool_result":
                i = tool_idx.get(ev.data["id"])
                if i is not None:
                    icon = "✅" if ev.data["ok"] else "⚠️"
                    body = ev.data["content"]
                    preview = body if len(body) < 3000 else body[:3000] + "\n[...]"
                    history[i]["content"] = (_fence(json.dumps(ev.data.get("arguments") or {}, ensure_ascii=False,
                                                               indent=2), "json")
                                             + f"\n\n**Resultado** {icon} {ev.data['summary']}\n\n"
                                             + _fence(preview))
                    history[i]["metadata"].update(status="done", duration=round(ev.data["duration"], 1),
                                                  log=ev.data["summary"][:60])
            elif k == "stats":
                stats = f"⚡ {ev.text}"
            elif k == "notice":
                add({"role": "assistant", "content": ev.text, "metadata": {"title": "ℹ️ Aviso", "status": "done"}})
            elif k == "error":
                close_thinking()
                add({"role": "assistant", "content": f"⚠️ {ev.text}"})
            elif k == "final":
                close_thinking()
                if ev.data.get("thinking") and not any(
                        m.get("metadata", {}).get("title") == "💭 Raciocínio" for m in history[-3:]):
                    add({"role": "assistant", "content": ev.data["thinking"],
                         "metadata": {"title": "💭 Raciocínio", "status": "done"}})
                    answer_idx = None
                if answer_idx is None:
                    answer_idx = add({"role": "assistant", "content": ""})
                history[answer_idx]["content"] = ev.text
            now = time.perf_counter()
            if k not in ("text", "thinking") or now - last > 0.05:
                last = now
                yield history, stats, conv
    except Exception as e:  # noqa: BLE001 - nunca derrubar a UI
        log.exception("erro no chat")
        history.append({"role": "assistant", "content": f"⚠️ Erro inesperado: {e}"})
    finally:
        if placeholder is not None and placeholder < len(history):
            history.pop(placeholder)
    yield history, stats, conv


def stop_generation() -> None:
    APP.cancel.set()


def new_chat(conv: Conversation):
    APP.cancel.set()
    conv.reset()
    return [], "", conv


# =============================================================================
# Layout
# =============================================================================
CSS = """
footer { display: none !important; }
.gradio-container { max-width: 100% !important; }
#chat, #chat.generating, #chat.pending { flex-grow: 1; border: none !important; box-shadow: none !important;
  background: transparent !important; }
#chat .bubble-wrap { background: transparent !important; }
#chat .message-wrap { max-width: 880px; margin: 0 auto; width: 100%; }
#chat .message.bot { background: transparent !important; border: none !important; box-shadow: none !important; }
#chat .message.user { border: none !important; box-shadow: none !important; border-radius: 18px !important;
  background: var(--neutral-100) !important; }
.dark #chat .message.user { background: var(--neutral-800) !important; }
#input { max-width: 880px; margin: 0 auto; width: 100%; background: transparent !important;
  border: none !important; box-shadow: none !important; }
#input .full-container { border: 1px solid var(--border-color-primary) !important; border-radius: 24px !important;
  background: var(--background-fill-primary) !important; box-shadow: 0 1px 6px rgba(0,0,0,.06); }
#input .input-wrapper { background: transparent !important; }
.form:has(> #input) { background: transparent !important; border: none !important; box-shadow: none !important; }
#stats { max-width: 880px; margin: 0 auto; width: 100%; opacity: .6; font-size: .78em; min-height: 1.2em; }
#stats p { margin: 0; }
#sidebar-title h2 { margin: 0 0 .2em 0; }
.sidebar small { line-height: 1.45; display: inline-block; }
"""

THEME = gr.themes.Soft(
    primary_hue="slate", neutral_hue="slate",
    font=["Inter", "Segoe UI", "system-ui", "-apple-system", "sans-serif"],
    font_mono=["Cascadia Code", "Consolas", "ui-monospace", "monospace"],
)

PLACEHOLDER = (
    "<div style='text-align:center; opacity:.8'><h2>Como posso ajudar?</h2>"
    "Envie uma mensagem, uma imagem ou qualquer arquivo (📎).<br>"
    "A IA pesquisa na web quando precisa de informação atual e usa o terminal para ler formatos difíceis.</div>"
)


def build_ui() -> gr.Blocks:
    s = APP.settings
    with gr.Blocks(title="IA Local", fill_height=True, analytics_enabled=False) as demo:
        conv = gr.State(Conversation())
        pending = gr.State(None)

        with gr.Sidebar(open=True, width=330, position="left"):
            gr.Markdown("## 🤖 IA Local", elem_id="sidebar-title")
            model_dd = gr.Dropdown(choices=APP.choices(), value=None, label="Modelo (VLM)", interactive=True)
            with gr.Row():
                refresh_btn = gr.Button("↻ Atualizar lista", size="sm")
                reload_btn = gr.Button("⟳ Recarregar", size="sm")
            models_info = gr.Markdown(models_note())
            ctx = gr.Slider(CTX_MIN, CTX_CAP, value=s.n_ctx, step=CTX_STEP, label="Contexto (tokens)")
            plan_box = gr.Markdown()
            status_box = gr.Markdown(status_md())
            with gr.Accordion("Avançado", open=False):
                temperature = gr.Slider(0.0, 1.5, value=s.temperature, step=0.05, label="Temperatura")
                max_tokens = gr.Slider(256, 16384, value=s.max_tokens, step=256, label="Máx. tokens por resposta")
                gpu_layers = gr.Textbox(value=s.gpu_layers, label="Camadas na GPU",
                                        info="'auto' calcula o máximo que cabe. Enter para aplicar.")
                kv_cache = gr.Radio(["auto", "f16", "q8_0"], value=s.kv_cache, label="Precisão do KV cache")
                vision_gpu = gr.Checkbox(value=s.vision_on_gpu, label="Encoder de visão na GPU (quando couber)")
                terminal = gr.Checkbox(value=s.terminal_enabled, label="Permitir que a IA use o terminal")
                steps = gr.Slider(1, 20, value=s.max_agent_steps, step=1, label="Máx. passos de ferramentas")
            new_btn = gr.Button("🗒️ Nova conversa", variant="secondary")
            gr.Markdown(hardware_md())
            if not tavily_api_key():
                gr.Markdown("<small>⚠️ TAVILY_API_KEY ausente no .env: busca web indisponível.</small>")

        chatbot = gr.Chatbot(elem_id="chat", show_label=False, scale=1, height="100%", placeholder=PLACEHOLDER,
                             buttons=["copy"], feedback_options=None, group_consecutive_messages=False,
                             latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                               {"left": "\\[", "right": "\\]", "display": True},
                                               {"left": "\\(", "right": "\\)", "display": False}])
        chat_in = gr.MultimodalTextbox(
            elem_id="input", show_label=False, file_count="multiple", sources=["upload"], autofocus=True,
            placeholder="Mensagem para a IA… (📎 anexe imagens ou qualquer arquivo)", stop_btn=False,
            max_lines=12,
        )
        stats_box = gr.Markdown(elem_id="stats")

        # ---- eventos: modelo e contexto
        demo.load(initial_load, None, [model_dd, ctx, status_box, plan_box, models_info])
        model_dd.input(on_model_change, [model_dd, ctx], [status_box, plan_box, ctx])
        refresh_btn.click(on_refresh, [model_dd], [model_dd, models_info])
        ctx.change(plan_md, [model_dd, ctx], plan_box, show_progress="hidden")
        ctx.release(load_model, [model_dd, ctx], [status_box, plan_box])
        reload_btn.click(load_model, [model_dd, ctx], [status_box, plan_box])
        gpu_layers.submit(on_load_setting("gpu_layers"), [gpu_layers, model_dd, ctx], [status_box, plan_box])
        kv_cache.input(on_load_setting("kv_cache"), [kv_cache, model_dd, ctx], [status_box, plan_box])
        vision_gpu.input(on_load_setting("vision_on_gpu"), [vision_gpu, model_dd, ctx], [status_box, plan_box])
        temperature.release(on_setting("temperature"), temperature, None)
        max_tokens.release(on_setting("max_tokens"), max_tokens, None)
        terminal.input(on_setting("terminal_enabled"), terminal, None)
        steps.release(on_setting("max_agent_steps"), steps, None)

        # ---- eventos: chat
        sub = chat_in.submit(add_user_message, [chat_in, chatbot], [chat_in, chatbot, pending], queue=False)
        sub.then(respond, [chatbot, pending, conv], [chatbot, stats_box, conv], show_progress="hidden") \
           .then(lambda: gr.MultimodalTextbox(interactive=True, submit_btn=True, stop_btn=False), None, chat_in)
        chat_in.stop(stop_generation, None, None, queue=False)
        new_btn.click(new_chat, [conv], [chatbot, stats_box, conv], queue=False)
        chatbot.clear(new_chat, [conv], [chatbot, stats_box, conv], queue=False)  # lixeira do chat
    return demo


def free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


def main() -> None:
    demo = build_ui()
    port = free_port(int(os.environ.get("APP_PORT", "7860")))
    host = os.environ.get("APP_HOST", "127.0.0.1")
    log.info("Abrindo a interface em http://%s:%s", host, port)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=host, server_port=port, inbrowser=os.environ.get("APP_NO_BROWSER") != "1",
        theme=THEME, css=CSS, show_error=True, max_file_size="2gb", footer_links=[],
    )


if __name__ == "__main__":
    main()
