"""Motor LLM: llama-cpp-python + VLM (libmtmd) com offload automático para GPU.

Destaques
---------
* Binding de GPU: expõe as DLLs do runtime CUDA instaladas via pip antes de
  importar o llama_cpp; lê a VRAM livre direto do ggml (CUDA/Vulkan/Metal).
* Offload: n_gpu_layers / tipo do KV cache calculados por core.offload a partir
  do GGUF real; se a carga falhar por falta de memória, tenta de novo com menos
  camadas (degradação graciosa, sem travar o app).
* VisionChatHandler: handler multimodal genérico (usa o chat template do próprio
  GGUF, então serve para Qwen-VL, Gemma 3/4, MiniCPM-V, SmolVLM, Mistral, ...)
  com CACHE DE PREFIXO: entre um passo e outro do agente só os tokens novos são
  processados; imagens já codificadas não passam de novo pelo encoder de visão.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from core import gpu_plugins, hardware
from core.config import Settings, load_hardware_profile
from core.gguf_info import GGUFInfo, read_gguf
from core.models import ModelEntry
from core.offload import KV_GGML_TYPE, OffloadPlan, plan_offload

log = logging.getLogger("ia_local.engine")

N_BATCH = 512


class ContextOverflowError(RuntimeError):
    """O prompt (histórico + anexos) não cabe na janela de contexto."""

    def __init__(self, needed: int, n_ctx: int):
        super().__init__(f"Prompt com {needed} tokens excede o contexto de {n_ctx} tokens")
        self.needed = needed
        self.n_ctx = n_ctx


class EngineError(RuntimeError):
    pass


_LLAMA: dict[str, Any] = {}


def _llama_modules() -> dict[str, Any]:
    """Importa o llama_cpp sob demanda (a UI abre mesmo se a instalação falhou)."""
    if not _LLAMA:
        hardware.setup_gpu_dll_paths()
        import llama_cpp  # type: ignore
        import llama_cpp.llama_chat_format as lcf  # type: ignore
        import llama_cpp.mtmd_cpp as mtmd_cpp  # type: ignore
        from llama_cpp import Llama  # type: ignore
        from llama_cpp._utils import suppress_stdout_stderr  # type: ignore

        _LLAMA.update(llama_cpp=llama_cpp, lcf=lcf, mtmd=mtmd_cpp, Llama=Llama, quiet=suppress_stdout_stderr)
        _LLAMA["Handler"] = _make_handler_class()
        try:  # plugin oficial de GPU (Vulkan/CUDA) instalado pelo start.bat
            loaded = gpu_plugins.load()
            if loaded:
                log.info("backend de GPU carregado: %s", ", ".join(loaded))
        except Exception as e:  # noqa: BLE001 - sem plugin: segue na CPU/wheel instalada
            log.warning("não foi possível carregar o plugin de GPU: %s", e)
        try:  # logs do libmtmd/clip vão para o logging do Python (só erros aparecem)
            from llama_cpp._logger import llama_log_callback  # type: ignore

            logging.getLogger("llama-cpp-python").setLevel(logging.ERROR)
            mtmd_cpp.mtmd_log_set(llama_log_callback, None)
            mtmd_cpp.mtmd_helper_log_set(llama_log_callback, None)
        except Exception:  # noqa: BLE001 - versões sem essas funções
            pass
    return _LLAMA


@dataclass(frozen=True)
class _Media:
    """Imagem já presente no KV cache (identificada pelo hash do arquivo)."""

    ident: str
    n_tokens: int
    n_pos: int


def _item_pos(item: Any) -> int:
    return item.n_pos if isinstance(item, _Media) else 1


def _item_cells(item: Any) -> int:
    return item.n_tokens if isinstance(item, _Media) else 1


def _make_handler_class() -> type:
    m = _LLAMA
    llama_cpp, lcf, mtmd, quiet = m["llama_cpp"], m["lcf"], m["mtmd"], m["quiet"]

    class VisionChatHandler(lcf.MTMDChatHandler):
        def __init__(self, clip_model_path: str, *, use_gpu: bool = True, image_max_tokens: int = 0,
                     prefix_cache: bool = True):
            super().__init__(clip_model_path=clip_model_path, verbose=False, use_gpu=use_gpu)
            self.image_max_tokens = image_max_tokens
            self.prefix_cache = prefix_cache
            self.reserve_tokens = 64            # espaço mínimo exigido para a resposta
            self._kv_items: list[Any] = []      # o que está no KV cache: tokens (int) e imagens (_Media)
            self._kv_pos = 0                    # posições ocupadas por _kv_items
            self._kv_cells = 0                  # células de KV ocupadas pelo prompt
            self.stats: dict[str, Any] = {}
            self.starts_in_thinking = False

        # -- contexto multimodal: flash-attn automático + limite de tokens por imagem
        def _init_mtmd_context(self, llama_model: Any) -> None:
            if self.mtmd_ctx is not None:
                return
            with quiet(disable=False):
                params = mtmd.mtmd_context_params_default()
                params.use_gpu = self.use_gpu
                params.print_timings = False
                params.n_threads = llama_model.n_threads
                params.flash_attn_type = llama_cpp.LLAMA_FLASH_ATTN_TYPE_AUTO
                if self.image_max_tokens > 0:
                    params.image_max_tokens = self.image_max_tokens
                self.mtmd_ctx = mtmd.mtmd_init_from_file(self.clip_model_path.encode(), llama_model.model, params)
            if self.mtmd_ctx is None:
                raise EngineError(f"Falha ao carregar o projetor de visão: {Path(self.clip_model_path).name}")
            if not mtmd.mtmd_support_vision(self.mtmd_ctx):
                mtmd.mtmd_free(self.mtmd_ctx)
                self.mtmd_ctx = None
                raise EngineError("O arquivo mmproj não contém encoder de visão.")

            def _free() -> None:
                if self.mtmd_ctx is not None:
                    mtmd.mtmd_free(self.mtmd_ctx)
                    self.mtmd_ctx = None

            llama_model._stack.callback(_free)

        # -- renderização do chat template embutido no GGUF
        def _render(self, llama: Any, messages: list, media_marker: str, image_urls: list[str],
                     template_kwargs: dict[str, Any]) -> str:
            env = lcf.ImmutableSandboxedEnvironment(
                trim_blocks=True, lstrip_blocks=True,
                extensions=[lcf.Jinja2ChatFormatter.IgnoreGenerationTags, lcf.jinja2.ext.loopcontrols],
            )
            env.filters["tojson"] = lcf.Jinja2ChatFormatter.tojson
            template = env.from_string(self._get_chat_template(llama))

            def raise_exception(message: str) -> None:
                raise ValueError(message)

            bos = self._special_text(llama, llama.token_bos())
            eos = self._special_text(llama, llama.token_eos())
            text = template.render(
                messages=self._get_template_messages(messages, media_marker),
                add_generation_prompt=True, bos_token=bos, eos_token=eos,
                raise_exception=raise_exception, strftime_now=lcf.Jinja2ChatFormatter.strftime_now,
                **template_kwargs,
            )
            text = self._postprocess_template_text(text, image_urls, media_marker)
            # Evita BOS duplicado: o tokenizer já adiciona o BOS quando o modelo pede.
            try:
                adds_bos = bool(llama._model.add_bos_token())
            except Exception:  # noqa: BLE001
                adds_bos = True
            if adds_bos and bos and text.startswith(bos):
                text = text[len(bos):]
            return text

        def _special_text(self, llama: Any, token: int) -> str:
            if token is None or token < 0:
                return ""
            try:
                return self._decode_token_piece(llama.detokenize([token], special=True))
            except Exception:  # noqa: BLE001
                return ""

        def _tokenize(self, text: str, image_urls: list[str]) -> tuple[Any, list[Any], dict[int, Any], list[Any]]:
            bitmaps = []
            try:
                for url in image_urls:
                    data = self.load_image(url)
                    bmp = self._create_bitmap_from_bytes(data)
                    bitmaps.append(bmp)
                    mtmd.mtmd_bitmap_set_id(bmp, hashlib.sha1(data).hexdigest().encode())
                inp = mtmd.mtmd_input_text()
                raw = text.encode("utf-8")
                inp.text, inp.text_len, inp.add_special, inp.parse_special = raw, len(raw), True, True
                chunks = mtmd.mtmd_input_chunks_init()
                if chunks is None:
                    raise EngineError("mtmd_input_chunks_init falhou")
                arr = (mtmd.mtmd_bitmap_p_ctypes * len(bitmaps))(*bitmaps)
                res = mtmd.mtmd_tokenize(self.mtmd_ctx, chunks, ctypes.byref(inp), arr, len(bitmaps))
                if res != 0:
                    mtmd.mtmd_input_chunks_free(chunks)
                    raise EngineError(f"Falha ao tokenizar a entrada multimodal (código {res})")
            except BaseException:
                for bmp in bitmaps:
                    mtmd.mtmd_bitmap_free(bmp)
                raise
            items: list[Any] = []
            chunk_of: dict[int, Any] = {}
            for i in range(mtmd.mtmd_input_chunks_size(chunks)):
                chunk = mtmd.mtmd_input_chunks_get(chunks, i)
                if chunk is None:
                    continue
                if mtmd.mtmd_input_chunk_get_type(chunk) == mtmd.MTMD_INPUT_CHUNK_TYPE_TEXT:
                    n = ctypes.c_size_t()
                    ptr = mtmd.mtmd_input_chunk_get_tokens_text(chunk, ctypes.byref(n))
                    items.extend(int(ptr[j]) for j in range(n.value))
                else:
                    ident = (mtmd.mtmd_input_chunk_get_id(chunk) or b"").decode(errors="replace") or f"media{i}"
                    items.append(_Media(ident, int(mtmd.mtmd_input_chunk_get_n_tokens(chunk)),
                                        int(mtmd.mtmd_input_chunk_get_n_pos(chunk))))
                    chunk_of[len(items) - 1] = chunk
            return chunks, items, chunk_of, bitmaps

        def _current_kv_items(self, llama: Any) -> list[Any]:
            """Reconstrói o conteúdo do KV: prompt anterior + tokens gerados depois dele."""
            if not self._kv_items or llama.n_tokens <= 0:
                return []
            if llama.n_tokens >= self._kv_pos:
                return self._kv_items + [int(t) for t in llama.input_ids[self._kv_pos:llama.n_tokens]]
            out, pos = [], 0
            for item in self._kv_items:
                if pos + _item_pos(item) > llama.n_tokens:
                    break
                out.append(item)
                pos += _item_pos(item)
            return out

        def _reset_kv(self, llama: Any) -> None:
            llama.reset()
            llama._ctx.kv_cache_clear()
            llama.n_tokens = 0
            self._kv_items, self._kv_pos, self._kv_cells = [], 0, 0

        def _prepare(self, llama: Any, messages: list, template_kwargs: dict[str, Any]) -> list[int]:
            self._init_mtmd_context(llama)
            image_urls = self.get_image_urls(messages)
            marker = mtmd.mtmd_default_marker().decode("utf-8")
            text = self._render(llama, messages, marker, image_urls, template_kwargs)
            self.starts_in_thinking = text.rstrip().endswith("<think>")
            chunks, items, chunk_of, bitmaps = self._tokenize(text, image_urls)
            try:
                cells = sum(_item_cells(it) for it in items)
                if cells + self.reserve_tokens > llama.n_ctx():
                    raise ContextOverflowError(cells + self.reserve_tokens, llama.n_ctx())

                # Maior prefixo em comum com o que já está no KV cache.
                keep, keep_pos = 0, 0
                if self.prefix_cache:
                    for old, new in zip(self._current_kv_items(llama), items):
                        if old != new:
                            break
                        keep += 1
                        keep_pos += _item_pos(new)
                    if keep == len(items) and keep > 0:  # prompt idêntico: reavalia o último token
                        keep -= 1
                        keep_pos -= _item_pos(items[keep])
                if keep_pos > 0 and not self._truncate_kv(llama, keep_pos):
                    keep, keep_pos = 0, 0
                if keep_pos == 0:
                    keep = 0
                    self._reset_kv(llama)
                llama.n_tokens = keep_pos

                t0 = time.perf_counter()
                i = keep
                while i < len(items):
                    if isinstance(items[i], _Media):
                        new_pos = llama_cpp.llama_pos(0)
                        res = mtmd.mtmd_helper_eval_chunk_single(
                            self.mtmd_ctx, llama._ctx.ctx, chunk_of[i], llama_cpp.llama_pos(llama.n_tokens),
                            llama_cpp.llama_seq_id(0), llama.n_batch, False, ctypes.byref(new_pos))
                        if res != 0:
                            self._kv_items = []
                            raise EngineError(f"Falha ao processar a imagem no modelo (código {res})")
                        llama.n_tokens = new_pos.value
                        i += 1
                    else:
                        j = i
                        while j < len(items) and not isinstance(items[j], _Media):
                            j += 1
                        llama.eval(items[i:j])
                        i = j
                self._kv_items, self._kv_pos, self._kv_cells = items, llama.n_tokens, cells
                self.stats = {
                    "prompt_tokens": cells,
                    "cached_tokens": sum(_item_cells(it) for it in items[:keep]),
                    "prompt_seconds": time.perf_counter() - t0,
                    "images": sum(isinstance(it, _Media) for it in items),
                }
                return [int(t) for t in llama.input_ids[: llama.n_tokens]]
            finally:
                mtmd.mtmd_input_chunks_free(chunks)
                for bmp in bitmaps:
                    mtmd.mtmd_bitmap_free(bmp)

        def _truncate_kv(self, llama: Any, keep_pos: int) -> bool:
            """Remove do KV tudo após keep_pos. False => é preciso reprocessar do zero."""
            if keep_pos >= llama.n_tokens:
                return True  # só acrescenta tokens: nada a remover
            try:
                if not llama._ctx.kv_cache_seq_rm(0, keep_pos, -1):
                    return False  # modelos recorrentes/híbridos não suportam remoção parcial
                n_swa = int(llama_cpp.llama_model_n_swa(llama._model.model))
                if n_swa > 0:
                    mem = llama_cpp.llama_get_memory(llama._ctx.ctx)
                    pos_min = int(llama_cpp.llama_memory_seq_pos_min(mem, 0))
                    # Cache com janela deslizante já descartou tokens necessários?
                    if pos_min > max(0, keep_pos - n_swa):
                        return False
                return True
            except Exception:  # noqa: BLE001
                return False

        def __call__(self, *, llama: Any, messages: list, functions: Any = None, function_call: Any = None,
                     tools: Any = None, tool_choice: Any = None, temperature: float = 0.2, top_p: float = 0.95,
                     top_k: int = 40, min_p: float = 0.05, typical_p: float = 1.0, stream: bool = False,
                     stop: Any = None, seed: Any = None, response_format: Any = None, max_tokens: Any = None,
                     presence_penalty: float = 0.0, frequency_penalty: float = 0.0, repeat_penalty: float = 1.0,
                     tfs_z: float = 1.0, mirostat_mode: int = 0, mirostat_tau: float = 5.0,
                     mirostat_eta: float = 0.1, model: Any = None, logits_processor: Any = None,
                     grammar: Any = None, logit_bias: Any = None, logprobs: Any = None, top_logprobs: Any = None,
                     **kwargs: Any) -> Any:
            template_kwargs = dict(kwargs)
            template_kwargs.update(tools=tools, tool_choice=tool_choice, functions=functions,
                                   function_call=function_call)
            try:
                prompt = self._prepare(llama, messages, template_kwargs)
            except (ContextOverflowError, EngineError):
                raise
            except Exception as e:  # noqa: BLE001 - inconsistência no cache: recomeça limpo
                log.warning("cache de prefixo desativado nesta chamada: %s", e)
                self._reset_kv(llama)
                if isinstance(e, ValueError) and "exceed" in str(e).lower():
                    raise ContextOverflowError(llama.n_ctx(), llama.n_ctx()) from e
                prompt = self._prepare(llama, messages, template_kwargs)

            room = llama.n_ctx() - self._kv_cells - 1
            if max_tokens is None or max_tokens <= 0 or max_tokens > room:
                max_tokens = max(1, room)
            completion = llama.create_completion(
                prompt=prompt, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
                typical_p=typical_p, logprobs=top_logprobs if logprobs else None, stream=stream,
                stop=stop or [], seed=seed, max_tokens=max_tokens, presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty, repeat_penalty=repeat_penalty, tfs_z=tfs_z,
                mirostat_mode=mirostat_mode, mirostat_tau=mirostat_tau, mirostat_eta=mirostat_eta,
                model=model, logits_processor=logits_processor, grammar=grammar, logit_bias=logit_bias,
            )
            return lcf._convert_completion_to_chat(completion, stream=stream)

    return VisionChatHandler


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    prompt_seconds: float = 0.0
    completion_tokens: int = 0
    gen_seconds: float = 0.0
    finish_reason: str = ""

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.gen_seconds if self.gen_seconds > 0 else 0.0

    def describe(self) -> str:
        cache = f" ({self.cached_tokens} em cache)" if self.cached_tokens else ""
        return (f"{self.tokens_per_second:.1f} tok/s · {self.completion_tokens} tokens gerados · "
                f"prompt {self.prompt_tokens} tokens{cache} em {self.prompt_seconds:.1f}s")


@dataclass
class LoadedModel:
    entry: ModelEntry
    info: GGUFInfo
    mm_info: GGUFInfo
    plan: OffloadPlan
    n_ctx: int
    n_threads: int
    backend: str
    native_tools: bool
    tool_role: bool
    load_seconds: float
    notes: list[str] = field(default_factory=list)


class LlamaEngine:
    """Mantém um único modelo carregado e serializa o acesso a ele."""

    def __init__(self) -> None:
        self.llm: Any = None
        self.handler: Any = None
        self.loaded: LoadedModel | None = None
        self.lock = threading.RLock()
        self.last_stats = GenerationStats()
        self._gguf_cache: dict[tuple[str, float], GGUFInfo] = {}
        self.profile = load_hardware_profile()

    # ---------------------------------------------------------------- info
    def gguf(self, path: Path) -> GGUFInfo:
        key = (str(path), path.stat().st_mtime)
        if key not in self._gguf_cache:
            self._gguf_cache[key] = read_gguf(path)
        return self._gguf_cache[key]

    def available(self) -> tuple[bool, str]:
        try:
            _llama_modules()
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, f"llama-cpp-python indisponível: {e}"

    def gpu_budget(self) -> tuple[str, int | None, int | None]:
        """(backend, VRAM livre, VRAM total) medidos agora pelo próprio ggml."""
        installed = self.profile.get("installed_backend") or hardware.installed_backend()
        try:
            _llama_modules()
            devs = [d for d in hardware.llama_devices() if d["type"] in ("gpu", "igpu")]
        except Exception:  # noqa: BLE001
            devs = []
        if devs:
            gpus = [d for d in devs if d["type"] == "gpu"] or devs
            name = gpus[0]["name"].lower()
            backend = ("cuda" if name.startswith(("cuda", "rocm", "hip")) else
                       "metal" if name.startswith(("metal", "mtl")) else "vulkan")
            return backend, sum(d["free_mb"] for d in gpus), sum(d["total_mb"] for d in gpus)
        if installed and installed != "cpu":
            log.warning("build '%s' instalada, mas nenhuma GPU foi encontrada pelo llama.cpp", installed)
        return "cpu", 0, 0

    def gpu_note(self) -> str:
        """Explica por que uma GPU dedicada detectada não está sendo usada ('' se estiver)."""
        gpus = [g for g in self.profile.get("gpus", []) if not g.get("integrated")]
        if not gpus:
            return ""
        installed = hardware.installed_backend() or self.profile.get("installed_backend", "")
        if installed in ("", "cpu"):
            return ("A instalação do llama.cpp ficou em modo CPU; o motivo está em logs/instalacao.log. "
                    "Atualize o app e rode o start.bat de novo.")
        if gpu_plugins.is_plugin_backend(installed) and not gpu_plugins._LOADED:
            return f"O plugin de GPU '{installed}' não carregou: {gpu_plugins.LAST_ERROR or 'veja logs/app.log'}"
        return ""

    def threads(self) -> int:
        rec = self.profile.get("recommended") or hardware.recommend_threads(hardware.detect_cpu())
        return int(rec.get("n_threads") or 4)

    def preview_plan(self, entry: ModelEntry, n_ctx: int, settings: Settings) -> OffloadPlan:
        """Estimativa rápida para a UI (usa a VRAM livre medida pelo start.bat)."""
        info, mm = self.gguf(entry.path), self.gguf(entry.mmproj) if entry.mmproj else None
        plan = plan_offload(info, mm, n_ctx, self.profile, kv_pref=settings.kv_cache,
                            vision_pref="auto" if settings.vision_on_gpu else "cpu")
        return _apply_manual_layers(plan, settings)

    # ------------------------------------------------------------- carga
    def unload(self) -> None:
        with self.lock:
            if self.llm is not None:
                try:
                    self.llm.close()
                except Exception:  # noqa: BLE001
                    pass
            self.llm = None
            self.handler = None
            self.loaded = None
            gc.collect()

    def load(self, entry: ModelEntry, n_ctx: int, settings: Settings) -> LoadedModel:
        if entry.mmproj is None:
            raise EngineError(f"'{entry.key}' não tem arquivo mmproj: este app exige um modelo de visão (VLM).")
        mods = _llama_modules()
        with self.lock:
            self.unload()
            t0 = time.perf_counter()
            info, mm = self.gguf(entry.path), self.gguf(entry.mmproj)
            n_ctx = max(1024, min(int(n_ctx), info.n_ctx_train or int(n_ctx)))
            backend, free_mb, total_mb = self.gpu_budget()
            plan = plan_offload(info, mm, n_ctx, self.profile, backend=backend, free_vram_mb=free_mb,
                                total_vram_mb=total_mb, kv_pref=settings.kv_cache,
                                vision_pref="auto" if settings.vision_on_gpu else "cpu")
            plan = _apply_manual_layers(plan, settings)
            n_threads = self.threads()
            errors: list[str] = []
            for ngl in _retry_layers(plan):
                try:
                    llm, handler = self._create(mods, entry, n_ctx, ngl, plan, n_threads, settings)
                except Exception as e:  # noqa: BLE001 - OOM, arquivo inválido, etc.
                    errors.append(f"n_gpu_layers={ngl}: {e}")
                    log.warning("falha ao carregar com n_gpu_layers=%s: %s", ngl, e)
                    gc.collect()
                    continue
                if ngl != plan.n_gpu_layers:
                    plan.notes.append(f"Carregado com {ngl} camadas na GPU após falta de memória.")
                    plan.gpu_layers = min(ngl, plan.n_layers)
                    plan.n_gpu_layers = ngl
                    plan.full_offload = ngl > plan.n_layers
                    if ngl == 0:
                        plan.backend = "cpu"
                self.llm, self.handler = llm, handler
                self.loaded = LoadedModel(
                    entry=entry, info=info, mm_info=mm, plan=plan, n_ctx=n_ctx, n_threads=n_threads,
                    backend=plan.backend, native_tools=info.supports_native_tools(),
                    tool_role=info.supports_tool_role(), load_seconds=time.perf_counter() - t0,
                    notes=list(plan.notes),
                )
                return self.loaded
            raise EngineError("Não foi possível carregar o modelo:\n" + "\n".join(errors[-3:]))

    def _create(self, mods: dict[str, Any], entry: ModelEntry, n_ctx: int, ngl: int, plan: OffloadPlan,
                n_threads: int, settings: Settings) -> tuple[Any, Any]:
        llama_cpp = mods["llama_cpp"]
        handler = mods["Handler"](
            str(entry.mmproj), use_gpu=bool(plan.vision_on_gpu and ngl > 0),
            image_max_tokens=max(256, min(1024, n_ctx // 4)), prefix_cache=settings.prefix_cache,
        )
        kv = KV_GGML_TYPE.get(plan.kv_type, 1)
        # O llama-cpp-python só aceita flash_attn True/False; "auto" (o llama.cpp decide
        # conforme o suporte da GPU) é selecionado trocando a constante durante a criação.
        cpp = llama_cpp.llama_cpp
        enabled = cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED
        cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED = cpp.LLAMA_FLASH_ATTN_TYPE_AUTO
        try:
            llm = mods["Llama"](
                model_path=str(entry.path), n_gpu_layers=ngl, n_ctx=n_ctx, n_batch=N_BATCH, n_ubatch=N_BATCH,
                n_threads=n_threads, n_threads_batch=n_threads, flash_attn=True, offload_kqv=True,
                type_k=kv, type_v=kv, use_mmap=True, chat_handler=handler, verbose=False,
            )
        finally:
            cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED = enabled
        try:
            handler._init_mtmd_context(llm)  # carrega o encoder de visão já (erro aparece agora)
        except Exception:
            llm.close()
            raise
        return llm, handler

    # ------------------------------------------------------------- geração
    def chat_stream(self, messages: list[dict[str, Any]], *, tools: list[dict] | None = None,
                    temperature: float = 0.6, max_tokens: int = 2048,
                    cancel: threading.Event | None = None) -> Iterator[str]:
        """Gera a resposta em streaming (pedaços de texto)."""
        with self.lock:
            if self.llm is None:
                raise EngineError("Nenhum modelo carregado.")
            stats = GenerationStats()
            self.last_stats = stats
            stream = self.llm.create_chat_completion(
                messages=messages, tools=tools or None, tool_choice="auto" if tools else None,
                temperature=temperature, top_p=0.95, top_k=40, min_p=0.05, repeat_penalty=1.0,
                max_tokens=max_tokens, stream=True,
            )
            h = self.handler.stats if self.handler is not None else {}
            stats.prompt_tokens = int(h.get("prompt_tokens", 0))
            stats.cached_tokens = int(h.get("cached_tokens", 0))
            stats.prompt_seconds = float(h.get("prompt_seconds", 0.0))
            t_gen = time.perf_counter()
            try:
                for chunk in stream:
                    choice = chunk["choices"][0]
                    text = (choice.get("delta") or {}).get("content") or ""
                    if choice.get("finish_reason"):
                        stats.finish_reason = choice["finish_reason"]
                    if text:
                        stats.completion_tokens += 1
                        yield text
                    if cancel is not None and cancel.is_set():
                        stats.finish_reason = "cancelled"
                        break
            finally:
                stats.gen_seconds = time.perf_counter() - t_gen
                close = getattr(stream, "close", None)
                if close:
                    close()

    @property
    def starts_in_thinking(self) -> bool:
        return bool(self.handler is not None and self.handler.starts_in_thinking)


def _apply_manual_layers(plan: OffloadPlan, settings: Settings) -> OffloadPlan:
    value = str(settings.gpu_layers).strip().lower()
    if value in ("", "auto"):
        return plan
    try:
        ngl = max(0, int(value))
    except ValueError:
        return plan
    plan.n_gpu_layers = ngl
    plan.gpu_layers = min(ngl, plan.n_layers)
    plan.full_offload = ngl > plan.n_layers
    plan.notes.append(f"Camadas na GPU definidas manualmente: {ngl}.")
    return plan


def _retry_layers(plan: OffloadPlan) -> list[int]:
    """Plano principal e, se faltar memória, versões com menos camadas (até CPU)."""
    first = plan.n_gpu_layers
    if first <= 0:
        return [0]
    seq, n = [first], min(first, plan.n_layers)
    for frac in (0.85, 0.65, 0.4):
        cand = int(n * frac)
        if 0 < cand < seq[-1]:
            seq.append(cand)
    seq.append(0)
    return seq
