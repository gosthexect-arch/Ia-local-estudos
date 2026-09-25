"""Integração real com o llama.cpp usando um Qwen2-VL minúsculo de pesos aleatórios.

Os GGUFs em tests/fixtures foram gerados pelo convert_hf_to_gguf.py oficial a partir
de um modelo HF aleatório (texto + mmproj), então exercitam o caminho completo:
carga, encoder de visão (libmtmd), M-RoPE e o cache de prefixo do VisionChatHandler.
"""

import base64
import io
from pathlib import Path

import pytest

llama_cpp = pytest.importorskip("llama_cpp")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from core.config import Settings  # noqa: E402
from core.engine import ContextOverflowError, LlamaEngine  # noqa: E402
from core.models import ModelEntry  # noqa: E402
from core.tools import tool_schemas  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def _img(color, size=(96, 64)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="module")
def engine():
    entry = ModelEntry("Tiny-Qwen2-VL-F16.gguf", FIX / "Tiny-Qwen2-VL-F16.gguf",
                       FIX / "mmproj-Tiny-Qwen2-VL-F16.gguf", 0.3)
    eng = LlamaEngine()
    eng.profile = {}
    eng.load(entry, 2048, Settings())
    yield eng
    eng.unload()


def _logits(llm):
    return np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm.n_vocab(),)).copy()


def test_loads_with_native_tools(engine):
    assert engine.loaded is not None
    assert engine.loaded.native_tools and engine.loaded.tool_role


def test_prefix_cache_matches_fresh_evaluation(engine):
    m1 = [{"role": "system", "content": "Você é útil."},
          {"role": "user", "content": [{"type": "image_url", "image_url": {"url": _img((255, 0, 0))}},
                                       {"type": "text", "text": "Descreva."}]}]
    first = "".join(engine.chat_stream(m1, temperature=0.0, max_tokens=8))
    m2 = m1 + [{"role": "assistant", "content": first},
               {"role": "user", "content": [{"type": "image_url", "image_url": {"url": _img((0, 0, 255), (64, 96))}},
                                            {"type": "text", "text": "E esta?"}]}]
    llm, h = engine.llm, engine.handler
    h._prepare(llm, m2, {})
    cached = dict(h.stats)
    logits_cached, pos_cached = _logits(llm), llm.n_tokens
    h._reset_kv(llm)
    h._prepare(llm, m2, {})
    assert cached["cached_tokens"] > 0 and h.stats["cached_tokens"] == 0
    assert pos_cached == llm.n_tokens < h.stats["prompt_tokens"]  # M-RoPE: menos posições que tokens
    assert np.allclose(logits_cached, _logits(llm), atol=1e-4)
    # geração gulosa idêntica com e sem cache
    a = "".join(engine.chat_stream(m2, temperature=0.0, max_tokens=12))
    engine.handler._reset_kv(engine.llm)
    b = "".join(engine.chat_stream(m2, temperature=0.0, max_tokens=12))
    assert a == b


def test_tools_are_rendered_by_native_template(engine):
    msgs = [{"role": "user", "content": "oi"}]
    list(engine.chat_stream(msgs, max_tokens=2))
    plain = engine.last_stats.prompt_tokens
    list(engine.chat_stream(msgs, tools=tool_schemas(), max_tokens=2))
    assert engine.last_stats.prompt_tokens > plain + 100


def test_context_overflow_is_detected_before_evaluation(engine):
    with pytest.raises(ContextOverflowError):
        next(engine.chat_stream([{"role": "user", "content": "palavra " * 4000}], max_tokens=4))
    # o motor continua utilizável depois do erro
    assert "".join(engine.chat_stream([{"role": "user", "content": "oi"}], max_tokens=3)) is not None
