"""GGUF sintético (só cabeçalho + tensores pequenos) para validar leitor e planejador."""

import pytest

gguf = pytest.importorskip("gguf")
np = pytest.importorskip("numpy")

from core.gguf_info import read_gguf  # noqa: E402
from core.models import find_mmproj, list_models  # noqa: E402
from core.offload import MiB, kv_bytes_per_layer, max_context_for_full_offload, plan_offload  # noqa: E402


def make_llm(path, n_layers=4, n_embd=256, n_head=8, n_head_kv=2, vocab=1000, arch="llama", swa=0):
    w = gguf.GGUFWriter(str(path), arch)
    w.add_block_count(n_layers)
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(n_embd * 2)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_head_kv)
    w.add_context_length(32768)
    if swa:
        w.add_sliding_window(swa)
    w.add_token_list([f"t{i}" for i in range(vocab)])
    w.add_chat_template("{% for m in messages %}{{ m.content }}{% endfor %}{% if tools %}<tool_call>{% endif %}")
    w.add_tensor("token_embd.weight", np.zeros((vocab, n_embd), dtype=np.float16))
    for i in range(n_layers):
        w.add_tensor(f"blk.{i}.attn_q.weight", np.zeros((n_embd, n_embd), dtype=np.float16))
        w.add_tensor(f"blk.{i}.ffn_up.weight", np.zeros((n_embd * 2, n_embd), dtype=np.float16))
    w.add_tensor("output.weight", np.zeros((vocab, n_embd), dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def make_mmproj(path):
    w = gguf.GGUFWriter(str(path), "clip")
    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", "mlp")
    w.add_tensor("v.patch_embd.weight", np.zeros((64, 64), dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_reader_exact_layer_sizes(tmp_path):
    make_llm(tmp_path / "m.gguf")
    info = read_gguf(tmp_path / "m.gguf")
    assert info.arch == "llama" and info.n_layers == 4 and info.n_vocab == 1000
    per_layer = 256 * 256 * 2 + 512 * 256 * 2
    assert all(abs(b - per_layer) <= 64 for b in info.layer_bytes)  # padding de alinhamento
    assert info.output_bytes >= 1000 * 256 * 2 and info.embd_bytes >= 1000 * 256 * 2
    assert info.supports_native_tools() is True  # o template renderiza `tools` e <tool_call>
    assert info.head_dims() == (32, 32)


def test_kv_estimate_and_swa(tmp_path):
    make_llm(tmp_path / "full.gguf")
    make_llm(tmp_path / "swa.gguf", arch="gemma3", swa=1024, n_layers=6)
    full = read_gguf(tmp_path / "full.gguf")
    kv = kv_bytes_per_layer(full, 8192, "f16")
    assert kv[0] == 8192 * 2 * (32 + 32) * 2
    assert kv_bytes_per_layer(full, 8192, "q8_0")[0] < kv[0] * 0.6
    swa = read_gguf(tmp_path / "swa.gguf")
    layers = kv_bytes_per_layer(swa, 32768, "f16")
    assert layers[5] > layers[0] * 10  # só a 6ª camada é global na Gemma 3


def test_plan_full_partial_and_cpu(tmp_path):
    make_llm(tmp_path / "m.gguf", n_layers=8, n_embd=512)
    make_mmproj(tmp_path / "mmproj-m.gguf")
    info, mm = read_gguf(tmp_path / "m.gguf"), read_gguf(tmp_path / "mmproj-m.gguf")
    assert mm.is_mmproj
    big = plan_offload(info, mm, 4096, backend="cuda", free_vram_mb=8000, total_vram_mb=8192)
    assert big.full_offload and big.n_gpu_layers == 9 and big.kv_type == "f16"
    need = big.vram_estimate_mb
    small = plan_offload(info, mm, 4096, backend="cuda", free_vram_mb=need - 10 + 300, total_vram_mb=8192)
    assert 0 < small.gpu_layers <= 8
    cpu = plan_offload(info, mm, 4096, backend="cpu", free_vram_mb=0)
    assert cpu.n_gpu_layers == 0 and cpu.backend == "cpu"


def test_q8_kv_used_when_it_enables_full_offload(tmp_path):
    make_llm(tmp_path / "m.gguf", n_layers=8, n_embd=512, n_head_kv=8)
    info = read_gguf(tmp_path / "m.gguf")
    f16 = plan_offload(info, None, 65536, backend="cuda", free_vram_mb=100000, total_vram_mb=100000)
    tight_mb = f16.vram_estimate_mb - 400
    plan = plan_offload(info, None, 65536, backend="cuda", free_vram_mb=tight_mb + 300 + 2000, total_vram_mb=100000)
    assert plan.full_offload
    ctx = max_context_for_full_offload(info, None, {}, backend="cuda", free_vram_mb=3000, total_vram_mb=4096)
    assert ctx % 1024 == 0 and ctx > 0


def test_model_discovery_pairs_mmproj(tmp_path):
    make_llm(tmp_path / "Qwen3-VL-4B-Instruct-Q4_K_M.gguf")
    make_mmproj(tmp_path / "mmproj-Qwen3-VL-4B-Instruct-F16.gguf")
    sub = tmp_path / "gemma"
    sub.mkdir()
    make_llm(sub / "gemma-3-4b-it-Q4_K_M.gguf")
    make_mmproj(sub / "mmproj-model-f16.gguf")
    make_llm(tmp_path / "texto-apenas.gguf")
    ready, orphans = list_models(tmp_path)
    keys = {m.key: m.mmproj.name for m in ready}
    assert keys["Qwen3-VL-4B-Instruct-Q4_K_M.gguf"] == "mmproj-Qwen3-VL-4B-Instruct-F16.gguf"
    assert keys["gemma/gemma-3-4b-it-Q4_K_M.gguf"] == "mmproj-model-f16.gguf"
    assert find_mmproj(tmp_path / "texto-apenas.gguf") is None  # não herda o projetor do Qwen
    assert [p.name for p in orphans] == ["texto-apenas.gguf"]
    make_llm(tmp_path / "Qwen3-VL-4B-Instruct-Q8_0.gguf")  # 2ª quantização do mesmo modelo
    ready, _ = list_models(tmp_path)
    assert {m.key for m in ready} >= {"Qwen3-VL-4B-Instruct-Q4_K_M.gguf", "Qwen3-VL-4B-Instruct-Q8_0.gguf"}
    assert MiB == 2**20
