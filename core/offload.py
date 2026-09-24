"""Planejador de offload: quantas camadas cabem na GPU para o MÁXIMO de tokens/s.

Estimativa baseada no GGUF real (bytes exatos por camada) + KV cache + buffers:

    VRAM necessária = camadas na GPU (pesos + KV cache dessas camadas)
                    + camada de saída (só no offload total)
                    + buffer de computação + encoder de visão (mmproj) + reserva do driver

Estratégia (a velocidade despenca quando parte do modelo fica na CPU):
    1. Tenta 100% na GPU com KV f16 -> depois com KV q8_0 (metade da memória,
       perda de qualidade desprezível) -> depois movendo a visão para a CPU.
    2. Se nada disso couber: offload parcial com o máximo de camadas possível.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.gguf_info import GGUFInfo

MiB = 2**20
KV_BYTES_PER_ELEM = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}
KV_GGML_TYPE = {"f16": 1, "q8_0": 8, "q4_0": 2}  # enum ggml_type


@dataclass
class OffloadPlan:
    backend: str
    n_gpu_layers: int          # valor para Llama(n_gpu_layers=...)
    gpu_layers: int            # camadas repetidas na GPU
    n_layers: int
    full_offload: bool
    kv_type: str               # "f16" | "q8_0"
    vision_on_gpu: bool
    vram_budget_mb: int
    vram_estimate_mb: int
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        if self.backend == "cpu" or self.gpu_layers == 0:
            return "CPU (sem offload para GPU)"
        pct = 100 if self.full_offload else round(100 * self.gpu_layers / max(self.n_layers, 1))
        vis = "visão na GPU" if self.vision_on_gpu else "visão na CPU"
        return (f"{self.gpu_layers}/{self.n_layers} camadas na GPU ({pct}%) · KV {self.kv_type} · {vis} · "
                f"~{self.vram_estimate_mb / 1024:.1f}/{self.vram_budget_mb / 1024:.1f} GB")


def kv_bytes_per_layer(info: GGUFInfo, n_ctx: int, kv_type: str, n_ubatch: int = 512) -> list[int]:
    n = info.n_layers
    if n <= 0:
        return []
    bpe = KV_BYTES_PER_ELEM.get(kv_type, 2.0)
    k_dim, v_dim = info.head_dims()
    heads_kv = info.n_head_kv()
    swa = info.swa_layers()
    window = int(info.arch_get("attention.sliding_window", 0) or 0)
    mla_rank = int(info.arch_get("attention.kv_lora_rank", 0) or 0)
    # Modelos híbridos (atenção + camadas recorrentes/SSM): só algumas camadas têm KV.
    interval = int(info.arch_get("full_attention_interval", 0) or 0)
    out = []
    for i in range(n):
        if heads_kv[i] <= 0 or (interval > 1 and (i + 1) % interval != 0):
            out.append(4 * MiB)  # estado recorrente de tamanho fixo (pequeno)
            continue
        cells = min(n_ctx, window + n_ubatch) if (swa[i] and window > 0) else n_ctx
        per_token = (mla_rank + 64) if mla_rank else heads_kv[i] * (k_dim + v_dim)
        out.append(int(cells * per_token * bpe))
    return out


def compute_buffer_bytes(info: GGUFInfo, n_ctx: int, n_ubatch: int, output_on_gpu: bool,
                         flash_attn: bool = True) -> int:
    """Pico do buffer de computação do grafo (o alocador do ggml reaproveita memória,
    então vale o maior tensor intermediário, não a soma)."""
    ub = max(1, min(n_ubatch, n_ctx))
    n_embd = info.n_embd or 4096
    n_head = max(info.n_head() or [32]) or 32
    logits = ub * info.n_vocab * 4 + ub * n_embd * 8 if output_on_gpu else 0
    ffn = ub * info.n_ff() * 4 * 3
    if flash_attn:
        attn = ub * n_ctx * 2 + ub * n_embd * 16
    else:
        attn = int(ub * n_ctx * n_head * 4 * 1.1)
    return int(max(logits, ffn, attn) * 1.15) + 96 * MiB


def vision_bytes(mm: GGUFInfo | None) -> int:
    if mm is None:
        return 0
    compute = min(max(int(mm.file_size * 0.5), 192 * MiB), 1024 * MiB)
    return mm.file_size + compute


def _reserve_bytes(backend: str, total_mb: int) -> int:
    if backend == "cuda":
        return int((300 + 0.02 * total_mb) * MiB)
    if backend == "metal":
        return int((768 + 0.05 * total_mb) * MiB)
    return int((450 + 0.04 * total_mb) * MiB)  # vulkan / outros


def gpu_budget_from_profile(profile: dict[str, Any]) -> tuple[str, int, int]:
    """(backend, VRAM livre MB, VRAM total MB) a partir do perfil do start.bat."""
    installed = profile.get("installed_backend") or ""
    gpus = [g for g in profile.get("gpus", []) if not g.get("integrated")]
    backend = "cuda" if installed.startswith("cu") else (installed or "")
    if not backend:
        cands = profile.get("backend_candidates") or ["cpu"]
        backend = "cuda" if cands[0].startswith("cu") else cands[0]
    if backend == "cpu" or not gpus:
        return "cpu", 0, 0
    vendor = "nvidia" if backend == "cuda" else None
    pool = [g for g in gpus if vendor is None or g["vendor"] == vendor] or gpus
    total = sum(int(g.get("vram_total_mb") or 0) for g in pool)
    free = sum(int(g["vram_free_mb"]) if g.get("vram_free_mb") is not None
               else int(g.get("vram_total_mb", 0) * 0.92) for g in pool)
    return backend, free, total


def plan_offload(
    info: GGUFInfo,
    mmproj: GGUFInfo | None,
    n_ctx: int,
    profile: dict[str, Any] | None = None,
    *,
    backend: str | None = None,
    free_vram_mb: int | None = None,
    total_vram_mb: int | None = None,
    kv_pref: str = "auto",
    vision_pref: str = "auto",   # "auto" | "gpu" | "cpu"
    n_ubatch: int = 512,
) -> OffloadPlan:
    profile = profile or {}
    p_backend, p_free, p_total = gpu_budget_from_profile(profile)
    backend = backend or p_backend
    free_mb = free_vram_mb if free_vram_mb is not None else p_free
    total_mb = total_vram_mb if total_vram_mb is not None else (p_total or free_mb)
    n_layers = info.n_layers
    kv_default = "f16" if kv_pref == "auto" else kv_pref

    if backend == "cpu" or free_mb <= 0 or n_layers <= 0:
        return OffloadPlan("cpu", 0, 0, n_layers, False, kv_default, False, 0, 0,
                           ["Sem GPU utilizável: rodando na CPU."])

    budget = free_mb * MiB - _reserve_bytes(backend, total_mb)
    layers = info.layer_bytes
    vis = vision_bytes(mmproj)
    kv_options = ["f16", "q8_0"] if kv_pref == "auto" else [kv_pref]
    vision_options = {"auto": [True, False], "gpu": [True], "cpu": [False]}.get(vision_pref, [True, False])

    # 1) Offload total (todas as camadas + saída).
    for vision_gpu in vision_options:
        for kv in kv_options:
            need = (sum(layers) + info.output_bytes + sum(kv_bytes_per_layer(info, n_ctx, kv, n_ubatch))
                    + compute_buffer_bytes(info, n_ctx, n_ubatch, output_on_gpu=True)
                    + (vis if vision_gpu else 0))
            if need <= budget:
                notes = []
                if kv != "f16":
                    notes.append("KV cache q8_0 para caber 100% na GPU.")
                if not vision_gpu and mmproj is not None:
                    notes.append("Encoder de visão na CPU para liberar VRAM para o modelo.")
                return OffloadPlan(backend, n_layers + 1, n_layers, n_layers, True, kv, vision_gpu,
                                   int(budget / MiB), int(need / MiB), notes)

    # 2) Offload parcial: o llama.cpp coloca na GPU as ÚLTIMAS camadas.
    kv = "q8_0" if kv_pref == "auto" else kv_pref
    vision_gpu = vision_options[0]
    kv_layers = kv_bytes_per_layer(info, n_ctx, kv, n_ubatch)
    avail = budget - compute_buffer_bytes(info, n_ctx, n_ubatch, output_on_gpu=False) - (vis if vision_gpu else 0)
    if avail <= 0 and vision_gpu and len(vision_options) > 1:
        vision_gpu = False
        avail = budget - compute_buffer_bytes(info, n_ctx, n_ubatch, output_on_gpu=False)
    used, count = 0, 0
    for i in range(n_layers - 1, -1, -1):
        cost = layers[i] + kv_layers[i]
        if used + cost > avail:
            break
        used += cost
        count += 1
    notes = [f"Modelo não cabe inteiro na VRAM: {count}/{n_layers} camadas na GPU, o resto na CPU "
             "(reduza o contexto ou use uma quantização menor para ganhar velocidade)."]
    estimate = used + (budget - avail) if count else 0
    return OffloadPlan(backend if count else "cpu", count, count, n_layers, False, kv,
                       vision_gpu and count > 0, int(budget / MiB), int(max(estimate, 0) / MiB), notes)


def max_context_for_full_offload(info: GGUFInfo, mmproj: GGUFInfo | None, profile: dict[str, Any], *,
                                 backend: str | None = None, free_vram_mb: int | None = None,
                                 total_vram_mb: int | None = None, kv_pref: str = "auto") -> int:
    """Maior contexto (múltiplo de 1024) que ainda mantém 100% do modelo na GPU."""
    limit = info.n_ctx_train or 32768
    lo, hi, best = 1024, max(limit, 1024), 0
    while lo <= hi:
        mid = (lo + hi) // 2 // 1024 * 1024 or 1024
        plan = plan_offload(info, mmproj, mid, profile, backend=backend, free_vram_mb=free_vram_mb,
                            total_vram_mb=total_vram_mb, kv_pref=kv_pref, vision_pref="gpu")
        if plan.full_offload:
            best, lo = mid, mid + 1024
        else:
            hi = mid - 1024
    return best
