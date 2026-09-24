"""Leitor leve de metadados GGUF (somente biblioteca padrão).

Lê o cabeçalho (metadados + tabela de tensores) via mmap, sem carregar pesos,
para descobrir: arquitetura, nº de camadas, dimensões da atenção (para o KV
cache), tamanho exato em bytes de cada camada e o chat template embutido.
Esses números alimentam o planejador de offload (core/offload.py).
"""

from __future__ import annotations

import mmap
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GGUF_MAGIC = b"GGUF"

# gguf value types -> formato struct
_SCALAR_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_T_STRING, _T_ARRAY = 8, 9

# Arrays maiores que isto não são materializados (ex.: vocabulário com 250k tokens).
_MAX_NUMERIC_ARRAY = 16384
_MAX_STRING_ARRAY = 512

_SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_LAYER_RE = re.compile(r"^blk\.(\d+)\.")


class GGUFError(Exception):
    pass


@dataclass
class ArrayInfo:
    """Array grande não materializado: só guardamos o tipo e o tamanho."""

    item_type: int
    count: int


@dataclass
class GGUFInfo:
    path: Path
    file_size: int
    metadata: dict[str, Any]
    arch: str = ""
    n_layers: int = 0
    layer_bytes: list[int] = field(default_factory=list)
    output_bytes: int = 0       # camada de saída (ou token_embd duplicado se "tied")
    embd_bytes: int = 0         # token_embd (sempre fica na CPU no llama.cpp)
    other_bytes: int = 0
    tensor_count: int = 0

    # ---- acesso a metadados -------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def arch_get(self, suffix: str, default: Any = None) -> Any:
        return self.metadata.get(f"{self.arch}.{suffix}", default)

    @property
    def is_mmproj(self) -> bool:
        return self.arch == "clip" or any(k.startswith("clip.") for k in self.metadata)

    @property
    def name(self) -> str:
        return str(self.get("general.name") or self.path.stem)

    @property
    def chat_template(self) -> str:
        tpl = self.get("tokenizer.chat_template")
        return tpl if isinstance(tpl, str) else ""

    @property
    def n_embd(self) -> int:
        return _as_int(self.arch_get("embedding_length"), 0)

    @property
    def n_ctx_train(self) -> int:
        return _as_int(self.arch_get("context_length"), 0)

    @property
    def n_vocab(self) -> int:
        toks = self.get("tokenizer.ggml.tokens")
        if isinstance(toks, ArrayInfo):
            return toks.count
        if isinstance(toks, list):
            return len(toks)
        return _as_int(self.arch_get("vocab_size"), 32000)

    def per_layer(self, suffix: str, default: int) -> list[int]:
        """Valor por camada (alguns modelos guardam arrays por camada)."""
        v = self.arch_get(suffix)
        n = max(self.n_layers, 1)
        if isinstance(v, list) and v:
            vals = [_as_int(x, default) for x in v]
            return (vals + [vals[-1]] * n)[:n]
        return [_as_int(v, default)] * n

    def n_head(self) -> list[int]:
        return self.per_layer("attention.head_count", 1)

    def n_head_kv(self) -> list[int]:
        v = self.arch_get("attention.head_count_kv")
        if v is None:
            return self.n_head()
        return self.per_layer("attention.head_count_kv", 1)

    def n_ff(self) -> int:
        return max(self.per_layer("feed_forward_length", 4 * max(self.n_embd, 1)) or [0])

    def head_dims(self) -> tuple[int, int]:
        heads = max(self.n_head() or [1]) or 1
        default = self.n_embd // heads if heads else 128
        k = _as_int(self.arch_get("attention.key_length"), default)
        v = _as_int(self.arch_get("attention.value_length"), k)
        return k, v

    def swa_layers(self) -> list[bool]:
        """Quais camadas usam sliding-window attention (KV reduzido)."""
        n = self.n_layers
        window = _as_int(self.arch_get("attention.sliding_window"), 0)
        if window <= 0 or n <= 0:
            return [False] * n
        pattern = self.arch_get("attention.sliding_window_pattern")
        if isinstance(pattern, list) and pattern:
            return [bool(pattern[i % len(pattern)]) for i in range(n)]
        period = _as_int(pattern, 0)
        if period <= 0:
            # Padrões conhecidos do llama.cpp quando o GGUF não informa.
            period = {"gemma3": 6, "gemma2": 2, "cohere2": 4, "gpt-oss": 2}.get(self.arch, 0)
        if period <= 1:
            return [False] * n  # desconhecido: assume atenção completa (conservador)
        return [((i + 1) % period) != 0 for i in range(n)]

    def supports_native_tools(self) -> bool:
        tpl = self.chat_template
        return bool(tpl) and "tools" in tpl and ("tool_call" in tpl or "function" in tpl)

    def supports_tool_role(self) -> bool:
        tpl = self.chat_template
        return bool(tpl) and bool(re.search(r"""['"]tool['"]""", tpl))

    def summary(self) -> dict[str, Any]:
        return {
            "arch": self.arch,
            "name": self.name,
            "file_size_mb": round(self.file_size / 2**20, 1),
            "n_layers": self.n_layers,
            "n_ctx_train": self.n_ctx_train,
            "n_embd": self.n_embd,
        }


def _as_int(v: Any, default: int) -> int:
    try:
        if isinstance(v, list):
            v = max(v) if v else default
        return int(v)
    except (TypeError, ValueError):
        return default


class _Reader:
    def __init__(self, buf: mmap.mmap):
        self.buf = buf
        self.pos = 0

    def unpack(self, fmt: str) -> Any:
        vals = struct.unpack_from("<" + fmt, self.buf, self.pos)
        self.pos += struct.calcsize("<" + fmt)
        return vals[0] if len(vals) == 1 else vals

    def string(self) -> str:
        n = self.unpack("Q")
        s = self.buf[self.pos:self.pos + n]
        self.pos += n
        return s.decode("utf-8", errors="replace")

    def skip_string(self) -> None:
        (n,) = struct.unpack_from("<Q", self.buf, self.pos)
        self.pos += 8 + n

    def value(self, vtype: int) -> Any:
        if vtype in _SCALAR_FMT:
            return self.unpack(_SCALAR_FMT[vtype])
        if vtype == _T_STRING:
            return self.string()
        if vtype == _T_ARRAY:
            item_type = self.unpack("I")
            count = self.unpack("Q")
            if item_type in _SCALAR_FMT:
                fmt = _SCALAR_FMT[item_type]
                size = struct.calcsize(fmt) * count
                if count <= _MAX_NUMERIC_ARRAY:
                    vals = list(struct.unpack_from(f"<{count}{fmt}", self.buf, self.pos)) if count else []
                    self.pos += size
                    return vals
                self.pos += size
                return ArrayInfo(item_type, count)
            if item_type == _T_STRING:
                if count <= _MAX_STRING_ARRAY:
                    return [self.string() for _ in range(count)]
                for _ in range(count):
                    self.skip_string()
                return ArrayInfo(item_type, count)
            if item_type == _T_ARRAY:  # arrays aninhados: raros, apenas pula
                for _ in range(count):
                    self.value(_T_ARRAY)
                return ArrayInfo(item_type, count)
            raise GGUFError(f"tipo de array desconhecido: {item_type}")
        raise GGUFError(f"tipo de valor desconhecido: {vtype}")


def _parse_single(path: Path) -> tuple[dict[str, Any], list[tuple[str, int]], int, int]:
    """Retorna (metadados, [(nome_tensor, tamanho_bytes)], tamanho_arquivo, n_tensores)."""
    with open(path, "rb") as f:
        file_size = f.seek(0, 2)
        if file_size < 24:
            raise GGUFError("arquivo pequeno demais para ser GGUF")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            r = _Reader(buf)
            if bytes(buf[0:4]) != GGUF_MAGIC:
                raise GGUFError("assinatura GGUF ausente")
            r.pos = 4
            version = r.unpack("I")
            if version not in (2, 3):
                raise GGUFError(f"versão GGUF não suportada: {version}")
            n_tensors = r.unpack("Q")
            n_kv = r.unpack("Q")
            meta: dict[str, Any] = {}
            for _ in range(n_kv):
                key = r.string()
                vtype = r.unpack("I")
                meta[key] = r.value(vtype)
            infos: list[tuple[str, int]] = []
            for _ in range(n_tensors):
                name = r.string()
                n_dims = r.unpack("I")
                r.pos += 8 * n_dims  # dimensões
                r.pos += 4           # tipo ggml
                offset = r.unpack("Q")
                infos.append((name, offset))
            alignment = _as_int(meta.get("general.alignment"), 32) or 32
            data_start = (r.pos + alignment - 1) // alignment * alignment
    # Tamanho de cada tensor = distância até o próximo offset (dispensa a
    # tabela de tamanhos de cada tipo de quantização).
    ordered = sorted(infos, key=lambda t: t[1])
    sizes: list[tuple[str, int]] = []
    for i, (name, off) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else file_size - data_start
        sizes.append((name, max(0, end - off)))
    return meta, sizes, file_size, n_tensors


def split_parts(path: Path) -> list[Path]:
    """Lista os arquivos de um GGUF dividido (model-00001-of-00003.gguf)."""
    m = _SPLIT_RE.search(path.name)
    if not m:
        return [path]
    total = int(m.group(2))
    prefix = path.name[: m.start()]
    return [path.with_name(f"{prefix}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total + 1)]


def is_secondary_split(path: Path) -> bool:
    m = _SPLIT_RE.search(path.name)
    return bool(m) and int(m.group(1)) != 1


def read_gguf(path: str | Path) -> GGUFInfo:
    path = Path(path)
    meta, sizes, file_size, n_tensors = _parse_single(path)
    parts = split_parts(path)
    if len(parts) > 1:
        for extra in parts[1:]:
            if extra.exists():
                _m, s, fs, nt = _parse_single(extra)
                sizes.extend(s)
                file_size += fs
                n_tensors += nt

    info = GGUFInfo(path=path, file_size=file_size, metadata=meta, tensor_count=n_tensors)
    info.arch = str(meta.get("general.architecture", ""))
    info.n_layers = _as_int(meta.get(f"{info.arch}.block_count"), 0)

    layer_bytes = [0] * info.n_layers
    output = norm = embd = other = 0
    has_output = False
    for name, size in sizes:
        m = _LAYER_RE.match(name)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < info.n_layers:
                layer_bytes[idx] += size
            else:  # camadas extras (ex.: MTP/NextN) ficam junto da saída
                other += size
        elif name.startswith("output."):
            output += size
            has_output = True
        elif name.startswith("output_norm"):
            norm += size
        elif name.startswith("token_embd"):
            embd += size
        else:
            other += size
    info.layer_bytes = layer_bytes
    info.embd_bytes = embd
    # Embeddings "tied": o llama.cpp duplica token_embd como camada de saída.
    info.output_bytes = (output if has_output else embd) + norm
    info.other_bytes = other
    return info


def try_read_gguf(path: str | Path) -> GGUFInfo | None:
    try:
        return read_gguf(path)
    except (OSError, GGUFError, struct.error, ValueError, OverflowError):
        return None
