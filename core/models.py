"""Descoberta de modelos GGUF em models/ e pareamento com o projetor de visão (mmproj).

Um VLM no llama.cpp = GGUF do modelo de linguagem + GGUF "mmproj" (encoder de
visão/projetor). Somente pares completos são oferecidos na UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from core.config import MODELS_DIR
from core.gguf_info import is_secondary_split, try_read_gguf

_QUANT_TOKENS = re.compile(
    r"^(i?q\d(_[a-z0-9]+)*|f16|bf16|f32|fp16|fp32|gguf|mmproj|model|ud|imat|imatrix|it|instruct|chat|vision|vl)$"
)


@dataclass
class ModelEntry:
    key: str             # caminho relativo a models/ (valor do seletor)
    path: Path
    mmproj: Path | None
    size_mb: float

    @property
    def label(self) -> str:
        return f"{self.key}  ({self.size_mb / 1024:.1f} GB)"


def _is_mmproj(path: Path) -> bool:
    name = path.name.lower()
    if "mmproj" in name:
        return True
    try:
        if path.stat().st_size > 3 * 2**30:
            return False
    except OSError:
        return False
    if re.search(r"clip|vision|projector|proj", name):
        info = try_read_gguf(path)
        return bool(info and info.is_mmproj)
    return False


def _tokens(path: Path) -> set[str]:
    """Palavras do nome do arquivo, sem marcas de quantização (Q4_K_M, IQ4_XS, F16...)."""
    stem = path.name.lower().removesuffix(".gguf")
    out: set[str] = set()
    for part in re.split(r"[-.\s]+", stem):
        if not part or _QUANT_TOKENS.match(part):
            continue
        out.update(sub for sub in part.split("_") if sub and not _QUANT_TOKENS.match(sub))
    return out


def _similarity(a: Path, b: Path) -> float:
    ta, tb = _tokens(a), _tokens(b)
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def find_mmproj(model_path: Path) -> Path | None:
    """Escolhe o mmproj da mesma pasta que corresponde ao modelo.

    Com vários GGUFs na pasta, cada mmproj vai para o(s) modelo(s) de nome mais
    parecido: assim um GGUF só-texto não herda o projetor de outro modelo.
    """
    folder = model_path.parent
    ggufs = [p for p in folder.glob("*.gguf") if not is_secondary_split(p)]
    candidates = [p for p in ggufs if p != model_path and _is_mmproj(p)]
    if not candidates:
        return None
    siblings = [p for p in ggufs if p not in candidates]
    best = max(candidates, key=lambda c: _similarity(model_path, c))
    score = _similarity(model_path, best)
    if len(siblings) <= 1:
        return best  # modelo único na pasta: usa o (melhor) mmproj disponível
    top = max(_similarity(s, best) for s in siblings)
    if score >= top and (score > 0 or top == 0):
        return best
    return None


def list_models(models_dir: Path = MODELS_DIR) -> tuple[list[ModelEntry], list[Path]]:
    """Retorna (modelos VLM prontos, GGUFs sem mmproj)."""
    ready: list[ModelEntry] = []
    orphans: list[Path] = []
    if not models_dir.is_dir():
        return ready, orphans
    for path in sorted(models_dir.rglob("*.gguf"), key=lambda p: str(p).lower()):
        if is_secondary_split(path) or _is_mmproj(path):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        mmproj = find_mmproj(path)
        if mmproj is None:
            orphans.append(path)
            continue
        key = path.relative_to(models_dir).as_posix()
        ready.append(ModelEntry(key=key, path=path, mmproj=mmproj, size_mb=size / 2**20))
    return ready, orphans
