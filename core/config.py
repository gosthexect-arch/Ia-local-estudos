"""Caminhos, variáveis de ambiente (.env) e configurações persistentes da UI.

Somente biblioteca padrão (o start.bat importa este módulo antes do pip).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
WORKSPACE_DIR = ROOT_DIR / "workspace"
UPLOADS_DIR = WORKSPACE_DIR / "uploads"
LOGS_DIR = ROOT_DIR / "logs"
PERSONALITY_FILE = ROOT_DIR / "personality.txt"
ENV_FILE = ROOT_DIR / ".env"
HARDWARE_PROFILE_FILE = ROOT_DIR / "hardware_profile.json"
SETTINGS_FILE = ROOT_DIR / "settings.json"

DEFAULT_PERSONALITY = (
    "Você é um assistente de IA local, prestativo, direto e honesto. "
    "Responda em português do Brasil."
)


def ensure_dirs() -> None:
    for d in (MODELS_DIR, WORKSPACE_DIR, UPLOADS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_env(path: Path = ENV_FILE) -> None:
    """Carrega o .env no os.environ (sem sobrescrever variáveis já definidas)."""
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    # Fallback mínimo (usado antes de o python-dotenv estar instalado).
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def tavily_api_key() -> str:
    return os.environ.get("TAVILY_API_KEY", "").strip()


def read_personality() -> str:
    """Conteúdo do personality.txt (injetado como System Prompt)."""
    try:
        text = PERSONALITY_FILE.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeDecodeError):
        text = ""
    return text or DEFAULT_PERSONALITY


@dataclass
class Settings:
    """Preferências da UI, lembradas entre execuções (settings.json)."""

    model: str = ""
    n_ctx: int = 8192
    temperature: float = 0.6
    max_tokens: int = 2048
    gpu_layers: str = "auto"      # "auto" ou um número
    kv_cache: str = "auto"        # "auto" | "f16" | "q8_0"
    vision_on_gpu: bool = True
    terminal_enabled: bool = True
    terminal_timeout: int = 90    # segundos por comando
    image_max_side: int = 1280    # px; imagens maiores são reduzidas
    max_agent_steps: int = 8      # ciclos pensar -> ferramenta -> observar
    prefix_cache: bool = True     # reaproveita o KV cache entre turnos

    @classmethod
    def load(cls, path: Path = SETTINGS_FILE) -> "Settings":
        s = cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return s
        for f in fields(cls):
            if f.name in data:
                try:
                    setattr(s, f.name, type(getattr(s, f.name))(data[f.name]))
                except (TypeError, ValueError):
                    pass
        return s

    def save(self, path: Path = SETTINGS_FILE) -> None:
        try:
            path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass


def load_hardware_profile() -> dict:
    try:
        return json.loads(HARDWARE_PROFILE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
