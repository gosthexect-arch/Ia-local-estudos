"""Configuracao do monitor.

Ordem de precedencia: padroes do codigo < arquivo TOML < variaveis de ambiente.
Nenhuma credencial e lida ou exigida: o escopo do projeto e somente dados
publicos de mercado.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ENV_PREFIX = "CRIPTO_MONITOR_"

DEFAULT_MODEL_PATH = "/home/ght/models/gemma4/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"


def _xdg(var: str, fallback: str) -> Path:
    raw = os.environ.get(var)
    base = Path(raw) if raw else Path.home() / fallback
    return base


def default_config_path() -> Path:
    return _xdg("XDG_CONFIG_HOME", ".config") / "cripto-monitor" / "monitor.toml"


def default_data_dir() -> Path:
    return _xdg("XDG_DATA_HOME", ".local/share") / "cripto-monitor"


@dataclass(frozen=True, slots=True)
class ExchangeConfig:
    name: str = "binance"
    rest_base: str = "https://api.binance.com"
    ws_base: str = "wss://stream.binance.com:9443"
    symbol: str = "ETHUSDT"
    primary_timeframe: str = "5m"
    timeframes: tuple[str, ...] = ("5m", "15m", "1h")

    @property
    def ws_stream_url(self) -> str:
        stream = f"{self.symbol.lower()}@kline_{self.primary_timeframe}"
        return f"{self.ws_base}/ws/{stream}"


@dataclass(frozen=True, slots=True)
class LLMConfig:
    base_url: str = "http://127.0.0.1:8080/v1"
    model_path: str = DEFAULT_MODEL_PATH
    # Latencia aceitavel entre o fechamento da vela e o alerta (spec: fase 4).
    max_latency_s: float = 60.0
    request_timeout_s: float = 90.0

    @property
    def server_root(self) -> str:
        """Raiz do llama-server (endpoints /health e /props ficam fora de /v1)."""
        return self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url


@dataclass(frozen=True, slots=True)
class PathsConfig:
    data_dir: Path = field(default_factory=default_data_dir)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "state.sqlite"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    connect_timeout_s: float = 10.0
    # Quantas mensagens ler do WebSocket durante a sondagem de diagnostico.
    probe_messages: int = 3
    probe_timeout_s: float = 30.0
    # Acima disso o relogio local e considerado fora de sincronia.
    max_clock_drift_ms: float = 1000.0


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    # Sem mensagem por este tempo, a conexao e considerada morta mesmo "aberta".
    stale_after_s: float = 90.0
    ping_interval_s: float = 20.0
    ping_timeout_s: float = 20.0
    backoff_initial_s: float = 1.0
    backoff_max_s: float = 60.0
    # Fracao de aleatoriedade somada ao backoff, para nao reconectar em manada.
    backoff_jitter: float = 0.3
    # Velas buscadas por REST ao conectar, para cobrir o periodo offline.
    backfill_limit: int = 500
    # Grava cada mensagem original em JSONL comprimido (auditoria e replay fiel).
    store_raw: bool = True


@dataclass(frozen=True, slots=True)
class AppConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    source_path: Path | None = None


def _coerce(current: Any, value: Any) -> Any:
    if isinstance(current, Path):
        return Path(str(value)).expanduser()
    if isinstance(current, tuple):
        return tuple(value)
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, int):
        return int(value)
    return type(current)(value) if current is not None else value


def _apply_section(section: Any, values: dict[str, Any], where: str) -> Any:
    known = {f for f in section.__slots__}
    updates: dict[str, Any] = {}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"chave desconhecida em {where}: {key!r}")
        updates[key] = _coerce(getattr(section, key), value)
    return replace(section, **updates)


def apply_toml(config: AppConfig, raw: dict[str, Any]) -> AppConfig:
    """Aplica um dicionario ja carregado de TOML sobre a configuracao."""
    sections = {"exchange", "llm", "paths", "network", "collector"}
    updates: dict[str, Any] = {}
    for name, values in raw.items():
        if name not in sections:
            raise ValueError(f"secao desconhecida no arquivo de configuracao: [{name}]")
        if not isinstance(values, dict):
            raise ValueError(f"secao [{name}] deve ser uma tabela TOML")
        updates[name] = _apply_section(getattr(config, name), values, f"[{name}]")
    return replace(config, **updates)


def apply_env(config: AppConfig, env: dict[str, str] | None = None) -> AppConfig:
    """Aplica variaveis CRIPTO_MONITOR_<SECAO>_<CHAVE>."""
    env = os.environ if env is None else env
    grouped: dict[str, dict[str, Any]] = {}
    for raw_key, value in env.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        remainder = raw_key[len(ENV_PREFIX) :].lower()
        section, _, key = remainder.partition("_")
        if not key:
            raise ValueError(f"variavel de ambiente invalida: {raw_key}")
        grouped.setdefault(section, {})[key] = value
    return apply_toml(config, grouped) if grouped else config


def load_config(path: Path | None = None, env: dict[str, str] | None = None) -> AppConfig:
    """Carrega a configuracao efetiva. Arquivo ausente nao e erro."""
    config = AppConfig()
    candidate = path or default_config_path()
    if candidate.is_file():
        with candidate.open("rb") as handle:
            raw = tomllib.load(handle)
        config = apply_toml(config, raw)
        config = replace(config, source_path=candidate)
    elif path is not None:
        raise FileNotFoundError(f"arquivo de configuracao nao encontrado: {candidate}")
    return apply_env(config, env)


def as_dict(config: AppConfig) -> dict[str, Any]:
    return {
        "exchange": {
            "name": config.exchange.name,
            "rest_base": config.exchange.rest_base,
            "ws_base": config.exchange.ws_base,
            "ws_stream_url": config.exchange.ws_stream_url,
            "symbol": config.exchange.symbol,
            "primary_timeframe": config.exchange.primary_timeframe,
            "timeframes": list(config.exchange.timeframes),
        },
        "llm": {
            "base_url": config.llm.base_url,
            "model_path": config.llm.model_path,
            "max_latency_s": config.llm.max_latency_s,
            "request_timeout_s": config.llm.request_timeout_s,
        },
        "paths": {
            "data_dir": str(config.paths.data_dir),
            "db_path": str(config.paths.db_path),
            "raw_dir": str(config.paths.raw_dir),
            "log_dir": str(config.paths.log_dir),
        },
        "network": {
            "connect_timeout_s": config.network.connect_timeout_s,
            "probe_messages": config.network.probe_messages,
            "probe_timeout_s": config.network.probe_timeout_s,
            "max_clock_drift_ms": config.network.max_clock_drift_ms,
        },
        "collector": {
            "stale_after_s": config.collector.stale_after_s,
            "ping_interval_s": config.collector.ping_interval_s,
            "ping_timeout_s": config.collector.ping_timeout_s,
            "backoff_initial_s": config.collector.backoff_initial_s,
            "backoff_max_s": config.collector.backoff_max_s,
            "backoff_jitter": config.collector.backoff_jitter,
            "backfill_limit": config.collector.backfill_limit,
            "store_raw": config.collector.store_raw,
        },
        "source_path": str(config.source_path) if config.source_path else None,
    }
