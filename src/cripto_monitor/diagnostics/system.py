"""Diagnostico somente leitura do ambiente local.

Nenhuma funcao deste modulo escreve, instala ou altera configuracao do sistema.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cripto_monitor.config import AppConfig
from cripto_monitor.diagnostics.base import CheckResult, Status
from cripto_monitor.term import human_bytes

MIN_PYTHON = (3, 11)
# Baselines vindos da especificacao do projeto (Xeon E5-2650 v4 / RX 580 / 15 GiB).
MIN_RAM_GIB = 6.0
MIN_FREE_DISK_GIB = 5.0


def parse_meminfo(text: str) -> dict[str, int]:
    """Converte /proc/meminfo em bytes por chave."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if not key or not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        if len(parts) > 1 and parts[1].lower() == "kb":
            value *= 1024
        out[key.strip()] = value
    return out


def parse_cpuinfo(text: str) -> dict[str, Any]:
    """Extrai modelo e contagem de processadores logicos de /proc/cpuinfo."""
    model = ""
    logical = 0
    physical_ids: set[str] = set()
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if key == "processor":
            logical += 1
        elif key == "model name" and not model:
            model = value
        elif key == "physical id":
            physical_ids.add(value)
    return {"model": model, "logical_cpus": logical, "sockets": len(physical_ids) or 1}


def _read(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def check_python() -> CheckResult:
    version = ".".join(str(p) for p in sys.version_info[:3])
    ok = sys.version_info[:2] >= MIN_PYTHON
    return CheckResult(
        name="python",
        status=Status.OK if ok else Status.FAIL,
        summary=f"Python {version} ({sys.executable})",
        details={"version": version, "executable": sys.executable, "minimo": "3.11"},
        hint=None if ok else "A Fase 1 usa tomllib, disponivel a partir do Python 3.11.",
    )


def check_host() -> CheckResult:
    uname = platform.uname()
    details = {
        "hostname": socket.gethostname(),
        "usuario": os.environ.get("USER") or os.environ.get("LOGNAME") or "?",
        "sistema": f"{uname.system} {uname.release}",
        "arquitetura": uname.machine,
        "sessao": os.environ.get("XDG_SESSION_TYPE", "desconhecida"),
        "desktop": os.environ.get("XDG_CURRENT_DESKTOP", "desconhecido"),
    }
    return CheckResult(
        name="host",
        status=Status.OK,
        summary=f"{details['hostname']} - {details['sistema']} ({details['arquitetura']})",
        details=details,
    )


def check_cpu() -> CheckResult:
    try:
        info = parse_cpuinfo(_read("/proc/cpuinfo"))
    except OSError as exc:
        return CheckResult("cpu", Status.WARN, f"/proc/cpuinfo indisponivel: {exc}")
    load1, load5, load15 = os.getloadavg()
    logical = info["logical_cpus"] or os.cpu_count() or 0
    saturado = logical and load1 > logical * 0.9
    return CheckResult(
        name="cpu",
        status=Status.WARN if saturado else Status.OK,
        summary=f"{info['model'] or 'CPU desconhecida'} - {logical} threads, load1 {load1:.2f}",
        details={
            "modelo": info["model"],
            "threads": logical,
            "sockets": info["sockets"],
            "loadavg": [round(load1, 2), round(load5, 2), round(load15, 2)],
        },
        hint="Carga alta antes de iniciar o coletor pode mascarar latencia." if saturado else None,
    )


def check_memory() -> CheckResult:
    try:
        mem = parse_meminfo(_read("/proc/meminfo"))
    except OSError as exc:
        return CheckResult("memoria", Status.WARN, f"/proc/meminfo indisponivel: {exc}")
    total = mem.get("MemTotal", 0)
    available = mem.get("MemAvailable", mem.get("MemFree", 0))
    swap_total = mem.get("SwapTotal", 0)
    swap_free = mem.get("SwapFree", 0)
    swap_usado = swap_total - swap_free
    gib = available / (1024**3)
    status = Status.OK
    hint = None
    if gib < MIN_RAM_GIB:
        status = Status.WARN
        hint = (
            f"Menos de {MIN_RAM_GIB:.0f} GiB livres. O llama-server com contexto de 65k "
            "pode empurrar o sistema para o swap."
        )
    if swap_total and swap_usado > swap_total * 0.25:
        status = Status.WARN
        hint = "Swap ja em uso relevante; medir antes de subir o modelo."
    return CheckResult(
        name="memoria",
        status=status,
        summary=(
            f"RAM {human_bytes(available)} livres de {human_bytes(total)} | "
            f"swap {human_bytes(swap_usado)} usados de {human_bytes(swap_total)}"
        ),
        details={
            "ram_total_bytes": total,
            "ram_disponivel_bytes": available,
            "swap_total_bytes": swap_total,
            "swap_usado_bytes": swap_usado,
        },
        hint=hint,
    )


def check_gpu() -> CheckResult:
    """Le /sys para identificar a GPU e o driver. Nao executa carga na placa."""
    cards: list[dict[str, str]] = []
    for device in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        entry: dict[str, str] = {"card": device.parent.name}
        for field_name in ("vendor", "device"):
            try:
                entry[field_name] = _read(device / field_name).strip()
            except OSError:
                continue
        driver = device / "driver"
        if driver.is_symlink():
            entry["driver"] = os.path.basename(os.readlink(driver))
        cards.append(entry)

    ferramentas = {
        nome: shutil.which(nome)
        for nome in ("vulkaninfo", "radeontop", "rocm-smi", "llama-server")
    }
    tem_amdgpu = any(c.get("driver") == "amdgpu" for c in cards)
    if not cards:
        return CheckResult(
            name="gpu",
            status=Status.WARN,
            summary="nenhuma GPU encontrada em /sys/class/drm",
            details={"ferramentas": ferramentas},
            hint="Sem GPU a inferencia cai para CPU e a latencia por vela sobe muito.",
        )
    resumo = ", ".join(f"{c['card']}:{c.get('driver', '?')}" for c in cards)
    return CheckResult(
        name="gpu",
        status=Status.OK if tem_amdgpu else Status.WARN,
        summary=f"{len(cards)} dispositivo(s) DRM ({resumo})",
        details={"dispositivos": cards, "ferramentas": ferramentas},
        hint=None if tem_amdgpu else "Driver amdgpu esperado para a RX 580 nao foi encontrado.",
    )


def check_clock() -> CheckResult:
    """Confere que o processo consegue trabalhar em UTC (requisito da spec)."""
    agora = datetime.now(timezone.utc)
    local = datetime.now().astimezone()
    offset = local.utcoffset()
    return CheckResult(
        name="relogio",
        status=Status.OK,
        summary=f"UTC {agora.isoformat(timespec='seconds')} (fuso local {local.tzname()})",
        details={
            "utc": agora.isoformat(timespec="milliseconds"),
            "offset_local_s": int(offset.total_seconds()) if offset else 0,
            "tz_env": os.environ.get("TZ", "nao definido"),
        },
        hint="A deriva real contra a exchange e medida no bloco de dados de mercado.",
    )


def check_paths(config: AppConfig) -> CheckResult:
    """Verifica se os diretorios de dados sao gravaveis. Nao cria nada."""
    data_dir = config.paths.data_dir
    existente = data_dir
    while not existente.exists() and existente != existente.parent:
        existente = existente.parent
    gravavel = os.access(existente, os.W_OK)
    try:
        uso = shutil.disk_usage(existente)
        livre_gib = uso.free / (1024**3)
    except OSError:
        livre_gib = 0.0
        uso = None
    status = Status.OK
    hint = None
    if not gravavel:
        status = Status.FAIL
        hint = f"Sem permissao de escrita em {existente}."
    elif livre_gib < MIN_FREE_DISK_GIB:
        status = Status.WARN
        hint = "Pouco espaco livre para candles brutos e Parquet."
    if not data_dir.exists() and hint is None:
        hint = f"Diretorio {data_dir} sera criado na Fase 2."
    return CheckResult(
        name="armazenamento",
        status=status,
        summary=(
            f"{data_dir} ({'existe' if data_dir.exists() else 'a criar'}), "
            f"{human_bytes(uso.free) if uso else '?'} livres"
        ),
        details={
            "data_dir": str(data_dir),
            "existe": data_dir.exists(),
            "ponto_verificado": str(existente),
            "gravavel": gravavel,
            "livre_bytes": uso.free if uso else None,
        },
        hint=hint,
    )


def build_checks(config: AppConfig) -> list[tuple[str, object]]:
    return [
        ("python", check_python),
        ("host", check_host),
        ("cpu", check_cpu),
        ("memoria", check_memory),
        ("gpu", check_gpu),
        ("relogio", check_clock),
        ("armazenamento", lambda: check_paths(config)),
    ]
