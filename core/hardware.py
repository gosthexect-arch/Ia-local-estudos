"""Scanner de hardware (CPU, RAM, GPUs) e escolha do backend do llama.cpp.

Somente biblioteca padrão: o start.bat executa este módulo ANTES de instalar
as dependências, para decidir qual build do llama-cpp-python instalar.

Uso (linha de comando):
    python -m core.hardware                  # escaneia e grava hardware_profile.json
    python -m core.hardware --backend-order  # ordem de backends p/ tentar (ex.: "cu125 cuda vulkan cpu")
    python -m core.hardware --cuda-runtime cu125   # pacotes pip do runtime CUDA, se faltarem DLLs
    python -m core.hardware --install-plugin vulkan  # baixa o plugin oficial de GPU do llama.cpp
    python -m core.hardware --verify-llama vulkan [--quick]  # testa se o llama.cpp usa a GPU de verdade

Backends: "cu125"/"cu124"/"cu118" = wheels CUDA do llama-cpp-python; "cuda"/"vulkan" =
wheel CPU + plugin oficial do llama.cpp (ver core/gpu_plugins.py); "cpu".
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from core import gpu_plugins
from core.config import HARDWARE_PROFILE_FILE, LOGS_DIR, MODELS_DIR, ROOT_DIR, load_env

IS_WINDOWS = sys.platform == "win32"
_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0  # CREATE_NO_WINDOW

# Runtime CUDA (cudart + cuBLAS) compatível com cada wheel CUDA do llama-cpp-python.
CUDA_RUNTIME_PKGS = {
    "cu125": ["nvidia-cuda-runtime-cu12==12.5.82", "nvidia-cublas-cu12==12.5.3.2"],
    "cu124": ["nvidia-cuda-runtime-cu12==12.4.127", "nvidia-cublas-cu12==12.4.5.8"],
    "cu118": ["nvidia-cuda-runtime-cu11==11.8.89", "nvidia-cublas-cu11==11.11.3.6"],
}
_CUDART_DLL = {"cu125": "cudart64_12.dll", "cu124": "cudart64_12.dll", "cu118": "cudart64_110.dll"}


def _run(cmd: list[str], timeout: float = 15) -> str | None:
    try:
        out = subprocess.run(
            cmd, capture_output=True, timeout=timeout, creationflags=_NO_WINDOW,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.decode("utf-8", errors="replace")


# ----------------------------------------------------------------------------
# CPU
# ----------------------------------------------------------------------------
def _win_cpu_topology() -> dict[str, int] | None:
    """Núcleos físicos/lógicos e núcleos P/E (CPUs híbridas) via Win32 API."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    relation_core = 0
    length = wintypes.DWORD(0)
    kernel32.GetLogicalProcessorInformationEx(relation_core, None, ctypes.byref(length))
    if length.value == 0:
        return None
    buf = (ctypes.c_byte * length.value)()
    if not kernel32.GetLogicalProcessorInformationEx(relation_core, buf, ctypes.byref(length)):
        return None
    raw = bytes(buf)
    cores: list[tuple[int, int]] = []  # (classe de eficiência, threads)
    off = 0
    while off + 8 <= len(raw):
        relationship, size = struct.unpack_from("<II", raw, off)
        if size == 0:
            break
        if relationship == relation_core:
            eff_class = raw[off + 9]
            group_count = struct.unpack_from("<H", raw, off + 8 + 22)[0]
            threads = 0
            for g in range(max(group_count, 1)):
                mask = struct.unpack_from("<Q", raw, off + 32 + 16 * g)[0]
                threads += bin(mask).count("1")
            cores.append((eff_class, threads))
        off += size
    if not cores:
        return None
    classes = {c for c, _ in cores}
    p_cores = sum(1 for c, _ in cores if c == max(classes)) if len(classes) > 1 else len(cores)
    return {
        "physical_cores": len(cores),
        "logical_cores": sum(t for _, t in cores),
        "performance_cores": p_cores,
        "efficiency_cores": len(cores) - p_cores,
    }


def _linux_cpu_topology() -> dict[str, int] | None:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return None
    pairs, phys_id = set(), "0"
    for line in text.splitlines():
        if line.startswith("physical id"):
            phys_id = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            pairs.add((phys_id, line.split(":", 1)[1].strip()))
    logical = os.cpu_count() or 1
    physical = len(pairs) or max(1, logical // 2)
    p_cores = physical
    core_cpus = Path("/sys/devices/cpu_core/cpus")  # CPUs Intel híbridas
    if core_cpus.exists():
        ids = _expand_cpu_list(core_cpus.read_text().strip())
        p_ids = set()
        for i in ids:
            try:
                p_ids.add(Path(f"/sys/devices/system/cpu/cpu{i}/topology/core_id").read_text().strip())
            except OSError:
                pass
        p_cores = len(p_ids) or physical
    return {
        "physical_cores": physical,
        "logical_cores": logical,
        "performance_cores": p_cores,
        "efficiency_cores": max(0, physical - p_cores),
    }


def _expand_cpu_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def _cpu_name() -> str:
    if IS_WINDOWS:
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name.strip()
    return platform.processor() or "CPU desconhecida"


def detect_cpu() -> dict[str, Any]:
    topo = None
    try:
        if IS_WINDOWS:
            topo = _win_cpu_topology()
        elif sys.platform.startswith("linux"):
            topo = _linux_cpu_topology()
        elif sys.platform == "darwin":
            phys = int((_run(["sysctl", "-n", "hw.physicalcpu"]) or "0").strip() or 0)
            perf = int((_run(["sysctl", "-n", "hw.perflevel0.physicalcpu"]) or "0").strip() or 0)
            if phys:
                topo = {"physical_cores": phys, "logical_cores": os.cpu_count() or phys,
                        "performance_cores": perf or phys, "efficiency_cores": max(0, phys - (perf or phys))}
    except Exception:  # noqa: BLE001 - detecção nunca deve derrubar o start
        topo = None
    if not topo:
        logical = os.cpu_count() or 1
        topo = {"physical_cores": max(1, logical // 2), "logical_cores": logical,
                "performance_cores": max(1, logical // 2), "efficiency_cores": 0}
    return {"name": _cpu_name(), **topo}


def recommend_threads(cpu: dict[str, Any]) -> dict[str, int]:
    """Threads ideais para o llama.cpp.

    A geração de tokens é limitada pela banda de memória: usar mais threads que
    núcleos físicos (hyper-threading) ou incluir núcleos E de CPUs híbridas
    costuma DEIXAR MAIS LENTO. Por isso: nº de núcleos físicos de performance.
    """
    p = int(cpu.get("performance_cores") or cpu.get("physical_cores") or 1)
    return {"n_threads": max(1, p), "n_threads_batch": max(1, p)}


# ----------------------------------------------------------------------------
# RAM
# ----------------------------------------------------------------------------
def detect_ram() -> dict[str, int]:
    total = avail = 0
    try:
        if IS_WINDOWS:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                total, avail = st.ullTotalPhys, st.ullAvailPhys
        elif sys.platform.startswith("linux"):
            info = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, v = line.split(":", 1)
                info[k] = int(v.strip().split()[0]) * 1024
            total, avail = info.get("MemTotal", 0), info.get("MemAvailable", 0)
        elif sys.platform == "darwin":
            total = int((_run(["sysctl", "-n", "hw.memsize"]) or "0").strip() or 0)
            avail = total // 2
    except Exception:  # noqa: BLE001
        pass
    return {"total_mb": int(total / 2**20), "available_mb": int(avail / 2**20)}


# ----------------------------------------------------------------------------
# GPU
# ----------------------------------------------------------------------------
def _find_nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if IS_WINDOWS:
        for p in (
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
        ):
            if p.exists():
                return str(p)
    return None


def _guess_compute_capability(name: str) -> float | None:
    n = name.upper()
    table = [
        (r"RTX\s*PRO\s*\d{4}\s*BLACKWELL|RTX\s*50\d{2}", 12.0), (r"RTX\s*40\d{2}|\bL4\b|\bL40", 8.9),
        (r"\bH100|\bH200|\bGH200", 9.0), (r"RTX\s*30\d{2}|\bA[2-6]000\b|\bA10G?\b|\bA40\b", 8.6),
        (r"\bA100\b|\bA30\b", 8.0), (r"RTX\s*20\d{2}|GTX\s*16\d{2}|TITAN\s*RTX|QUADRO\s*RTX|\bT4\b|MX[45]50", 7.5),
        (r"TITAN\s*V|\bV100\b", 7.0), (r"GTX\s*10\d{2}|TITAN\s*XP?\b|\bP[1-6]000\b|\bP40\b|\bP100\b|MX[1-3]\d0", 6.1),
        (r"GTX\s*9\d{2}|GTX\s*TITAN\s*X|\bM[2-6]000\b|GTX\s*750", 5.2),
    ]
    for pattern, cc in table:
        if re.search(pattern, n):
            return cc
    return None


def _nvidia_gpus() -> tuple[list[dict[str, Any]], float | None]:
    smi = _find_nvidia_smi()
    if not smi:
        return [], None
    fields = "index,name,memory.total,memory.used,memory.free,driver_version,compute_cap"
    out = _run([smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    has_cc = out is not None
    if out is None:
        out = _run([smi, "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
                    "--format=csv,noheader,nounits"])
    gpus: list[dict[str, Any]] = []
    for line in (out or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            cc = float(parts[6]) if has_cc and len(parts) > 6 and parts[6] not in ("", "[N/A]") else None
        except ValueError:
            cc = None
        gpus.append({
            "vendor": "nvidia",
            "index": int(parts[0]) if parts[0].isdigit() else len(gpus),
            "name": parts[1],
            "vram_total_mb": _to_int(parts[2]),
            "vram_used_mb": _to_int(parts[3]),
            "vram_free_mb": _to_int(parts[4]),
            "driver": parts[5],
            "compute_capability": cc if cc is not None else _guess_compute_capability(parts[1]),
            "integrated": False,
        })
    cuda_version = None
    header = _run([smi]) or ""
    m = re.search(r"CUDA Version:\s*([\d.]+)", header)
    if m:
        try:
            cuda_version = float(".".join(m.group(1).split(".")[:2]))
        except ValueError:
            pass
    return gpus, cuda_version


def _to_int(s: str) -> int:
    try:
        return int(float(s))
    except ValueError:
        return 0


_VENDOR_IDS = {"10de": "nvidia", "1002": "amd", "1022": "amd", "8086": "intel", "5143": "qualcomm"}
_SKIP_ADAPTERS = re.compile(
    r"basic display|basic render|remote|virtual|parsec|citrix|hyper-v|displaylink|spacedesk|idd|"
    r"meta virtual|oray|sunshine|vmware|virtualbox", re.IGNORECASE)


def _windows_display_adapters() -> list[dict[str, Any]]:
    """Adaptadores de vídeo via registro (VRAM real em QWORD, sem limite de 4 GB do WMI)."""
    import winreg

    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    adapters: list[dict[str, Any]] = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    except OSError:
        return adapters
    with root:
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(root, sub) as k:
                    def val(name: str) -> Any:
                        try:
                            return winreg.QueryValueEx(k, name)[0]
                        except OSError:
                            return None

                    name = str(val("DriverDesc") or "").strip()
                    if not name or _SKIP_ADAPTERS.search(name):
                        continue
                    mem = val("HardwareInformation.qwMemorySize")
                    if mem is None:
                        mem = val("HardwareInformation.MemorySize")
                        if isinstance(mem, (bytes, bytearray)):
                            mem = int.from_bytes(bytes(mem)[:8], "little")
                    dev_id = str(val("MatchingDeviceId") or "").lower()
                    m = re.search(r"ven_([0-9a-f]{4})", dev_id)
                    vendor = _VENDOR_IDS.get(m.group(1) if m else "", "")
                    if not vendor:
                        low = name.lower()
                        vendor = ("nvidia" if "nvidia" in low else "amd" if ("amd" in low or "radeon" in low)
                                  else "intel" if "intel" in low else "other")
                    vram_mb = int((mem or 0) / 2**20)
                    adapters.append({
                        "vendor": vendor, "name": name, "vram_total_mb": vram_mb, "vram_free_mb": None,
                        "driver": str(val("DriverVersion") or ""), "compute_capability": None,
                        "integrated": _looks_integrated(vendor, name, vram_mb),
                    })
            except OSError:
                continue
    return adapters


def _looks_integrated(vendor: str, name: str, vram_mb: int) -> bool:
    if vendor == "intel":
        # Intel Arc dedicadas têm número de modelo (A770, B580...). O resto é integrado.
        return not re.search(r"\bArc\b.*\b[AB]\d{3}\b", name, re.IGNORECASE)
    if vendor == "amd":
        if re.search(r"Radeon\(TM\) Graphics|Radeon Graphics|Vega \d+\b|\b\d{3}M\b", name, re.IGNORECASE):
            return True
        return 0 < vram_mb < 2048
    return False


def _linux_other_gpus() -> list[dict[str, Any]]:
    gpus = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]")):
        dev = card / "device"
        try:
            vendor_id = (dev / "vendor").read_text().strip().lower().replace("0x", "")
        except OSError:
            continue
        vendor = _VENDOR_IDS.get(vendor_id)
        if vendor in (None, "nvidia"):
            continue
        total = 0
        try:
            total = int((dev / "mem_info_vram_total").read_text()) // 2**20
        except (OSError, ValueError):
            pass
        gpus.append({"vendor": vendor, "name": f"{vendor.upper()} GPU ({card.name})", "vram_total_mb": total,
                     "vram_free_mb": None, "driver": "", "compute_capability": None,
                     "integrated": total < 2048})
    return gpus


def detect_gpus() -> tuple[list[dict[str, Any]], float | None]:
    try:
        nvidia, cuda_version = _nvidia_gpus()
    except Exception:  # noqa: BLE001
        nvidia, cuda_version = [], None
    others: list[dict[str, Any]] = []
    try:
        if IS_WINDOWS:
            others = _windows_display_adapters()
        elif sys.platform.startswith("linux"):
            others = _linux_other_gpus()
        elif sys.platform == "darwin" and platform.machine() == "arm64":
            ram = detect_ram()["total_mb"]
            others = [{"vendor": "apple", "name": "Apple Silicon (Metal)", "vram_total_mb": int(ram * 0.7),
                       "vram_free_mb": None, "driver": "", "compute_capability": None, "integrated": False}]
    except Exception:  # noqa: BLE001
        others = []
    if nvidia:  # nvidia-smi é a fonte confiável para placas NVIDIA
        others = [g for g in others if g["vendor"] != "nvidia"]
    return nvidia + others, cuda_version


def backend_candidates(gpus: list[dict[str, Any]], cuda_version: float | None) -> list[str]:
    """Ordem de builds do llama-cpp-python a tentar (a primeira que funcionar vence)."""
    forced = os.environ.get("IA_BACKEND", "").strip().lower()
    if forced:
        order = [b for b in re.split(r"[\s,;]+", forced) if b]
        return order + [b for b in ("cpu",) if b not in order]
    if sys.platform == "darwin":
        return ["metal", "cpu"] if platform.machine() == "arm64" else ["cpu"]
    nvidia = [g for g in gpus if g["vendor"] == "nvidia"]
    discrete = [g for g in gpus if g["vendor"] in ("amd", "intel", "other") and not g.get("integrated")]
    order: list[str] = []
    if nvidia:
        ccs = [g["compute_capability"] for g in nvidia if g.get("compute_capability")]
        cc = min(ccs) if ccs else None
        driver = cuda_version if cuda_version is not None else 99.0  # desconhecido: deixa o teste decidir
        if cc is None or cc >= 6.0:  # wheels CUDA 12 do llama-cpp-python cobrem Pascal (6.x) em diante
            if driver >= 12.5:
                order.append("cu125")
            if driver >= 12.4:
                order.append("cu124")
        if (cc is None or cc >= 5.0) and driver >= 12.4:
            order.append("cuda")  # plugin oficial CUDA 12.4 do llama.cpp
        if not order and 11.8 <= driver < 12.4 and (cc is None or 6.0 <= cc < 9.0):
            order.append("cu118")
        order.append("vulkan")  # driver antigo ou GPU antiga: Vulkan ainda acelera
    elif discrete:
        order.append("vulkan")  # AMD / Intel Arc: plugin Vulkan oficial do llama.cpp
    order.append("cpu")
    return order


def backend_kind(backend: str) -> str:
    return "cuda" if backend.startswith("cu") else backend


def wheel_index(backend: str) -> str:
    """Índice de wheels do llama-cpp-python usado por cada backend (plugins usam a wheel CPU)."""
    return "cpu" if gpu_plugins.is_plugin_backend(backend) else backend


# ----------------------------------------------------------------------------
# DLLs do runtime CUDA (Windows)
# ----------------------------------------------------------------------------
def setup_gpu_dll_paths() -> list[str]:
    """Expõe as DLLs do runtime CUDA instaladas via pip (nvidia-*-cu12) ao llama.dll.

    Deve ser chamado ANTES de `import llama_cpp`. Sem efeito fora do Windows.
    """
    if not IS_WINDOWS:
        return []
    added: list[str] = []
    for sp in _site_packages_dirs():
        nv = Path(sp) / "nvidia"
        if not nv.is_dir():
            continue
        for bin_dir in nv.glob("*/bin"):
            if any(bin_dir.glob("*.dll")):
                d = str(bin_dir)
                try:
                    os.add_dll_directory(d)
                except (OSError, AttributeError):
                    pass
                if d not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                added.append(d)
    return added


def _site_packages_dirs() -> list[str]:
    import site

    dirs = []
    try:
        dirs.extend(site.getsitepackages())
    except AttributeError:
        pass
    try:
        dirs.append(site.getusersitepackages())
    except AttributeError:
        pass
    dirs.extend(p for p in sys.path if p.endswith("site-packages"))
    seen, out = set(), []
    for d in dirs:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def cuda_runtime_missing(backend: str) -> bool:
    dll = _CUDART_DLL.get(backend)
    if not dll:
        return False
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path and (Path(cuda_path) / "bin" / dll).exists():
        return False
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if d and (Path(d) / dll).exists():
            return False
    for sp in _site_packages_dirs():
        if list((Path(sp) / "nvidia").glob(f"*/bin/{dll}")):
            return False
    return True


# ----------------------------------------------------------------------------
# Verificação do llama-cpp-python instalado
# ----------------------------------------------------------------------------
def llama_devices() -> list[dict[str, Any]]:
    """Dispositivos que o ggml (llama.cpp) realmente enxerga, com memória livre/total."""
    import ctypes

    from llama_cpp import _ggml  # type: ignore

    lib = _ggml.libggml
    lib.ggml_backend_dev_count.restype = ctypes.c_size_t
    lib.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
    lib.ggml_backend_dev_get.restype = ctypes.c_void_p
    lib.ggml_backend_dev_name.argtypes = [ctypes.c_void_p]
    lib.ggml_backend_dev_name.restype = ctypes.c_char_p
    lib.ggml_backend_dev_description.argtypes = [ctypes.c_void_p]
    lib.ggml_backend_dev_description.restype = ctypes.c_char_p
    lib.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
    lib.ggml_backend_dev_type.restype = ctypes.c_int
    lib.ggml_backend_dev_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                            ctypes.POINTER(ctypes.c_size_t)]
    lib.ggml_backend_dev_memory.restype = None
    kinds = {0: "cpu", 1: "gpu", 2: "igpu", 3: "accel", 4: "meta"}
    devices = []
    for i in range(lib.ggml_backend_dev_count()):
        dev = lib.ggml_backend_dev_get(i)
        free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
        lib.ggml_backend_dev_memory(dev, ctypes.byref(free), ctypes.byref(total))
        devices.append({
            "name": (lib.ggml_backend_dev_name(dev) or b"").decode(errors="replace"),
            "description": (lib.ggml_backend_dev_description(dev) or b"").decode(errors="replace"),
            "type": kinds.get(lib.ggml_backend_dev_type(dev), "?"),
            "free_mb": int(free.value / 2**20),
            "total_mb": int(total.value / 2**20),
        })
    return devices


SMOKE_MODEL = ROOT_DIR / "tests" / "fixtures" / "Tiny-Qwen2-VL-F16.gguf"


def verify_llama(backend: str, smoke: bool = True) -> tuple[bool, str]:
    """Confirma que a instalação funciona: import, plugin, GPU visível e inferência real na GPU."""
    setup_gpu_dll_paths()
    try:
        import llama_cpp  # type: ignore
    except Exception as e:  # noqa: BLE001 - DLL ausente, wheel incompatível etc.
        return False, f"falha ao importar llama_cpp: {e}"
    version = getattr(llama_cpp, "__version__", "?")
    if version != gpu_plugins.LLAMA_CPP_PYTHON_VERSION:
        return False, f"llama_cpp {version} instalado; o app requer {gpu_plugins.LLAMA_CPP_PYTHON_VERSION}"
    if gpu_plugins.is_plugin_backend(backend):
        try:
            loaded = gpu_plugins.load(backend)
        except Exception as e:  # noqa: BLE001
            return False, f"falha ao carregar o plugin {backend}: {e}"
        if not loaded:
            return False, (f"o plugin {backend} não carregou (arquivo ausente, driver sem suporte ou "
                           f"DLL bloqueada) — pasta {gpu_plugins.plugin_dir(backend)}")
    if backend_kind(backend) == "cpu":
        return True, f"llama_cpp {version} (CPU)"
    try:
        gpus = [d for d in llama_devices() if d["type"] in ("gpu", "igpu")]
    except Exception as e:  # noqa: BLE001
        return False, f"llama_cpp {version}: não foi possível listar dispositivos ({e})"
    if not gpus:
        return False, f"llama_cpp {version}: backend '{backend}' não encontrou nenhuma GPU utilizável"
    desc = ", ".join(f"{d['description'] or d['name']} ({d['total_mb']} MB)" for d in gpus)
    if smoke and SMOKE_MODEL.exists():
        print("      testando inferência na GPU (a 1ª execução pode compilar kernels por alguns minutos)...",
              flush=True)
        try:
            llm = llama_cpp.Llama(model_path=str(SMOKE_MODEL), n_gpu_layers=-1, n_ctx=256, verbose=False)
            llm.create_completion("Olá", max_tokens=4, temperature=0.0)
            llm.close()
        except Exception as e:  # noqa: BLE001
            return False, f"GPU encontrada ({desc}), mas a inferência de teste falhou: {e}"
    return True, f"llama_cpp {version} [{backend}] -> {desc}"


def msvc_runtime_version() -> tuple[int, int] | None:
    """Versão (major, minor) do msvcp140.dll do sistema; None se não der para ler."""
    if not IS_WINDOWS:
        return None
    import ctypes
    from ctypes import wintypes

    path = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "msvcp140.dll")
    if not os.path.exists(path):
        return (0, 0)
    try:
        ver = ctypes.WinDLL("version")
        size = ver.GetFileVersionInfoSizeW(path, None)
        if not size:
            return None
        buf = ctypes.create_string_buffer(size)
        if not ver.GetFileVersionInfoW(path, 0, size, buf):
            return None
        ptr, length = ctypes.c_void_p(), wintypes.UINT()
        if not ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)):
            return None
        info = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32 * 13)).contents  # VS_FIXEDFILEINFO
        return info[2] >> 16, info[2] & 0xFFFF
    except (OSError, AttributeError, ValueError):
        return None


def needs_vcredist(minimum: tuple[int, int] = (14, 50)) -> bool:
    version = msvc_runtime_version()
    return version is not None and version < minimum


def log_install(message: str) -> None:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOGS_DIR / "instalacao.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except OSError:
        pass


# ----------------------------------------------------------------------------
# Perfil completo
# ----------------------------------------------------------------------------
def os_description() -> str:
    if IS_WINDOWS:
        build = sys.getwindowsversion().build  # type: ignore[attr-defined]
        return f"Windows {'11' if build >= 22000 else '10'} (build {build})"
    return f"{platform.system()} {platform.release()}"


def installed_backend() -> str:
    marker = Path(sys.prefix) / "llama_backend.txt"
    try:
        return marker.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def scan(include_models: bool = True) -> dict[str, Any]:
    cpu = detect_cpu()
    gpus, cuda_version = detect_gpus()
    profile: dict[str, Any] = {
        "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "os": os_description(),
        "python": platform.python_version(),
        "cpu": cpu,
        "ram": detect_ram(),
        "gpus": gpus,
        "cuda_driver_version": cuda_version,
        "backend_candidates": backend_candidates(gpus, cuda_version),
        "installed_backend": installed_backend(),
        "recommended": recommend_threads(cpu),
    }
    if include_models:
        profile["models"] = _model_plans(profile)
    return profile


def _model_plans(profile: dict[str, Any], n_ctx: int = 8192) -> dict[str, Any]:
    """Pré-calcula n_gpu_layers de cada GGUF em models/ (contexto padrão)."""
    from core.gguf_info import is_secondary_split, try_read_gguf
    from core.models import find_mmproj
    from core.offload import plan_offload

    plans: dict[str, Any] = {}
    if not MODELS_DIR.is_dir():
        return plans
    for path in sorted(MODELS_DIR.rglob("*.gguf")):
        if "mmproj" in path.name.lower() or is_secondary_split(path):
            continue
        info = try_read_gguf(path)
        if info is None or info.is_mmproj:
            continue
        mmproj = find_mmproj(path)
        mm_info = try_read_gguf(mmproj) if mmproj else None
        plan = plan_offload(info, mm_info, n_ctx=n_ctx, profile=profile)
        plans[str(path.relative_to(MODELS_DIR))] = {
            **info.summary(), "mmproj": mmproj.name if mmproj else None, "n_ctx": n_ctx, **plan.as_dict(),
        }
    return plans


def write_profile(profile: dict[str, Any]) -> None:
    HARDWARE_PROFILE_FILE.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(p: dict[str, Any]) -> None:
    cpu, ram = p["cpu"], p["ram"]
    hybrid = f" | P-cores: {cpu['performance_cores']} E-cores: {cpu['efficiency_cores']}" if cpu.get("efficiency_cores") else ""
    print(f"  Sistema : {p['os']} | Python {p['python']}")
    print(f"  CPU     : {cpu['name']} | {cpu['physical_cores']} nucleos / {cpu['logical_cores']} threads{hybrid}")
    print(f"  RAM     : {ram['total_mb'] / 1024:.1f} GB (livre {ram['available_mb'] / 1024:.1f} GB)")
    if not p["gpus"]:
        print("  GPU     : nenhuma GPU dedicada detectada (modo CPU)")
    for g in p["gpus"]:
        extra = f" | CC {g['compute_capability']}" if g.get("compute_capability") else ""
        free = f" (livre {g['vram_free_mb']} MB)" if g.get("vram_free_mb") is not None else ""
        kind = " [integrada]" if g.get("integrated") else ""
        print(f"  GPU     : {g['name']}{kind} | VRAM {g['vram_total_mb']} MB{free}{extra}")
    if p.get("cuda_driver_version"):
        print(f"  Driver CUDA suporta ate: {p['cuda_driver_version']}")
    r = p["recommended"]
    print(f"  Threads ideais: n_threads={r['n_threads']} n_threads_batch={r['n_threads_batch']}")
    print(f"  Backends (ordem de tentativa): {' -> '.join(p['backend_candidates'])}")
    for name, m in (p.get("models") or {}).items():
        state = "100% na GPU" if m.get("full_offload") else f"{m.get('gpu_layers')}/{m.get('n_layers')} camadas na GPU"
        print(f"  Modelo  : {name} -> {state} @ ctx {m.get('n_ctx')} (KV {m.get('kv_type')})")


def _arg_after(argv: list[str], flag: str, default: str) -> str:
    i = argv.index(flag)
    return argv[i + 1] if len(argv) > i + 1 and not argv[i + 1].startswith("--") else default


def main(argv: list[str]) -> int:
    load_env()  # permite IA_BACKEND no .env
    if "--backend-order" in argv:
        gpus, cuda = detect_gpus()
        print(" ".join(backend_candidates(gpus, cuda)))
        return 0
    if "--cuda-runtime" in argv:
        bk = argv[argv.index("--cuda-runtime") + 1] if len(argv) > argv.index("--cuda-runtime") + 1 else ""
        if cuda_runtime_missing(bk):
            print(" ".join(CUDA_RUNTIME_PKGS.get(bk, [])))
        return 0
    if "--needs-vcredist" in argv:
        return 0 if needs_vcredist() else 1
    if "--wheel-index" in argv:
        print(wheel_index(_arg_after(argv, "--wheel-index", "cpu")))
        return 0
    if "--install-plugin" in argv:
        bk = _arg_after(argv, "--install-plugin", "")
        ok, msg = gpu_plugins.install(bk, log=print)
        log_install(f"[install-plugin {bk}] {'OK' if ok else 'FALHOU'}: {msg}")
        print(("      [OK] " if ok else "      [FALHOU] ") + msg)
        return 0 if ok else 1
    if "--verify-llama" in argv:
        bk = _arg_after(argv, "--verify-llama", "cpu")
        ok, msg = verify_llama(bk, smoke="--quick" not in argv)
        log_install(f"[verify {bk}] {'OK' if ok else 'FALHOU'}: {msg}")
        print(("      [OK] " if ok else "      [FALHOU] ") + msg)
        return 0 if ok else 1
    profile = scan(include_models=True)
    write_profile(profile)
    if "--quiet" not in argv:
        print_summary(profile)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
