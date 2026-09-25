"""Aceleração por GPU com os plugins OFICIAIS do llama.cpp (ggml-vulkan / ggml-cuda).

Wheels do llama-cpp-python com GPU nem sempre existem para Windows (a build
Vulkan, por exemplo, pode não estar publicada). Já o llama.cpp oficial publica
em toda release o backend de GPU como um plugin carregável em tempo de execução
(GGML_BACKEND_DL). O plugin só depende da pasta ggml/ do llama.cpp; as releases em
COMPATIBLE_TAGS têm a pasta ggml/ byte a byte idêntica à do llama-cpp-python
fixado (mesma árvore git), então a ABI é a mesma: instala-se a wheel CPU e o
plugin é registrado com ggml_backend_load() antes de abrir o modelo.

Somente biblioteca padrão (usado pelo start.bat antes do pip e pelo app).
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

IS_WINDOWS = sys.platform == "win32"

LLAMA_CPP_PYTHON_VERSION = "0.3.35"  # vendoriza ggml-org/llama.cpp @ 4df29be4f (tag b10454)
# Releases com ggml/ idêntica ao 4df29be4f (árvore git 009135aa3448), da mais próxima para a
# mais antiga. A própria b10454 é um commit "[no ci]" e não tem binários publicados.
COMPATIBLE_TAGS = ["b10453", "b10452", "b10451", "b10450", "b10448", "b10447", "b10446", "b10444", "b10443",
                   "b10442", "b10454"]
RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{name}"

PLUGINS: dict[str, dict[str, list[str]]] = {
    "vulkan": {  # AMD, Intel e NVIDIA (qualquer GPU com driver Vulkan 1.2+)
        "archives": ["llama-{tag}-bin-win-vulkan-x64.zip"],
        "files": ["ggml-vulkan.dll"],
        "load": ["ggml-vulkan.dll"],
        "preload": [],
    },
    "cuda": {  # NVIDIA com driver CUDA 12.4+ (runtime cudart/cuBLAS incluído)
        "archives": ["llama-{tag}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"],
        "files": ["ggml-cuda.dll", "cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll"],
        "load": ["ggml-cuda.dll"],
        "preload": ["cudart64_*.dll", "cublasLt64_*.dll", "cublas64_*.dll"],
    },
}

_LOADED: list[str] | None = None
LAST_ERROR = ""          # motivo da última falha ao carregar um plugin (mostrado na UI)
_HANDLES: list[Any] = []  # mantém as DLLs carregadas durante todo o processo


def is_plugin_backend(backend: str) -> bool:
    return backend in PLUGINS


def plugin_dir(backend: str, root: Path | None = None) -> Path:
    return (root or Path(sys.prefix)) / "llama_gpu" / backend


class NotPublished(OSError):
    """A release não tem o arquivo (HTTP 404): tenta a próxima release compatível."""


# --------------------------------------------------------------------------- instalação
def _download(url: str, dest: Path, opener: Callable[..., Any], log: Callable[[str], None]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ia-local-setup"})
            with opener(req, timeout=60) as resp, open(dest, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done, shown = 0, -1
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    pct = int(done * 100 / total) if total else -1
                    if pct >= 0 and pct // 5 != shown:
                        shown = pct // 5
                        print(f"\r      baixando {dest.name}: {pct:3d}% ({done >> 20}/{total >> 20} MB)", end="",
                              flush=True)
                print(flush=True)
                if total and done != total:
                    raise OSError(f"download incompleto ({done} de {total} bytes)")
            return
        except Exception as e:  # noqa: BLE001 - rede instável: tenta de novo
            if getattr(e, "code", None) == 404:
                raise NotPublished(f"{url.rsplit('/', 1)[-1]} não existe nesta release") from e
            last_error = e
            log(f"      tentativa {attempt} falhou: {e}")
            time.sleep(2 * attempt)
    raise OSError(f"não foi possível baixar {url}: {last_error}")


def _extract(archive: Path, patterns: list[str], dest: Path) -> list[str]:
    found = []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            base = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
            if base and any(fnmatch.fnmatch(base.lower(), p.lower()) for p in patterns):
                with z.open(info) as src, open(dest / base, "wb") as out:
                    shutil.copyfileobj(src, out)
                found.append(base)
    return found


def install(backend: str, root: Path | None = None, *, opener: Callable[..., Any] = urllib.request.urlopen,
            log: Callable[[str], None] = print, windows: bool = IS_WINDOWS) -> tuple[bool, str]:
    """Baixa o plugin oficial da release fixada e extrai só as DLLs necessárias."""
    spec = PLUGINS.get(backend)
    if spec is None:
        return False, f"'{backend}' não é um plugin de GPU"
    if not windows:
        return False, "plugins oficiais são baixados apenas no Windows"
    dest = plugin_dir(backend, root)
    tmp = dest.with_name(dest.name + ".tmp")
    skipped: list[str] = []
    try:
        for tag in COMPATIBLE_TAGS:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            try:
                for archive in spec["archives"]:
                    name = archive.format(tag=tag)
                    zip_path = tmp / name
                    _download(RELEASE_URL.format(tag=tag, name=name), zip_path, opener, log)
                    _extract(zip_path, spec["files"], tmp)
                    zip_path.unlink()
            except NotPublished:
                skipped.append(tag)
                log(f"      release {tag} sem binários de {backend}; tentando a próxima compatível...")
                continue
            missing = [f for f in spec["load"] if not (tmp / f).exists()]
            if missing:
                return False, f"arquivos ausentes no pacote oficial {tag}: {', '.join(missing)}"
            (tmp / "release.txt").write_text(tag, encoding="utf-8")
            shutil.rmtree(dest, ignore_errors=True)
            tmp.rename(dest)
            return True, f"plugin {backend} (llama.cpp {tag}) instalado em {dest}"
    except (OSError, zipfile.BadZipFile) as e:
        return False, str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return False, f"nenhuma release compatível tem o plugin {backend} ({', '.join(skipped)})"


def installed_release(backend: str, root: Path | None = None) -> str:
    try:
        return (plugin_dir(backend, root) / "release.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


# --------------------------------------------------------------------------- carga
def _native_path(path: Path) -> bytes:
    """Caminho no formato que o ggml espera em `const char *` (ANSI no Windows)."""
    text = str(path)
    if not IS_WINDOWS:
        return text.encode("utf-8")
    try:
        return text.encode("mbcs", errors="strict")
    except UnicodeEncodeError:  # nome com caracteres fora da página de código: usa o nome curto 8.3
        import ctypes

        buf = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.kernel32.GetShortPathNameW(text, buf, len(buf)):
            return buf.value.encode("mbcs", errors="replace")
        return text.encode("mbcs", errors="replace")


def load_plugin_files(plugins: list[Path], preload: list[Path] = ()) -> list[str]:
    """Registra no ggml os backends das DLLs/SOs indicadas; retorna os carregados."""
    import ctypes

    from llama_cpp import _ggml  # type: ignore

    for dep in preload:  # dependências (ex.: cudart/cuBLAS) já carregadas = resolvidas por nome
        try:
            ctypes.CDLL(str(dep))
        except OSError:
            pass
    global LAST_ERROR
    lib = _ggml.libggml
    lib.ggml_backend_load.argtypes = [ctypes.c_char_p]
    lib.ggml_backend_load.restype = ctypes.c_void_p
    loaded = []
    for path in plugins:
        path = Path(path).resolve()  # dlopen/LoadLibrary sem caminho absoluto não procuram na pasta atual
        if not path.exists():
            LAST_ERROR = f"{path.name} não encontrado em {path.parent}"
            continue
        try:  # carrega antes via ctypes só para obter a mensagem de erro do sistema, se houver
            _HANDLES.append(ctypes.CDLL(str(path)))
        except OSError as e:
            LAST_ERROR = f"{path.name} não pôde ser carregado: {e}"
            continue
        if lib.ggml_backend_load(_native_path(path)):
            loaded.append(path.name)
        else:
            LAST_ERROR = f"{path.name} recusado pelo ggml (versão incompatível ou GPU/driver sem suporte)"
    return loaded


def load(backend: str | None = None, root: Path | None = None) -> list[str]:
    """Carrega o plugin do backend instalado (uma única vez por processo)."""
    global _LOADED, LAST_ERROR
    if _LOADED is not None:
        return _LOADED
    _LOADED = []
    if backend is None:
        try:
            backend = (Path(sys.prefix) / "llama_backend.txt").read_text(encoding="utf-8").strip()
        except OSError:
            backend = ""
    spec = PLUGINS.get(backend)
    folder = plugin_dir(backend, root) if spec else None
    if spec is None or folder is None or not folder.is_dir():
        if spec is not None:
            LAST_ERROR = f"plugin {backend} não instalado ({folder})"
        return _LOADED
    if backend == "cuda":
        # Kernels compilados na hora (PTX) ficam em cache: só a 1ª execução é lenta.
        os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 2**30))
    os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
    if IS_WINDOWS:
        try:
            os.add_dll_directory(str(folder))
        except OSError:
            pass
    preload = [p for pattern in spec["preload"] for p in sorted(folder.glob(pattern))]
    _LOADED = load_plugin_files([folder / name for name in spec["load"]], preload)
    return _LOADED
