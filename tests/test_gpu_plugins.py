"""Instalador dos plugins oficiais de GPU do llama.cpp (download simulado)."""

import io
import urllib.error
import zipfile

import pytest

from core import gpu_plugins
from core.hardware import backend_candidates, wheel_index


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_opener(archives: dict[str, bytes], calls: list[str]):
    def opener(req, timeout=60):
        url = req.full_url
        calls.append(url)
        tag, name = url.rsplit("/", 2)[-2:]
        if (tag, name) not in archives:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return FakeResponse(archives[(tag, name)])
    return opener


def test_install_vulkan_extracts_only_the_plugin(tmp_path):
    tag = gpu_plugins.COMPATIBLE_TAGS[0]
    archives = {(tag, f"llama-{tag}-bin-win-vulkan-x64.zip"): _zip({
        "ggml-vulkan.dll": b"VK", "llama.dll": b"x", "ggml-base.dll": b"x", "llama-server.exe": b"x"})}
    calls: list[str] = []
    ok, msg = gpu_plugins.install("vulkan", tmp_path, opener=make_opener(archives, calls), windows=True,
                                  log=lambda m: None)
    folder = gpu_plugins.plugin_dir("vulkan", tmp_path)
    assert ok, msg
    assert sorted(p.name for p in folder.iterdir()) == ["ggml-vulkan.dll", "release.txt"]
    assert gpu_plugins.installed_release("vulkan", tmp_path) == tag
    assert calls == [f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-win-vulkan-x64.zip"]


def test_install_skips_releases_without_binaries(tmp_path):
    first, second = gpu_plugins.COMPATIBLE_TAGS[:2]
    archives = {(second, f"llama-{second}-bin-win-vulkan-x64.zip"): _zip({"ggml-vulkan.dll": b"VK"})}
    calls: list[str] = []
    logs: list[str] = []
    ok, msg = gpu_plugins.install("vulkan", tmp_path, opener=make_opener(archives, calls), windows=True,
                                  log=logs.append)
    assert ok and second in msg
    assert [c.split("/")[-2] for c in calls] == [first, second]  # 404 não é repetido: vai direto à próxima
    assert any(first in m for m in logs)
    assert gpu_plugins.installed_release("vulkan", tmp_path) == second


def test_install_cuda_brings_runtime_dlls(tmp_path):
    tag = gpu_plugins.COMPATIBLE_TAGS[0]
    archives = {
        (tag, f"llama-{tag}-bin-win-cuda-12.4-x64.zip"): _zip({"ggml-cuda.dll": b"CU", "ggml-cpu-haswell.dll": b"x"}),
        (tag, "cudart-llama-bin-win-cuda-12.4-x64.zip"): _zip({
            "cudart64_12.dll": b"r", "cublas64_12.dll": b"b", "cublasLt64_12.dll": b"l"}),
    }
    ok, msg = gpu_plugins.install("cuda", tmp_path, opener=make_opener(archives, []), windows=True, log=lambda m: None)
    assert ok, msg
    names = sorted(p.name for p in gpu_plugins.plugin_dir("cuda", tmp_path).iterdir())
    assert names == ["cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll", "ggml-cuda.dll", "release.txt"]


def test_install_failures_are_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_plugins.time, "sleep", lambda s: None)
    ok, msg = gpu_plugins.install("vulkan", tmp_path, opener=make_opener({}, []), windows=True, log=lambda m: None)
    assert not ok and "nenhuma release" in msg
    assert not gpu_plugins.plugin_dir("vulkan", tmp_path).exists()
    tag = gpu_plugins.COMPATIBLE_TAGS[0]
    bad = {(tag, f"llama-{tag}-bin-win-vulkan-x64.zip"): _zip({"outra.dll": b"x"})}
    ok, msg = gpu_plugins.install("vulkan", tmp_path, opener=make_opener(bad, []), windows=True, log=lambda m: None)
    assert not ok and "ggml-vulkan.dll" in msg

    def offline(req, timeout=60):
        raise OSError("sem rede")
    ok, msg = gpu_plugins.install("vulkan", tmp_path, opener=offline, windows=True, log=lambda m: None)
    assert not ok and "sem rede" in msg
    assert gpu_plugins.install("metal", tmp_path, windows=True)[0] is False
    assert gpu_plugins.install("vulkan", tmp_path, windows=False)[0] is False


def test_load_without_installed_plugin_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_plugins, "_LOADED", None)
    assert gpu_plugins.load("vulkan", tmp_path) == []
    monkeypatch.setattr(gpu_plugins, "_LOADED", None)
    assert gpu_plugins.load("cpu", tmp_path) == []


@pytest.mark.parametrize("gpus, cuda, expected", [
    ([{"vendor": "amd", "integrated": False}], None, ["vulkan", "cpu"]),
    ([{"vendor": "nvidia", "compute_capability": 6.1}], 12.8, ["cu125", "cu124", "cuda", "vulkan", "cpu"]),
    ([{"vendor": "nvidia", "compute_capability": 6.1}], 12.2, ["cu118", "vulkan", "cpu"]),
    ([{"vendor": "nvidia", "compute_capability": 5.2}], 12.6, ["cuda", "vulkan", "cpu"]),
    ([{"vendor": "intel", "integrated": True}], None, ["cpu"]),
    ([], None, ["cpu"]),
])
def test_backend_order(gpus, cuda, expected, monkeypatch):
    monkeypatch.delenv("IA_BACKEND", raising=False)
    assert backend_candidates(gpus, cuda) == expected


def test_plugin_backends_use_the_cpu_wheel():
    assert wheel_index("vulkan") == "cpu" and wheel_index("cuda") == "cpu"
    assert wheel_index("cu124") == "cu124" and wheel_index("cpu") == "cpu"
