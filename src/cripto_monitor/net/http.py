"""Cliente HTTP minimo sobre urllib, com medicao de latencia.

Somente GET e POST de dados publicos ou locais. Nao envia credenciais.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

USER_AGENT = "cripto-monitor/0.1 (diagnostico somente leitura)"

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

# Opener sem proxy: o llama-server e local e nunca deve sair pela rede.
_direct_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def is_local(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname or ""
    return host in LOOPBACK_HOSTS or host.startswith("127.")


@dataclass(slots=True)
class HttpResponse:
    status: int
    body: bytes
    headers: dict[str, str]
    elapsed_ms: float
    url: str

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


class HttpError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


def request(
    url: str,
    *,
    method: str = "GET",
    payload: Any = None,
    timeout: float = 10.0,
    headers: dict[str, str] | None = None,
) -> HttpResponse:
    data = None
    all_headers = {"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        all_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=all_headers, method=method)
    opener = _direct_opener if is_local(url) else urllib.request.build_opener()
    started = time.perf_counter()
    try:
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read()
            elapsed = (time.perf_counter() - started) * 1000
            return HttpResponse(
                status=resp.status,
                body=body,
                headers={k.lower(): v for k, v in resp.headers.items()},
                elapsed_ms=elapsed,
                url=url,
            )
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise HttpError(
            f"HTTP {exc.code} em {url}",
            status=exc.code,
            body=body.decode("utf-8", errors="replace")[:500],
        ) from exc
    except urllib.error.URLError as exc:
        raise HttpError(f"falha de conexao com {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise HttpError(f"timeout ({timeout:.0f}s) em {url}") from exc


def get_json(url: str, *, timeout: float = 10.0) -> tuple[Any, HttpResponse]:
    resp = request(url, timeout=timeout)
    try:
        return resp.json(), resp
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HttpError(f"resposta nao e JSON valido em {url}: {exc}") from exc
