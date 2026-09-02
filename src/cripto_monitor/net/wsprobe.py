"""Cliente WebSocket minimo (RFC 6455) para sondagem de diagnostico.

Proposito unico: provar, na Fase 1, que o ambiente alcanca o stream publico da
exchange e recebe eventos de vela. Nao e o coletor de producao -- a Fase 2 usara
uma biblioteca assincrona dedicada, com heartbeat, backoff e re-subscription.
Implementado apenas com a biblioteca padrao para nao exigir instalacao de nada.
"""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import ssl
import time
from dataclasses import dataclass
from urllib.parse import urlparse

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONT = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

MAX_PAYLOAD = 4 * 1024 * 1024


class WebSocketError(RuntimeError):
    pass


@dataclass(slots=True)
class Frame:
    fin: bool
    opcode: int
    payload: bytes


def accept_key(client_key: str) -> str:
    """Valor esperado em Sec-WebSocket-Accept para uma dada Sec-WebSocket-Key."""
    digest = hashlib.sha1((client_key + GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def encode_frame(opcode: int, payload: bytes = b"", *, mask: bytes | None = None) -> bytes:
    """Monta um frame de cliente. O cliente sempre mascara (RFC 6455, 5.3)."""
    mask = os.urandom(4) if mask is None else mask
    if len(mask) != 4:
        raise ValueError("mask deve ter 4 bytes")
    header = bytearray([0x80 | opcode])
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    elif length < 1 << 16:
        header.append(0x80 | 126)
        header += length.to_bytes(2, "big")
    else:
        header.append(0x80 | 127)
        header += length.to_bytes(8, "big")
    header += mask
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return bytes(header) + masked


def decode_frames(buffer: bytes) -> tuple[list[Frame], bytes]:
    """Extrai frames completos do buffer e devolve o resto ainda incompleto."""
    frames: list[Frame] = []
    offset = 0
    total = len(buffer)
    while True:
        if total - offset < 2:
            break
        first, second = buffer[offset], buffer[offset + 1]
        fin = bool(first & 0x80)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        cursor = offset + 2
        if length == 126:
            if total - cursor < 2:
                break
            length = int.from_bytes(buffer[cursor : cursor + 2], "big")
            cursor += 2
        elif length == 127:
            if total - cursor < 8:
                break
            length = int.from_bytes(buffer[cursor : cursor + 8], "big")
            cursor += 8
        if length > MAX_PAYLOAD:
            raise WebSocketError(f"frame acima do limite aceito: {length} bytes")
        mask_key = b""
        if masked:
            if total - cursor < 4:
                break
            mask_key = buffer[cursor : cursor + 4]
            cursor += 4
        if total - cursor < length:
            break
        payload = buffer[cursor : cursor + length]
        if masked:
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        frames.append(Frame(fin=fin, opcode=opcode, payload=payload))
        offset = cursor + length
    return frames, buffer[offset:]


LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _bypass_proxy(host: str, env: dict[str, str]) -> bool:
    if host in LOOPBACK_HOSTS or host.startswith("127."):
        return True
    regras = env.get("no_proxy") or env.get("NO_PROXY") or ""
    for regra in (r.strip().lstrip(".").lower() for r in regras.split(",")):
        if regra and (regra == "*" or host.lower() == regra or host.lower().endswith("." + regra)):
            return True
    return False


def _proxy_for(scheme: str, host: str, env: dict[str, str] | None = None) -> tuple[str, int] | None:
    env = os.environ if env is None else env
    if _bypass_proxy(host, env):
        return None
    var = "https_proxy" if scheme == "wss" else "http_proxy"
    raw = env.get(var) or env.get(var.upper())
    if not raw:
        return None
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    if not parsed.hostname:
        return None
    return parsed.hostname, parsed.port or 8080


def _open_socket(host: str, port: int, scheme: str, timeout: float) -> socket.socket:
    proxy = _proxy_for(scheme, host)
    target = proxy or (host, port)
    sock = socket.create_connection(target, timeout=timeout)
    if proxy:
        sock.sendall(
            f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode("ascii")
        )
        head = _read_until(sock, b"\r\n\r\n", timeout)
        status_line = head.split(b"\r\n", 1)[0].decode("latin-1")
        if " 200 " not in status_line:
            sock.close()
            raise WebSocketError(f"proxy recusou o tunel: {status_line}")
    if scheme == "wss":
        context = ssl.create_default_context()
        sock = context.wrap_socket(sock, server_hostname=host)
    return sock


def _read_until(sock: socket.socket, terminator: bytes, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    buffer = b""
    while terminator not in buffer:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WebSocketError("timeout aguardando resposta do servidor")
        sock.settimeout(remaining)
        try:
            chunk = sock.recv(4096)
        except TimeoutError as exc:
            raise WebSocketError("timeout aguardando resposta do servidor") from exc
        if not chunk:
            raise WebSocketError("conexao fechada durante o handshake")
        buffer += chunk
        if len(buffer) > 64 * 1024:
            raise WebSocketError("cabecalho de handshake absurdamente grande")
    return buffer


@dataclass(slots=True)
class ProbeResult:
    handshake_ms: float
    first_message_ms: float
    messages: list[str]
    pings_recebidos: int
    close_code: int | None


def probe(
    url: str,
    *,
    messages: int = 3,
    timeout: float = 30.0,
    connect_timeout: float = 10.0,
) -> ProbeResult:
    """Abre a conexao, coleta ate `messages` mensagens de texto e encerra limpo."""
    parsed = urlparse(url)
    if parsed.scheme not in {"ws", "wss"}:
        raise WebSocketError(f"esquema invalido para WebSocket: {parsed.scheme!r}")
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    key = base64.b64encode(os.urandom(16)).decode("ascii")
    started = time.perf_counter()
    sock = _open_socket(host, port, parsed.scheme, connect_timeout)
    try:
        handshake = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "User-Agent: cripto-monitor/0.1 (diagnostico somente leitura)\r\n\r\n"
        )
        sock.sendall(handshake.encode("ascii"))
        head = _read_until(sock, b"\r\n\r\n", connect_timeout)
        header_blob, _, rest = head.partition(b"\r\n\r\n")
        lines = header_blob.decode("latin-1").split("\r\n")
        if "101" not in lines[0]:
            raise WebSocketError(f"handshake recusado: {lines[0]}")
        headers = {}
        for line in lines[1:]:
            name, _, value = line.partition(":")
            headers[name.strip().lower()] = value.strip()
        if headers.get("sec-websocket-accept") != accept_key(key):
            raise WebSocketError("Sec-WebSocket-Accept invalido: handshake nao confiavel")
        handshake_ms = (time.perf_counter() - started) * 1000

        buffer = rest
        coletadas: list[str] = []
        pings = 0
        fragmentos: list[bytes] = []
        primeira_ms = 0.0
        deadline = time.monotonic() + timeout
        while len(coletadas) < messages:
            frames, buffer = decode_frames(buffer)
            for frame in frames:
                if frame.opcode in (OP_TEXT, OP_BINARY, OP_CONT):
                    fragmentos.append(frame.payload)
                    if frame.fin:
                        texto = b"".join(fragmentos).decode("utf-8", errors="replace")
                        fragmentos.clear()
                        if not primeira_ms:
                            primeira_ms = (time.perf_counter() - started) * 1000
                        coletadas.append(texto)
                elif frame.opcode == OP_PING:
                    pings += 1
                    sock.sendall(encode_frame(OP_PONG, frame.payload))
                elif frame.opcode == OP_CLOSE:
                    code = int.from_bytes(frame.payload[:2], "big") if frame.payload else None
                    return ProbeResult(handshake_ms, primeira_ms, coletadas, pings, code)
            if len(coletadas) >= messages:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WebSocketError(
                    f"conexao aberta, mas sem {messages} mensagens em {timeout:.0f}s"
                )
            sock.settimeout(remaining)
            try:
                chunk = sock.recv(65536)
            except TimeoutError as exc:
                raise WebSocketError(
                    f"conexao aberta, mas sem {messages} mensagens em {timeout:.0f}s"
                ) from exc
            if not chunk:
                raise WebSocketError("servidor encerrou a conexao sem frame de close")
            buffer += chunk
        sock.sendall(encode_frame(OP_CLOSE, (1000).to_bytes(2, "big")))
        return ProbeResult(handshake_ms, primeira_ms, coletadas, pings, 1000)
    finally:
        try:
            sock.close()
        except OSError:
            pass
