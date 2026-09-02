"""Testes do cliente WebSocket usado na sondagem de diagnostico."""

from __future__ import annotations

import json
import socket
import threading
import unittest

from cripto_monitor.net import wsprobe
from cripto_monitor.net.wsprobe import Frame, accept_key, decode_frames, encode_frame


def server_frame(opcode: int, payload: bytes = b"", fin: bool = True) -> bytes:
    """Frame do lado servidor: mesmo formato, porem sem mascara (RFC 6455, 5.1)."""
    header = bytearray([(0x80 if fin else 0x00) | opcode])
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < 1 << 16:
        header.append(126)
        header += length.to_bytes(2, "big")
    else:
        header.append(127)
        header += length.to_bytes(8, "big")
    return bytes(header) + payload


class TestHandshake(unittest.TestCase):
    def test_accept_key_usa_o_vetor_da_rfc6455(self) -> None:
        self.assertEqual(accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")


class TestFrames(unittest.TestCase):
    def test_ida_e_volta_com_mascara(self) -> None:
        bruto = encode_frame(wsprobe.OP_TEXT, "vela fechada".encode("utf-8"))
        frames, resto = decode_frames(bruto)
        self.assertEqual(resto, b"")
        self.assertEqual(frames, [Frame(fin=True, opcode=wsprobe.OP_TEXT, payload=b"vela fechada")])

    def test_cliente_sempre_mascara(self) -> None:
        bruto = encode_frame(wsprobe.OP_TEXT, b"x")
        self.assertTrue(bruto[1] & 0x80, "bit de mascara obrigatorio no frame do cliente")

    def test_payload_medio_e_grande(self) -> None:
        for tamanho in (125, 126, 70000):
            with self.subTest(tamanho=tamanho):
                payload = b"a" * tamanho
                frames, resto = decode_frames(server_frame(wsprobe.OP_BINARY, payload))
                self.assertEqual(resto, b"")
                self.assertEqual(frames[0].payload, payload)

    def test_buffer_incompleto_nao_produz_frame(self) -> None:
        bruto = server_frame(wsprobe.OP_TEXT, b"parcial")
        frames, resto = decode_frames(bruto[:5])
        self.assertEqual(frames, [])
        self.assertEqual(resto, bruto[:5])
        # o resto guardado, somado ao restante, fecha o frame
        frames, resto = decode_frames(resto + bruto[5:])
        self.assertEqual(frames[0].payload, b"parcial")
        self.assertEqual(resto, b"")

    def test_frames_multiplos_em_um_recv(self) -> None:
        bruto = server_frame(wsprobe.OP_TEXT, b"um") + server_frame(wsprobe.OP_PING, b"p")
        frames, resto = decode_frames(bruto)
        self.assertEqual([f.opcode for f in frames], [wsprobe.OP_TEXT, wsprobe.OP_PING])
        self.assertEqual(resto, b"")

    def test_payload_absurdo_e_recusado(self) -> None:
        cabecalho = bytes([0x81, 127]) + (wsprobe.MAX_PAYLOAD + 1).to_bytes(8, "big")
        with self.assertRaises(wsprobe.WebSocketError):
            decode_frames(cabecalho)


class FakeWebSocketServer:
    """Servidor WebSocket minimo, em thread, para exercitar probe() de ponta a ponta."""

    def __init__(self, frames: list[bytes], *, aceitar_invalido: bool = False) -> None:
        self.frames = frames
        self.aceitar_invalido = aceitar_invalido
        self.parar = threading.Event()
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.pongs: list[bytes] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "FakeWebSocketServer":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.parar.set()
        self.sock.close()
        self.thread.join(timeout=5)

    def _run(self) -> None:
        try:
            conn, _ = self.sock.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(5)
            pedido = b""
            while b"\r\n\r\n" not in pedido:
                pedido += conn.recv(4096)
            chave = ""
            for linha in pedido.decode("latin-1").split("\r\n"):
                nome, _, valor = linha.partition(":")
                if nome.strip().lower() == "sec-websocket-key":
                    chave = valor.strip()
            aceite = "chave-errada" if self.aceitar_invalido else accept_key(chave)
            conn.sendall(
                (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {aceite}\r\n\r\n"
                ).encode("ascii")
            )
            for frame in self.frames:
                conn.sendall(frame)
            # Mantem a conexao aberta ate o cliente encerrar: sem isso, um EOF do
            # servidor mascararia o cenario de "conexao viva, porem sem dados".
            conn.settimeout(0.2)
            while not self.parar.is_set():
                try:
                    dados = conn.recv(4096)
                except TimeoutError:
                    continue
                except OSError:
                    break
                if not dados:
                    break
                self.pongs.append(dados)


class TestProbe(unittest.TestCase):
    def test_coleta_mensagens_e_responde_ping(self) -> None:
        eventos = [
            json.dumps({"e": "kline", "E": 1, "k": {"t": 0, "T": 1, "c": "1", "v": "1", "x": False}}),
            json.dumps({"e": "kline", "E": 2, "k": {"t": 0, "T": 1, "c": "2", "v": "1", "x": True}}),
        ]
        frames = [
            server_frame(wsprobe.OP_TEXT, eventos[0].encode()),
            server_frame(wsprobe.OP_PING, b"hb"),
            server_frame(wsprobe.OP_TEXT, eventos[1].encode()),
        ]
        with FakeWebSocketServer(frames) as servidor:
            resultado = wsprobe.probe(
                f"ws://127.0.0.1:{servidor.port}/ws/ethusdt@kline_5m", messages=2, timeout=5
            )
        self.assertEqual(resultado.messages, eventos)
        self.assertEqual(resultado.pings_recebidos, 1)
        self.assertGreater(resultado.handshake_ms, 0)

    def test_remonta_mensagem_fragmentada(self) -> None:
        frames = [
            server_frame(wsprobe.OP_TEXT, b'{"parte":', fin=False),
            server_frame(wsprobe.OP_CONT, b'"dois"}'),
        ]
        with FakeWebSocketServer(frames) as servidor:
            resultado = wsprobe.probe(f"ws://127.0.0.1:{servidor.port}/", messages=1, timeout=5)
        self.assertEqual(resultado.messages, ['{"parte":"dois"}'])

    def test_handshake_com_aceite_invalido_falha(self) -> None:
        with FakeWebSocketServer([], aceitar_invalido=True) as servidor:
            with self.assertRaisesRegex(wsprobe.WebSocketError, "Accept invalido"):
                wsprobe.probe(f"ws://127.0.0.1:{servidor.port}/", messages=1, timeout=5)

    def test_conexao_viva_sem_mensagens_estoura_timeout(self) -> None:
        """Cenario 'conexao aparentemente viva, mas sem dados' citado na spec."""
        with FakeWebSocketServer([]) as servidor:
            with self.assertRaisesRegex(wsprobe.WebSocketError, "sem 1 mensagens"):
                wsprobe.probe(f"ws://127.0.0.1:{servidor.port}/", messages=1, timeout=1)

    def test_esquema_invalido(self) -> None:
        with self.assertRaises(wsprobe.WebSocketError):
            wsprobe.probe("https://exemplo.invalido/", messages=1, timeout=1)


class TestProxyBypass(unittest.TestCase):
    def test_local_e_no_proxy_nao_usam_tunel(self) -> None:
        env = {"https_proxy": "http://proxy:3128", "no_proxy": "example.com"}
        self.assertIsNone(wsprobe._proxy_for("wss", "127.0.0.1", env))
        self.assertIsNone(wsprobe._proxy_for("wss", "api.example.com", env))
        self.assertEqual(wsprobe._proxy_for("wss", "stream.binance.com", env), ("proxy", 3128))


if __name__ == "__main__":
    unittest.main()
