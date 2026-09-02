"""Persistencia local: SQLite para estado operacional, JSONL para o bruto.

Duas garantias sustentam todo o resto do projeto:

1. Somente vela fechada entra em `candles`. Vela em formacao vive em memoria.
2. Reprocessar a mesma vela e inofensivo. `save_closed_candle` devolve True
   apenas na primeira vez que aquela chave e persistida, e e esse retorno que
   impedira, na Fase 4, um alerta duplicado depois de reconexao ou reinicio.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import time
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cripto_monitor.candles import Candle, merge

SCHEMA_VERSION = "2"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Somente velas fechadas. A chave primaria e a chave de idempotencia do projeto.
CREATE TABLE IF NOT EXISTS candles (
    source        TEXT    NOT NULL,
    symbol        TEXT    NOT NULL,
    timeframe     TEXT    NOT NULL,
    open_time_ms  INTEGER NOT NULL,
    close_time_ms INTEGER NOT NULL,
    open          REAL    NOT NULL,
    high          REAL    NOT NULL,
    low           REAL    NOT NULL,
    close         REAL    NOT NULL,
    volume        REAL    NOT NULL,
    quote_volume  REAL    NOT NULL,
    trades        INTEGER NOT NULL,
    origin        TEXT    NOT NULL,
    first_seen_ms INTEGER NOT NULL,
    updated_ms    INTEGER NOT NULL,
    PRIMARY KEY (source, symbol, timeframe, open_time_ms)
);

CREATE INDEX IF NOT EXISTS idx_candles_busca
    ON candles (symbol, timeframe, open_time_ms);

-- Diario operacional: reconexoes, lacunas, dados invalidos, atrasos.
CREATE TABLE IF NOT EXISTS ops_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms     INTEGER NOT NULL,
    kind      TEXT    NOT NULL,
    severity  TEXT    NOT NULL,
    symbol    TEXT,
    timeframe TEXT,
    detail    TEXT
);

CREATE INDEX IF NOT EXISTS idx_ops_events_ts ON ops_events (ts_ms DESC);

CREATE TABLE IF NOT EXISTS runtime_state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_ms INTEGER NOT NULL
);
"""

_CAMPOS = (
    "source, symbol, timeframe, open_time_ms, close_time_ms, open, high, low, close, "
    "volume, quote_volume, trades, origin, first_seen_ms, updated_ms"
)


def agora_ms() -> int:
    return int(time.time() * 1000)


def _para_candle(linha: sqlite3.Row) -> Candle:
    return Candle(
        source=linha["source"],
        symbol=linha["symbol"],
        timeframe=linha["timeframe"],
        open_time_ms=linha["open_time_ms"],
        close_time_ms=linha["close_time_ms"],
        open=linha["open"],
        high=linha["high"],
        low=linha["low"],
        close=linha["close"],
        volume=linha["volume"],
        quote_volume=linha["quote_volume"],
        trades=linha["trades"],
        closed=True,
        origin=linha["origin"],
    )


class Storage:
    """Estado operacional em SQLite e trilha bruta em JSONL comprimido."""

    def __init__(self, db_path: Path, raw_dir: Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self._conn: sqlite3.Connection | None = None
        self._raw_handle: Any = None
        self._raw_dia: str | None = None

    # ---------------------------------------------------------------- conexao

    def __enter__(self) -> "Storage":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, isolation_level=None)
            conn.row_factory = sqlite3.Row
            # WAL: o dashboard pode ler enquanto o coletor escreve.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (SCHEMA_VERSION,),
            )
            self._conn = conn
        return self._conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self.connect()

    def close(self) -> None:
        if self._raw_handle is not None:
            self._raw_handle.close()
            self._raw_handle = None
            self._raw_dia = None
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ----------------------------------------------------------------- velas

    def save_closed_candle(self, candle: Candle) -> bool:
        """Persiste uma vela fechada. True apenas na primeira gravacao da chave.

        Reentregas apos reconexao, backfill sobreposto e replay caem no ramo de
        atualizacao e devolvem False: e isso que torna o pipeline idempotente.
        """
        if not candle.closed:
            raise ValueError(
                f"vela em formacao nao pode ser persistida: {candle.open_time_utc}"
            )
        existente = self.get_candle(candle.key)
        agora = agora_ms()
        if existente is None:
            self.conn.execute(
                f"INSERT INTO candles ({_CAMPOS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    candle.source, candle.symbol, candle.timeframe, candle.open_time_ms,
                    candle.close_time_ms, candle.open, candle.high, candle.low, candle.close,
                    candle.volume, candle.quote_volume, candle.trades, candle.origin,
                    agora, agora,
                ),
            )
            return True
        final = merge(existente, candle)
        self.conn.execute(
            "UPDATE candles SET close_time_ms=?, open=?, high=?, low=?, close=?, volume=?, "
            "quote_volume=?, trades=?, origin=?, updated_ms=? "
            "WHERE source=? AND symbol=? AND timeframe=? AND open_time_ms=?",
            (
                final.close_time_ms, final.open, final.high, final.low, final.close,
                final.volume, final.quote_volume, final.trades, final.origin, agora,
                *candle.key,
            ),
        )
        return False

    def save_many(self, candles: Iterable[Candle]) -> int:
        """Persiste um lote e devolve quantas velas eram ineditas."""
        novas = 0
        with self.conn:
            for candle in candles:
                if candle.closed and self.save_closed_candle(candle):
                    novas += 1
        return novas

    def get_candle(self, key: tuple[str, str, str, int]) -> Candle | None:
        linha = self.conn.execute(
            f"SELECT {_CAMPOS} FROM candles "
            "WHERE source=? AND symbol=? AND timeframe=? AND open_time_ms=?",
            key,
        ).fetchone()
        return _para_candle(linha) if linha else None

    def iter_candles(
        self,
        symbol: str,
        timeframe: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> Iterator[Candle]:
        sql = f"SELECT {_CAMPOS} FROM candles WHERE symbol=? AND timeframe=?"
        params: list[Any] = [symbol, timeframe]
        if start_ms is not None:
            sql += " AND open_time_ms >= ?"
            params.append(start_ms)
        if end_ms is not None:
            sql += " AND open_time_ms <= ?"
            params.append(end_ms)
        sql += " ORDER BY open_time_ms " + ("DESC" if newest_first else "ASC")
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        for linha in self.conn.execute(sql, params):
            yield _para_candle(linha)

    def open_times(
        self, symbol: str, timeframe: str, *, start_ms: int | None = None
    ) -> list[int]:
        sql = "SELECT open_time_ms FROM candles WHERE symbol=? AND timeframe=?"
        params: list[Any] = [symbol, timeframe]
        if start_ms is not None:
            sql += " AND open_time_ms >= ?"
            params.append(start_ms)
        sql += " ORDER BY open_time_ms"
        return [linha[0] for linha in self.conn.execute(sql, params)]

    def last_candle(self, symbol: str, timeframe: str) -> Candle | None:
        return next(iter(self.iter_candles(symbol, timeframe, limit=1, newest_first=True)), None)

    def count_candles(self, symbol: str, timeframe: str) -> int:
        linha = self.conn.execute(
            "SELECT COUNT(*) FROM candles WHERE symbol=? AND timeframe=?", (symbol, timeframe)
        ).fetchone()
        return int(linha[0])

    # ------------------------------------------------------------ diario ops

    def record_event(
        self,
        kind: str,
        *,
        severity: str = "info",
        symbol: str | None = None,
        timeframe: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO ops_events (ts_ms, kind, severity, symbol, timeframe, detail) "
            "VALUES (?,?,?,?,?,?)",
            (
                agora_ms(),
                kind,
                severity,
                symbol,
                timeframe,
                json.dumps(detail, ensure_ascii=False, default=str) if detail else None,
            ),
        )

    def recent_events(self, limit: int = 20, *, kind: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT ts_ms, kind, severity, symbol, timeframe, detail FROM ops_events"
        params: list[Any] = []
        if kind:
            sql += " WHERE kind=?"
            params.append(kind)
        sql += " ORDER BY ts_ms DESC, id DESC LIMIT ?"
        params.append(limit)
        eventos = []
        for linha in self.conn.execute(sql, params):
            evento = dict(linha)
            evento["detail"] = json.loads(evento["detail"]) if evento["detail"] else None
            eventos.append(evento)
        return eventos

    def count_events(self, kind: str, *, since_ms: int | None = None) -> int:
        sql = "SELECT COUNT(*) FROM ops_events WHERE kind=?"
        params: list[Any] = [kind]
        if since_ms is not None:
            sql += " AND ts_ms >= ?"
            params.append(since_ms)
        return int(self.conn.execute(sql, params).fetchone()[0])

    # ---------------------------------------------------------------- estado

    def set_state(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO runtime_state (key, value, updated_ms) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_ms=excluded.updated_ms",
            (key, json.dumps(value, ensure_ascii=False, default=str), agora_ms()),
        )

    def get_state(self, key: str, default: Any = None) -> Any:
        linha = self.conn.execute(
            "SELECT value FROM runtime_state WHERE key=?", (key,)
        ).fetchone()
        return json.loads(linha[0]) if linha else default

    # ------------------------------------------------------------ trilha bruta

    def append_raw(self, source: str, symbol: str, message: str) -> None:
        """Guarda a mensagem original, para auditoria e replay fiel."""
        if self.raw_dir is None:
            return
        dia = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._raw_dia != dia:
            if self._raw_handle is not None:
                self._raw_handle.close()
            destino = self.raw_dir / source / symbol.upper()
            destino.mkdir(parents=True, exist_ok=True)
            self._raw_handle = gzip.open(destino / f"{dia}.jsonl.gz", "at", encoding="utf-8")
            self._raw_dia = dia
        self._raw_handle.write(
            json.dumps({"ts_ms": agora_ms(), "raw": message}, ensure_ascii=False) + "\n"
        )

    def flush_raw(self) -> None:
        if self._raw_handle is not None:
            self._raw_handle.flush()

    # ----------------------------------------------------------------- saude

    def stats(self, symbol: str, timeframe: str) -> dict[str, Any]:
        ultima = self.last_candle(symbol, timeframe)
        primeira = next(iter(self.iter_candles(symbol, timeframe, limit=1)), None)
        return {
            "db_path": str(self.db_path),
            "schema_version": SCHEMA_VERSION,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": self.count_candles(symbol, timeframe),
            "primeira_vela_utc": primeira.open_time_utc if primeira else None,
            "ultima_vela_utc": ultima.open_time_utc if ultima else None,
            "ultima_vela_ms": ultima.close_time_ms if ultima else None,
            "idade_ultima_vela_s": (
                round((agora_ms() - ultima.close_time_ms) / 1000, 1) if ultima else None
            ),
            "reconexoes": self.count_events("reconnect"),
            "lacunas": self.count_events("gap"),
            "descartes": self.count_events("invalid_message"),
        }
