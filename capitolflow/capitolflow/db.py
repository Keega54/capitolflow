"""SQLite access layer. Plain sqlite3 — no ORM, so the schema stays readable."""
from __future__ import annotations
import hashlib, json, sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

from .config import SETTINGS

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    p = Path(path or SETTINGS.db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), timeout=60, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA_PATH.read_text())


@contextmanager
def session(path=None):
    con = connect(path)
    try:
        init_db(con)
        yield con
    finally:
        con.close()


@contextmanager
def tx(con: sqlite3.Connection):
    con.execute("BEGIN")
    try:
        yield con
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------- helpers
def upsert(con: sqlite3.Connection, table: str, row: dict, *, keys: Sequence[str] | None = None,
           mode: str = "REPLACE") -> None:
    cols = list(row.keys())
    ph = ",".join("?" * len(cols))
    sql = f"INSERT OR {mode} INTO {table} ({','.join(cols)}) VALUES ({ph})"
    con.execute(sql, [row[c] for c in cols])


def upsert_many(con: sqlite3.Connection, table: str, rows: Iterable[dict], mode: str = "IGNORE") -> int:
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0].keys())
    ph = ",".join("?" * len(cols))
    sql = f"INSERT OR {mode} INTO {table} ({','.join(cols)}) VALUES ({ph})"
    before = con.total_changes
    con.executemany(sql, [[r.get(c) for c in cols] for r in rows])
    return con.total_changes - before


def txn_id(filing_id: str, asset: str, txn_date, txn_type, amount_low, owner) -> str:
    """Deterministic id so re-parsing the same filing never duplicates rows."""
    key = "|".join(str(x) for x in (filing_id, asset, txn_date, txn_type, amount_low, owner))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


def json_safe(obj):
    """Recursively replace non-finite floats with None.

    Python's json module happily writes NaN and Infinity, which are NOT valid
    JSON. A browser's JSON.parse rejects the entire document, so one NaN buried
    in a statistics blob silently blanks a whole dashboard panel. Every value
    that leaves this process as JSON goes through here.
    """
    import math
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    # numpy scalars
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return json_safe(obj.item())
        except Exception:
            return obj
    return obj


def get_kv(con, k, default=None):
    r = con.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return json.loads(r["v"]) if r else default


def set_kv(con, k, v):
    con.execute("INSERT OR REPLACE INTO kv (k,v) VALUES (?,?)",
                (k, json.dumps(json_safe(v), default=str)))


def start_run(con, source: str) -> int:
    cur = con.execute(
        "INSERT INTO ingest_runs (source, started_at, status) VALUES (?, datetime('now'), 'running')",
        (source,))
    return cur.lastrowid


def finish_run(con, run_id: int, status: str, n_filings=0, n_txns=0, note: str = "") -> None:
    con.execute(
        "UPDATE ingest_runs SET finished_at=datetime('now'), status=?, n_new_filings=?, n_new_txns=?, note=? WHERE id=?",
        (status, n_filings, n_txns, note[:2000], run_id))


def existing_doc_ids(con, source: str) -> set[str]:
    return {r["doc_id"] for r in con.execute(
        "SELECT doc_id FROM filings WHERE source=? AND doc_id IS NOT NULL", (source,))}


def df(con, sql: str, params: Sequence = ()):
    import pandas as pd
    return pd.read_sql_query(sql, con, params=list(params))
