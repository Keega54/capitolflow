"""Storage-layer invariants: re-ingesting the same filing must not duplicate rows."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capitolflow import db
from capitolflow.sources.house import _store_rows
from capitolflow.util.tickers import TickerResolver
from capitolflow.parse.house_ptr import parse_ptr_text

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def con(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.init_db(c)
    yield c
    c.close()


def test_reingest_is_idempotent(con):
    rows = list(parse_ptr_text((FIX / "house_ptr_sample.txt").read_text()))
    r = TickerResolver.from_sec_json({"0": {"ticker": "AAPL", "title": "Apple Inc."}})
    db.upsert(con, "filings", {"filing_id": "house:1", "source": "house", "doc_id": "1",
                               "filing_type": "ptr", "parse_status": "ok"})
    n1 = _store_rows(con, rows, "house:1", None, "2026-02-19", "house", r)
    n2 = _store_rows(con, rows, "house:1", None, "2026-02-19", "house", r)
    total = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    assert n1 == len(rows)
    assert n2 == 0, "a second parse of the same filing must insert nothing"
    assert total == len(rows)


def test_filing_delay_is_stored(con):
    rows = list(parse_ptr_text((FIX / "house_ptr_sample.txt").read_text()))
    db.upsert(con, "filings", {"filing_id": "house:2", "source": "house", "doc_id": "2",
                               "filing_type": "ptr", "parse_status": "ok"})
    _store_rows(con, rows, "house:2", None, "2026-02-19", "house", TickerResolver({}))
    r = con.execute("SELECT transaction_date, filed_date, filing_delay_days FROM transactions "
                    "WHERE transaction_date='2026-01-14'").fetchone()
    assert r["filing_delay_days"] == 36


def test_health_counts(con):
    from capitolflow.pipeline import health
    h = health(con)
    assert h["transactions"] == 0 and h["filings"] == 0


def test_schema_has_no_orphan_transactions(con):
    """Foreign keys must be enforced so a transaction cannot outlive its filing."""
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO transactions (txn_id, filing_id) VALUES ('x','missing')")


# ---------------------------------------------------------------- circuit breaker
def test_circuit_breaker_trips_and_recovers():
    """A host that keeps failing must be skipped instantly rather than retried."""
    import pytest
    from capitolflow.util.http import HostUnavailable, _CircuitBreaker

    cb = _CircuitBreaker(threshold=3, reset_after=0.05)
    url = "https://example.invalid/a.pdf"
    for _ in range(3):
        cb.check(url)                      # not tripped yet
        cb.record_failure(url)
    with pytest.raises(HostUnavailable):
        cb.check(url)

    import time
    time.sleep(0.06)
    cb.check(url)                          # half-open: one probe allowed
    cb.record_success(url)
    cb.check(url)                          # fully closed again


def test_circuit_breaker_is_per_host():
    from capitolflow.util.http import HostUnavailable, _CircuitBreaker
    import pytest
    cb = _CircuitBreaker(threshold=2, reset_after=60)
    for _ in range(2):
        cb.record_failure("https://down.invalid/x")
    with pytest.raises(HostUnavailable):
        cb.check("https://down.invalid/x")
    cb.check("https://up.invalid/x")        # unrelated host is unaffected


# ---------------------------------------------------------------- run budget
def test_budget_is_shared_across_years():
    """A multi-year backfill must share one allowance, not spend it per year."""
    from capitolflow.sources.house import Budget
    b = Budget(10)
    assert b.remaining == 10 and not b.exhausted
    b.spend(4)
    assert b.remaining == 6
    b.spend(6)
    assert b.exhausted and b.remaining == 0
    b.spend(3)
    assert b.remaining == 0, "remaining must never go negative"


def test_zero_budget_is_immediately_exhausted():
    from capitolflow.sources.house import Budget
    assert Budget(0).exhausted
