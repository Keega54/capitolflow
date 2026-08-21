"""Tests for price provider failover and loud failure.

These exist because a live deployment reported the price stage as "ok" while
fetching zero rows, and the resulting empty dashboard gave no hint why."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capitolflow.sources import prices
from capitolflow.sources.prices import ProviderRefused, _status, provider_order


def test_zero_rows_is_never_ok():
    r = {"benchmark_rows": 0, "rows": 0, "providers_retired": {"stooq": "limit"},
         "benchmark": "SPY", "tickers_with_data": 0}
    assert _status(r, 50).startswith("FAILED")
    assert "SPY" in _status(r, 50)


def test_missing_benchmark_fails_even_with_other_data():
    """Other tickers are worthless without the benchmark: no benchmark, no excess return."""
    r = {"benchmark_rows": 0, "rows": 90000, "providers_retired": {},
         "benchmark": "SPY", "tickers_with_data": 140}
    assert _status(r, 150).startswith("FAILED")


def test_partial_coverage_is_degraded_not_ok():
    r = {"benchmark_rows": 4000, "rows": 9000, "providers_retired": {"yahoo": "429"},
         "benchmark": "SPY", "tickers_with_data": 5}
    assert _status(r, 150).startswith("DEGRADED")


def test_nothing_to_do_is_ok():
    r = {"benchmark_rows": 4000, "rows": 0, "providers_retired": {},
         "benchmark": "SPY", "tickers_with_data": 0}
    assert _status(r, 0) == "ok"


def test_healthy_run_is_ok():
    r = {"benchmark_rows": 4000, "rows": 90000, "providers_retired": {},
         "benchmark": "SPY", "tickers_with_data": 140}
    assert _status(r, 150) == "ok"


def test_provider_order_puts_configured_first_and_keeps_fallbacks():
    order = provider_order()
    assert order[0] == "yahoo", "cloud-friendly provider should lead by default"
    assert set(order) == {"stooq", "yahoo", "fmp"}, "all fallbacks must remain available"


def test_rate_limit_text_raises_refused_not_empty(monkeypatch):
    """A rate-limit page must not be mistaken for 'this ticker has no data' —
    otherwise the run cheerfully asks 800 more times and reports success."""
    monkeypatch.setattr(prices, "get_bytes",
                        lambda *a, **k: b"Exceeded the daily hits limit")
    with pytest.raises(ProviderRefused):
        prices.fetch_stooq(None, "SPY")


def test_unknown_ticker_returns_empty_not_refused(monkeypatch):
    monkeypatch.setattr(prices, "get_bytes", lambda *a, **k: b"<html>404</html>")
    assert prices.fetch_stooq(None, "ZZZZ") == []


def test_failover_uses_second_provider(monkeypatch, tmp_path):
    """When the first provider refuses, the run must continue on the next."""
    from capitolflow import db
    calls = []

    def refuse(session, ticker, **kw):
        calls.append(("stooq", ticker))
        raise ProviderRefused("daily limit")

    def works(session, ticker, **kw):
        calls.append(("yahoo", ticker))
        return [{"ticker": ticker.upper(), "date": "2024-01-02", "open": 1.0,
                 "high": 1.0, "low": 1.0, "close": 1.0, "adj_close": 1.0, "volume": 1}]

    monkeypatch.setitem(prices.ALL_PROVIDERS, "stooq", refuse)
    monkeypatch.setitem(prices.ALL_PROVIDERS, "yahoo", works)
    monkeypatch.setattr(prices, "provider_order", lambda: ["stooq", "yahoo"])
    monkeypatch.setattr(prices, "make_session", lambda: None)

    with db.session(tmp_path / "p.db") as con:
        rep = prices.sync_prices(con, tickers=["SPY", "AAPL", "MSFT"],
                                 include_benchmark=False)
    assert rep["rows"] == 3
    assert rep["by_provider"].get("yahoo") == 3
    assert "stooq" in rep["providers_retired"]
    # stooq must be asked once, then retired — not retried for every ticker.
    assert sum(1 for p, _ in calls if p == "stooq") == 1


def test_all_providers_dead_stops_early(monkeypatch, tmp_path):
    from capitolflow import db

    def refuse(session, ticker, **kw):
        raise ProviderRefused("blocked")

    monkeypatch.setitem(prices.ALL_PROVIDERS, "stooq", refuse)
    monkeypatch.setitem(prices.ALL_PROVIDERS, "yahoo", refuse)
    monkeypatch.setattr(prices, "provider_order", lambda: ["stooq", "yahoo"])
    monkeypatch.setattr(prices, "make_session", lambda: None)
    with db.session(tmp_path / "p2.db") as con:
        rep = prices.sync_prices(con, tickers=["A", "B", "C", "D"],
                                 include_benchmark=False)
    assert rep["rows"] == 0
    assert rep["status"].startswith("FAILED")
    assert len(rep["providers_retired"]) == 2


def test_benchmark_is_fetched_first(tmp_path):
    """Providers truncate runs; whatever is first is what survives."""
    from capitolflow import db
    from capitolflow.config import SETTINGS
    with db.session(tmp_path / "p3.db") as con:
        order = prices._ordered_tickers(con, universe_first=True)
    assert order[0] == SETTINGS.benchmark.upper()
