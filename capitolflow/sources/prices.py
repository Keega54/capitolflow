"""Daily OHLC prices. Stooq needs no key and is the default; Yahoo and FMP are
drop-in alternates. Prices are cached in SQLite so a re-run costs nothing."""
from __future__ import annotations
import csv, io, logging, time
from datetime import date, datetime, timedelta

from ..config import SETTINGS, STOOQ_CSV, YAHOO_CHART
from ..db import upsert_many
from ..util.http import get_bytes, get_json, make_session

log = logging.getLogger(__name__)

# Tickers that need a suffix or a different symbol on a given provider.
STOOQ_OVERRIDES = {"BRK.B": "brk-b.us", "BF.B": "bf-b.us"}


def _stooq_symbol(t: str) -> str:
    return STOOQ_OVERRIDES.get(t.upper(), f"{t.lower().replace('.', '-')}.us")


def fetch_stooq(session, ticker: str) -> list[dict]:
    url = STOOQ_CSV.format(sym=_stooq_symbol(ticker))
    blob = get_bytes(session, url, suffix=".csv", max_age_s=20 * 3600)
    text = blob.decode("utf8", "replace")
    if "Date" not in text.split("\n")[0]:
        return []
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        try:
            out.append({"ticker": ticker.upper(), "date": r["Date"],
                        "open": float(r["Open"]), "high": float(r["High"]),
                        "low": float(r["Low"]), "close": float(r["Close"]),
                        "adj_close": float(r["Close"]),
                        "volume": float(r.get("Volume") or 0)})
        except (ValueError, KeyError, TypeError):
            continue
    return out


def fetch_yahoo(session, ticker: str, start: str = "2010-01-01") -> list[dict]:
    p1 = int(datetime.fromisoformat(start).timestamp())
    p2 = int(time.time())
    url = YAHOO_CHART.format(sym=ticker) + f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"
    js = get_json(session, url, max_age_s=20 * 3600)
    res = (js.get("chart", {}).get("result") or [None])[0]
    if not res:
        return []
    ts = res.get("timestamp") or []
    q = (res.get("indicators", {}).get("quote") or [{}])[0]
    adj = ((res.get("indicators", {}).get("adjclose") or [{}])[0] or {}).get("adjclose") or []
    out = []
    for i, t in enumerate(ts):
        c = (q.get("close") or [None] * len(ts))[i]
        if c is None:
            continue
        out.append({
            "ticker": ticker.upper(),
            "date": datetime.utcfromtimestamp(t).date().isoformat(),
            "open": (q.get("open") or [None])[i], "high": (q.get("high") or [None])[i],
            "low": (q.get("low") or [None])[i], "close": c,
            "adj_close": adj[i] if i < len(adj) and adj[i] is not None else c,
            "volume": (q.get("volume") or [None])[i],
        })
    return out


def needed_tickers(con, min_confidence: float = 0.6) -> list[str]:
    rows = con.execute("""
        SELECT ticker, COUNT(*) n FROM transactions
        WHERE ticker IS NOT NULL AND ticker_confidence >= ?
          AND asset_type IN ('stock','fund','option')
        GROUP BY ticker ORDER BY n DESC""", (min_confidence,)).fetchall()
    return [r["ticker"] for r in rows]


def sync_prices(con, tickers=None, *, session=None, include_benchmark: bool = True,
                universe_first: bool = True,
                max_tickers: int | None = None, refresh_days: int = 3) -> int:
    s = session or make_session()
    tickers = list(tickers or needed_tickers(con))
    if include_benchmark and SETTINGS.benchmark not in tickers:
        tickers.insert(0, SETTINGS.benchmark)
    if max_tickers:
        tickers = tickers[:max_tickers]

    fresh_cut = (date.today() - timedelta(days=refresh_days)).isoformat()
    have = {r["ticker"]: r["mx"] for r in con.execute(
        "SELECT ticker, MAX(date) mx FROM prices GROUP BY ticker")}

    fetch = fetch_yahoo if SETTINGS.price_provider == "yahoo" else fetch_stooq
    total = 0
    for t in tickers:
        if have.get(t) and have[t] >= fresh_cut:
            continue
        try:
            rows = fetch(s, t)
        except Exception as e:
            log.warning("price fetch failed for %s: %s", t, e)
            continue
        if not rows:
            log.info("no price data for %s", t)
            continue
        total += upsert_many(con, "prices", rows, mode="REPLACE")
    return total
