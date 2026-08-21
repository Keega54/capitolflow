"""Daily prices, with provider failover and loud failure.

Two lessons are baked into this module, both learned the hard way on a live run.

**Silence is the enemy.** The first deployment reported the price stage as "ok"
while fetching exactly zero rows. Everything downstream — returns, member
scores, features, the backtest, the rankings — depends on prices, so the whole
dashboard came up empty with no indication of why. A stage that fetches nothing
now reports an error and says which provider refused and what it said.

**One provider is not enough.** Stooq needs no key and is fine from a laptop,
but it hard-blocks datacenter IPs and enforces a daily request limit, which is
precisely the environment a scheduled cloud job runs in. So providers are tried
in order, a provider that starts refusing is retired for the rest of the run
rather than being asked 800 more times, and the run continues on the next one.
"""
from __future__ import annotations
import csv, io, logging, time
from datetime import date, datetime, timedelta

from ..config import SETTINGS, STOOQ_CSV, YAHOO_CHART
from ..db import upsert_many
from ..util.http import get_bytes, get_json, make_session

log = logging.getLogger(__name__)

STOOQ_OVERRIDES = {"BRK.B": "brk-b.us", "BF.B": "bf-b.us"}

# Give up on a provider after this many consecutive refusals in one run.
PROVIDER_PATIENCE = 6
# Phrases that mean "the provider is refusing us", not "this ticker is unknown".
_REFUSAL_MARKERS = ("exceeded", "limit", "denied", "forbidden", "captcha",
                    "too many requests", "unavailable")


class ProviderRefused(Exception):
    """The provider is rejecting us as a client — trying more tickers is futile."""


def _stooq_symbol(t: str) -> str:
    return STOOQ_OVERRIDES.get(t.upper(), f"{t.lower().replace('.', '-')}.us")


def fetch_stooq(session, ticker: str) -> list[dict]:
    url = STOOQ_CSV.format(sym=_stooq_symbol(ticker))
    blob = get_bytes(session, url, suffix=".csv", max_age_s=20 * 3600)
    text = blob.decode("utf8", "replace")
    head = text[:400].lower()
    if any(m in head for m in _REFUSAL_MARKERS):
        raise ProviderRefused(text[:200].strip().replace("\n", " "))
    if "date" not in text.split("\n")[0].lower():
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
    url = (YAHOO_CHART.format(sym=ticker) +
           f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit")
    try:
        js = get_json(session, url, max_age_s=20 * 3600,
                      headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
    except Exception as e:
        msg = str(e).lower()
        if any(m in msg for m in ("429", "401", "403", "too many")):
            raise ProviderRefused(str(e)[:200]) from e
        raise
    chart = (js or {}).get("chart") or {}
    if chart.get("error"):
        desc = str(chart["error"])
        if any(m in desc.lower() for m in _REFUSAL_MARKERS):
            raise ProviderRefused(desc[:200])
        return []
    res = (chart.get("result") or [None])[0]
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


def fetch_fmp(session, ticker: str) -> list[dict]:
    if not SETTINGS.fmp_key:
        raise ProviderRefused("no FMP api key configured")
    js = get_json(session,
                  f"https://financialmodelingprep.com/api/v3/historical-price-full/"
                  f"{ticker}?apikey={SETTINGS.fmp_key}", max_age_s=20 * 3600)
    hist = (js or {}).get("historical") or []
    if not hist and isinstance(js, dict) and js.get("Error Message"):
        raise ProviderRefused(str(js["Error Message"])[:200])
    return [{"ticker": ticker.upper(), "date": r["date"], "open": r.get("open"),
             "high": r.get("high"), "low": r.get("low"), "close": r.get("close"),
             "adj_close": r.get("adjClose", r.get("close")),
             "volume": r.get("volume")} for r in hist if r.get("date")]


ALL_PROVIDERS = {"stooq": fetch_stooq, "yahoo": fetch_yahoo, "fmp": fetch_fmp}


def provider_order() -> list[str]:
    """Configured provider first, then the others as fallbacks."""
    first = SETTINGS.price_provider
    rest = [p for p in ("stooq", "yahoo", "fmp") if p != first]
    return ([first] if first in ALL_PROVIDERS else []) + rest


def needed_tickers(con, min_confidence: float = 0.6) -> list[str]:
    rows = con.execute("""
        SELECT ticker, COUNT(*) n FROM transactions
        WHERE ticker IS NOT NULL AND ticker_confidence >= ?
          AND asset_type IN ('stock','fund','option')
        GROUP BY ticker ORDER BY n DESC""", (min_confidence,)).fetchall()
    return [r["ticker"] for r in rows]


def _ordered_tickers(con, universe_first: bool) -> list[str]:
    """Benchmark, then the core universe, then everything else by popularity.

    Order matters because providers rate-limit: whatever is fetched first is what
    survives a truncated run. The benchmark is non-negotiable — without it there
    are no excess returns and every downstream number is undefined.
    """
    out, seen = [], set()

    def add(t):
        t = (t or "").upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)

    add(SETTINGS.benchmark)
    if universe_first:
        try:
            from ..analytics.universe import tickers as uni_tickers
            for t in uni_tickers(con):
                add(t)
        except Exception as e:                       # universe table may not exist yet
            log.debug("universe unavailable: %s", e)
    for t in needed_tickers(con):
        add(t)
    return out


def sync_prices(con, tickers=None, *, session=None, include_benchmark: bool = True,
                universe_first: bool = True, max_tickers: int | None = None,
                refresh_days: int = 3) -> dict:
    """Fetch prices. Returns a report; a zero-row result is an error, not a success."""
    s = session or make_session()
    tickers = list(tickers) if tickers is not None else _ordered_tickers(con, universe_first)
    if include_benchmark and SETTINGS.benchmark.upper() not in {t.upper() for t in tickers}:
        tickers.insert(0, SETTINGS.benchmark)
    if max_tickers:
        tickers = tickers[:max_tickers]

    fresh_cut = (date.today() - timedelta(days=refresh_days)).isoformat()
    have = {r["ticker"]: r["mx"] for r in con.execute(
        "SELECT ticker, MAX(date) mx FROM prices GROUP BY ticker")}

    order = provider_order()
    dead: dict[str, str] = {}          # provider -> why it was retired
    misses: dict[str, int] = {p: 0 for p in order}
    by_provider: dict[str, int] = {p: 0 for p in order}
    no_data, total, attempted = [], 0, 0

    for t in tickers:
        if have.get(t) and have[t] >= fresh_cut:
            continue
        attempted += 1
        rows = []
        for p in order:
            if p in dead:
                continue
            try:
                rows = ALL_PROVIDERS[p](s, t)
            except ProviderRefused as e:
                dead[p] = str(e)
                log.warning("provider %s refused us (%s); failing over", p, e)
                continue
            except Exception as e:
                misses[p] += 1
                log.debug("%s failed for %s: %s", p, t, e)
                if misses[p] >= PROVIDER_PATIENCE and by_provider[p] == 0:
                    dead[p] = f"{misses[p]} consecutive errors, last: {e}"
                    log.warning("retiring provider %s for this run: %s", p, dead[p])
                continue
            if rows:
                misses[p] = 0
                by_provider[p] += 1
                break
        if rows:
            total += upsert_many(con, "prices", rows, mode="REPLACE")
        else:
            no_data.append(t)
        if len(dead) == len(order):
            log.error("every price provider refused; stopping early")
            break

    have_bench = con.execute(
        "SELECT COUNT(*) n FROM prices WHERE ticker=?", (SETTINGS.benchmark,)).fetchone()["n"]
    report = {
        "rows": total, "tickers_attempted": attempted,
        "tickers_with_data": attempted - len(no_data),
        "by_provider": {k: v for k, v in by_provider.items() if v},
        "providers_retired": dead,
        "benchmark": SETTINGS.benchmark, "benchmark_rows": int(have_bench),
        "no_data_sample": no_data[:15],
    }
    report["status"] = _status(report, attempted)
    if report["status"] != "ok":
        log.error("price sync problem: %s", report["status"])
    return report


def _status(report: dict, attempted: int) -> str:
    """A stage that fetched nothing must never call itself ok."""
    if attempted == 0:
        return "ok"                      # everything already fresh
    if report["benchmark_rows"] == 0:
        return (f"FAILED: no data for benchmark {report['benchmark']} — every downstream "
                f"number is undefined without it. Providers retired: "
                f"{report['providers_retired'] or 'none'}")
    if report["rows"] == 0:
        return (f"FAILED: fetched 0 rows across {attempted} tickers. Providers retired: "
                f"{report['providers_retired'] or 'none'}")
    if report["tickers_with_data"] < max(3, attempted * 0.2):
        return (f"DEGRADED: only {report['tickers_with_data']}/{attempted} tickers "
                f"returned data. Providers retired: {report['providers_retired'] or 'none'}")
    return "ok"
