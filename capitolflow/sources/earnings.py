"""Earnings calendar and surprise history.

Earnings are the single biggest scheduled source of single-stock variance, so a
model that ignores them will mistake earnings drift for whatever factor happened
to be correlated with it. Three providers, tried in order of data quality; each
is optional and the pipeline degrades to "no earnings features" rather than
failing, so the project still runs with zero API keys.
"""
from __future__ import annotations
import logging
from datetime import date, timedelta

from ..config import SETTINGS
from ..db import upsert_many
from ..util.dates import iso
from ..util.http import get_json, make_session

log = logging.getLogger(__name__)

FMP_HIST = "https://financialmodelingprep.com/api/v3/historical/earning_calendar/{sym}?apikey={k}"
FINNHUB_HIST = "https://finnhub.io/api/v1/stock/earnings?symbol={sym}&token={k}"
NASDAQ_SURPRISE = "https://api.nasdaq.com/api/company/{sym}/earnings-surprise"

_NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0", "Accept": "application/json",
    "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/",
}


def _pct(actual, est):
    try:
        a, e = float(actual), float(est)
    except (TypeError, ValueError):
        return None
    if e == 0:
        return None
    return (a - e) / abs(e)


def _from_fmp(s, ticker: str) -> list[dict]:
    rows = get_json(s, FMP_HIST.format(sym=ticker, k=SETTINGS.fmp_key), max_age_s=7 * 86400)
    out = []
    for r in rows or []:
        d = iso(r.get("date"))
        if not d:
            continue
        out.append({
            "ticker": ticker, "report_date": d, "fiscal_period": r.get("fiscalDateEnding"),
            "eps_actual": r.get("eps"), "eps_estimate": r.get("epsEstimated"),
            "surprise_pct": _pct(r.get("eps"), r.get("epsEstimated")),
            "revenue_actual": r.get("revenue"), "revenue_estimate": r.get("revenueEstimated"),
            "rev_surprise_pct": _pct(r.get("revenue"), r.get("revenueEstimated")),
            "time_of_day": (r.get("time") or "unknown"), "source": "fmp",
        })
    return out


def _from_finnhub(s, ticker: str) -> list[dict]:
    rows = get_json(s, FINNHUB_HIST.format(sym=ticker, k=SETTINGS.finnhub_key),
                    max_age_s=7 * 86400)
    out = []
    for r in rows or []:
        d = iso(r.get("period"))
        if not d:
            continue
        out.append({
            "ticker": ticker, "report_date": d, "fiscal_period": r.get("period"),
            "eps_actual": r.get("actual"), "eps_estimate": r.get("estimate"),
            "surprise_pct": (r.get("surprisePercent") / 100.0
                             if r.get("surprisePercent") is not None
                             else _pct(r.get("actual"), r.get("estimate"))),
            "revenue_actual": None, "revenue_estimate": None, "rev_surprise_pct": None,
            "time_of_day": "unknown", "source": "finnhub",
        })
    return out


def _from_nasdaq(s, ticker: str) -> list[dict]:
    js = get_json(s, NASDAQ_SURPRISE.format(sym=ticker.lower()), max_age_s=7 * 86400,
                  headers=_NASDAQ_HEADERS)
    rows = (((js or {}).get("data") or {}).get("earningsSurpriseTable") or {}).get("rows") or []
    out = []
    for r in rows:
        d = iso(r.get("dateReported"))
        if not d:
            continue
        def num(x):
            try:
                return float(str(x).replace("$", "").replace(",", "").replace("%", ""))
            except (TypeError, ValueError):
                return None
        out.append({
            "ticker": ticker, "report_date": d, "fiscal_period": r.get("fiscalQtrEnding"),
            "eps_actual": num(r.get("eps")), "eps_estimate": num(r.get("consensusForecast")),
            "surprise_pct": _pct(num(r.get("eps")), num(r.get("consensusForecast"))),
            "revenue_actual": None, "revenue_estimate": None, "rev_surprise_pct": None,
            "time_of_day": "unknown", "source": "nasdaq",
        })
    return out


def sync(con, tickers=None, *, session=None, max_tickers: int | None = None,
         refresh_days: int = 20) -> int:
    """Fetch earnings history for the tickers politicians actually trade."""
    s = session or make_session()
    if tickers is None:
        tickers = [r["ticker"] for r in con.execute("""
            SELECT ticker, COUNT(*) n FROM transactions
            WHERE ticker IS NOT NULL AND ticker_confidence >= 0.7
              AND asset_type IN ('stock','option')
            GROUP BY ticker ORDER BY n DESC""")]
    if max_tickers:
        tickers = tickers[:max_tickers]
    if not tickers:
        return 0

    providers = []
    if SETTINGS.fmp_key:
        providers.append(("fmp", _from_fmp))
    if SETTINGS.finnhub_key:
        providers.append(("finnhub", _from_finnhub))
    providers.append(("nasdaq", _from_nasdaq))

    fresh_cut = (date.today() - timedelta(days=refresh_days)).isoformat()
    have = {r["ticker"]: r["mx"] for r in con.execute(
        "SELECT ticker, MAX(report_date) mx FROM earnings GROUP BY ticker")}

    total = 0
    for t in tickers:
        if have.get(t) and have[t] >= fresh_cut:
            continue
        for name, fn in providers:
            try:
                rows = fn(s, t)
            except Exception as e:
                log.debug("earnings %s via %s failed: %s", t, name, e)
                continue
            if rows:
                total += upsert_many(con, "earnings", rows, mode="REPLACE")
                break
        else:
            log.info("no earnings data for %s from any provider", t)
    return total


def features(con) -> "pd.DataFrame":
    """Long frame of (ticker, report_date, surprise) for the feature builder."""
    import pandas as pd
    df = pd.read_sql_query(
        "SELECT ticker, report_date, eps_actual, eps_estimate, surprise_pct, rev_surprise_pct "
        "FROM earnings WHERE report_date IS NOT NULL", con)
    if df.empty:
        return df
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    return df.dropna(subset=["report_date"]).sort_values(["ticker", "report_date"])
