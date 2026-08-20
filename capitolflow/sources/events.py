"""Current-events regime signals from GDELT.

Answers the question "there's a war on — does that change what to hold?" in the
only way this data honestly can: by measuring how loudly the world's news is
talking about a theme each day, standardizing it against its own recent history,
and letting the model discover which sectors respond. It does NOT read the news
and form an opinion; it measures attention intensity and lets the backtest decide
whether that attention has ever predicted anything.

GDELT's DOC 2.0 API is free, needs no key, and covers global news back to 2017.
Themes are deliberately few and broad — a hundred narrow queries would be a
multiple-testing machine, and every extra theme is another chance to find a
spurious correlation.
"""
from __future__ import annotations
import logging
from datetime import date, datetime, timedelta

import numpy as np

from ..db import upsert_many
from ..util.http import get_json, make_session

log = logging.getLogger(__name__)

GDELT_TIMELINE = ("https://api.gdeltproject.org/api/v2/doc/doc"
                  "?query={q}&mode=timelinevol&startdatetime={start}&enddatetime={end}"
                  "&format=json")

# Broad, durable themes. Each maps to sectors the backtest can test against.
THEMES = {
    "conflict": '("war" OR "military strike" OR "invasion" OR "armed conflict") sourcelang:eng',
    "tariffs": '("tariff" OR "trade war" OR "export controls" OR "sanctions") sourcelang:eng',
    "energy": '("oil price" OR "opec" OR "energy crisis" OR "natural gas prices") sourcelang:eng',
    "health": '("outbreak" OR "pandemic" OR "vaccine" OR "fda approval") sourcelang:eng',
    "ai": '("artificial intelligence" OR "semiconductor" OR "chip shortage") sourcelang:eng',
    "monetary": '("federal reserve" OR "interest rate" OR "inflation report") sourcelang:eng',
}

# Which sectors each theme is hypothesized to move. The SIGN is not asserted —
# the backtest estimates it. Listing the pairing only limits the search space.
THEME_SECTORS = {
    "conflict": ["Industrials", "Energy", "Aerospace & Defense"],
    "tariffs": ["Industrials", "Consumer Cyclical", "Technology"],
    "energy": ["Energy", "Utilities", "Industrials"],
    "health": ["Healthcare"],
    "ai": ["Technology", "Communication Services"],
    "monetary": ["Financial Services", "Real Estate", "Utilities"],
}

ZWINDOW = 180  # trailing days for standardization


def fetch_theme(session, theme: str, query: str, start: date, end: date) -> list[dict]:
    url = GDELT_TIMELINE.format(
        q=_quote(query),
        start=start.strftime("%Y%m%d%H%M%S"), end=end.strftime("%Y%m%d%H%M%S"))
    js = get_json(session, url, max_age_s=12 * 3600)
    series = (js or {}).get("timeline") or []
    if not series:
        return []
    pts = series[0].get("data") or []
    out = []
    for p in pts:
        raw = p.get("date")
        if not raw:
            continue
        try:
            d = datetime.strptime(str(raw)[:8], "%Y%m%d").date()
        except ValueError:
            continue
        out.append({"theme": theme, "date": d.isoformat(),
                    "intensity": float(p.get("value") or 0.0),
                    "z_score": None, "source": "gdelt"})
    return out


def _quote(q: str) -> str:
    from urllib.parse import quote
    return quote(q, safe="")


def sync(con, *, session=None, start: date | None = None, end: date | None = None,
         chunk_days: int = 365) -> int:
    """GDELT caps each request's span, so long histories are fetched in chunks."""
    s = session or make_session()
    end = end or date.today()
    start = start or (end - timedelta(days=365 * 5))
    total = 0
    for theme, query in THEMES.items():
        cur = start
        rows = []
        while cur < end:
            stop = min(cur + timedelta(days=chunk_days), end)
            try:
                rows += fetch_theme(s, theme, query, cur, stop)
            except Exception as e:
                log.warning("gdelt %s %s..%s failed: %s", theme, cur, stop, e)
            cur = stop
        if rows:
            total += upsert_many(con, "event_index", rows, mode="REPLACE")
    if total:
        compute_zscores(con)
    return total


def compute_zscores(con, window: int = ZWINDOW) -> int:
    """Standardize each theme against its own trailing history.

    Raw article counts trend upward as GDELT indexes more sources, so an
    un-normalized series would encode "later date" rather than "louder news".
    The trailing z-score is what makes it comparable across years, and it is
    computed strictly from the past so it stays point-in-time correct.
    """
    import pandas as pd
    df = pd.read_sql_query("SELECT theme, date, intensity FROM event_index", con)
    if df.empty:
        return 0
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["theme", "date"])
    out = []
    for theme, g in df.groupby("theme"):
        s = g.set_index("date")["intensity"].astype(float)
        mu = s.rolling(window, min_periods=30).mean().shift(1)
        sd = s.rolling(window, min_periods=30).std(ddof=1).shift(1)
        z = ((s - mu) / sd).replace([np.inf, -np.inf], np.nan)
        for d, v in z.items():
            if pd.notna(v):
                out.append({"theme": theme, "date": d.date().isoformat(),
                            "z": float(np.clip(v, -6, 6))})
    if not out:
        return 0
    con.executemany("UPDATE event_index SET z_score=? WHERE theme=? AND date=?",
                    [(r["z"], r["theme"], r["date"]) for r in out])
    return len(out)


def sync_sectors(con, tickers=None, *, session=None) -> int:
    """Sector labels, needed to connect a theme to the stocks it should move."""
    from ..config import SETTINGS
    s = session or make_session()
    if tickers is None:
        tickers = [r["ticker"] for r in con.execute(
            "SELECT DISTINCT ticker FROM transactions WHERE ticker IS NOT NULL "
            "AND ticker_confidence >= 0.7")]
    have = {r["ticker"] for r in con.execute("SELECT ticker FROM ticker_sectors")}
    todo = [t for t in tickers if t not in have]
    if not todo or not SETTINGS.fmp_key:
        if todo:
            log.info("no sector provider configured; %d tickers unlabelled", len(todo))
        return 0
    rows = []
    for t in todo:
        try:
            js = get_json(s, f"https://financialmodelingprep.com/api/v3/profile/{t}"
                             f"?apikey={SETTINGS.fmp_key}", max_age_s=30 * 86400)
        except Exception as e:
            log.debug("sector lookup failed for %s: %s", t, e)
            continue
        if js:
            rows.append({"ticker": t, "sector": js[0].get("sector"),
                         "industry": js[0].get("industry"), "source": "fmp"})
    return upsert_many(con, "ticker_sectors", rows, mode="REPLACE")


def theme_panel(con):
    """Wide frame: index=date, columns=theme, values=z_score."""
    import pandas as pd
    df = pd.read_sql_query(
        "SELECT theme, date, z_score FROM event_index WHERE z_score IS NOT NULL", con)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="theme", values="z_score", aggfunc="last")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    return wide.reindex(full).ffill()
