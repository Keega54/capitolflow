"""The core universe: the names this project guarantees fresh data for.

Politicians have touched thousands of tickers, most of them once. Fetching daily
prices for all of them is slow, mostly wasted, and makes the feature panel
expensive to build. The core universe is the subset that actually carries the
signal — ranked by how many distinct members traded it, how often, and for how
much — and it is refreshed on every run so a name that becomes newly popular
enters automatically and a name that goes quiet ages out.

Ranking by distinct members rather than raw trade count matters: one member
rebalancing a position forty times is not the same evidence as forty members
independently buying, and a raw count would rank them identically.
"""
from __future__ import annotations
import logging
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_SIZE = 50
RECENCY_WINDOW_DAYS = 730      # only count activity from the last two years


def compute(con, size: int = DEFAULT_SIZE, window_days: int = RECENCY_WINDOW_DAYS) -> pd.DataFrame:
    cutoff = (pd.Timestamp.today() - pd.Timedelta(days=window_days)).date().isoformat()
    df = pd.read_sql_query("""
        SELECT ticker,
               COUNT(*)                      AS n_trades,
               COUNT(DISTINCT member_id)     AS n_members,
               SUM(COALESCE(amount_est,0))   AS gross_amount,
               MAX(transaction_date)         AS last_traded
        FROM transactions
        WHERE ticker IS NOT NULL AND ticker_confidence >= 0.7
          AND asset_type IN ('stock','fund','option')
          AND transaction_date >= ?
        GROUP BY ticker""", con, params=[cutoff])
    if df.empty:
        return df

    # Rank on a blend so no single dimension dominates. Members carry the most
    # weight because independent participants are the actual evidence.
    for c, w in (("n_members", 0.5), ("n_trades", 0.25), ("gross_amount", 0.25)):
        r = df[c].rank(pct=True)
        df[f"_{c}"] = r * w
    df["_score"] = df[[c for c in df.columns if c.startswith("_")]].sum(axis=1)
    df = df.sort_values("_score", ascending=False).head(size).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["added_on"] = date.today().isoformat()
    df["reason"] = df.apply(
        lambda r: f"{int(r.n_members)} members, {int(r.n_trades)} trades", axis=1)
    return df[["ticker", "rank", "n_trades", "n_members", "gross_amount",
               "last_traded", "added_on", "reason"]]


def store(con, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    # Preserve the original added_on for names already in the universe, so the
    # dashboard can show how long a company has been on the list.
    existing = {r["ticker"]: r["added_on"] for r in
                con.execute("SELECT ticker, added_on FROM core_universe")}
    rows = []
    for r in df.to_dict("records"):
        r["added_on"] = existing.get(r["ticker"], r["added_on"])
        rows.append(r)
    con.execute("DELETE FROM core_universe")
    from ..db import upsert_many
    return upsert_many(con, "core_universe", rows, mode="REPLACE")


def tickers(con, fallback_limit: int = 300) -> list[str]:
    """Universe tickers, or the most-traded names if the universe isn't built."""
    rows = [r["ticker"] for r in
            con.execute("SELECT ticker FROM core_universe ORDER BY rank")]
    if rows:
        return rows
    return [r["ticker"] for r in con.execute("""
        SELECT ticker, COUNT(*) n FROM transactions
        WHERE ticker IS NOT NULL AND ticker_confidence >= 0.7
        GROUP BY ticker ORDER BY n DESC LIMIT ?""", (fallback_limit,))]


def refresh(con, size: int = DEFAULT_SIZE) -> dict:
    df = compute(con, size=size)
    n = store(con, df)
    return {"universe_size": n,
            "tickers": df["ticker"].tolist() if not df.empty else []}
