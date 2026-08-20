"""Per-trade forward returns, measured from the TRADE date, not the filing date.

That distinction is the whole point of the project: a member who bought on
Jan 3 and disclosed on Feb 12 should be judged on what the stock did from Jan 3.
Returns are excess of a benchmark and signed by direction, so a well-timed sale
ahead of a drawdown scores positive just like a well-timed purchase.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from ..config import SETTINGS

log = logging.getLogger(__name__)

SCORABLE_TYPES = ("stock", "fund", "option")
SCORABLE_TXNS = ("buy", "sell", "sell_partial", "sell_full")


def load_prices(con, tickers=None) -> pd.DataFrame:
    sql = "SELECT ticker, date, adj_close FROM prices"
    params = []
    if tickers:
        sql += f" WHERE ticker IN ({','.join('?' * len(tickers))})"
        params = list(tickers)
    df = pd.read_sql_query(sql, con, params=params)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"])


def _price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide panel indexed by trading date, forward-filled onto a calendar index."""
    panel = prices.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
    full = pd.date_range(panel.index.min(), panel.index.max(), freq="D")
    return panel.reindex(full).ffill()


def _lookup(panel: pd.DataFrame, ticker: str, when: pd.Timestamp):
    """Price on/after `when` (trades settle at the next available close)."""
    if ticker not in panel.columns:
        return np.nan
    col = panel[ticker]
    idx = col.index.searchsorted(when, side="left")
    if idx >= len(col):
        return np.nan
    val = col.iloc[idx]
    return val if pd.notna(val) else np.nan


def compute_trade_returns(con, horizons=None, min_confidence: float = 0.7) -> pd.DataFrame:
    horizons = list(horizons or SETTINGS.horizons)
    bench = SETTINGS.benchmark

    txns = pd.read_sql_query(f"""
        SELECT txn_id, member_id, ticker, transaction_date, direction, amount_est, asset_type, txn_type
        FROM transactions
        WHERE ticker IS NOT NULL AND ticker_confidence >= {min_confidence}
          AND direction != 0
          AND asset_type IN {SCORABLE_TYPES}
          AND txn_type IN {SCORABLE_TXNS}
          AND transaction_date IS NOT NULL
    """, con)
    if txns.empty:
        return pd.DataFrame()
    txns["transaction_date"] = pd.to_datetime(txns["transaction_date"], errors="coerce")
    txns = txns.dropna(subset=["transaction_date"])

    need = sorted(set(txns["ticker"]) | {bench})
    prices = load_prices(con, need)
    if prices.empty:
        log.warning("no price data loaded; run `capitolflow prices` first")
        return pd.DataFrame()
    panel = _price_panel(prices)
    if bench not in panel.columns:
        log.warning("benchmark %s missing from prices; excess returns unavailable", bench)
        return pd.DataFrame()

    out = []
    for row in txns.itertuples(index=False):
        t0 = row.transaction_date
        p0 = _lookup(panel, row.ticker, t0)
        b0 = _lookup(panel, bench, t0)
        if not np.isfinite(p0) or not np.isfinite(b0) or p0 <= 0 or b0 <= 0:
            continue
        for h in horizons:
            t1 = t0 + pd.Timedelta(days=h)
            if t1 > panel.index[-1]:
                continue                      # horizon not yet complete: skip, never impute
            p1 = _lookup(panel, row.ticker, t1)
            b1 = _lookup(panel, bench, t1)
            if not np.isfinite(p1) or not np.isfinite(b1):
                continue
            ar = p1 / p0 - 1.0
            br = b1 / b0 - 1.0
            out.append({
                "txn_id": row.txn_id, "horizon_days": h,
                "asset_return": float(ar), "bench_return": float(br),
                "excess_return": float((ar - br) * row.direction),
            })
    return pd.DataFrame(out)


def store_trade_returns(con, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    rows = df.to_dict("records")
    con.executemany(
        "INSERT OR REPLACE INTO trade_returns (txn_id, horizon_days, asset_return, bench_return,"
        " excess_return, computed_at) VALUES (?,?,?,?,?,datetime('now'))",
        [(r["txn_id"], r["horizon_days"], r["asset_return"], r["bench_return"],
          r["excess_return"]) for r in rows])
    return len(rows)
