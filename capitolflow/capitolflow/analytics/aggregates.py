"""Leaderboards and time series, all keyed on trade date."""
from __future__ import annotations
import pandas as pd

CONF = 0.6


def _where(start=None, end=None, chamber=None, party=None, owner=None,
           asset_types=("stock", "fund", "option"), min_conf=CONF):
    w, p = ["t.ticker IS NOT NULL", f"t.ticker_confidence >= {float(min_conf)}"], []
    if start:
        w.append("t.transaction_date >= ?"); p.append(start)
    if end:
        w.append("t.transaction_date <= ?"); p.append(end)
    if chamber:
        w.append("m.chamber = ?"); p.append(chamber)
    if party:
        w.append("m.party = ?"); p.append(party)
    if owner:
        w.append("t.owner = ?"); p.append(owner)
    if asset_types:
        w.append(f"t.asset_type IN ({','.join('?' * len(asset_types))})"); p += list(asset_types)
    return " AND ".join(w), p


def ticker_leaderboard(con, *, horizon: int = 90, limit: int = 50, **kw) -> pd.DataFrame:
    """Per-ticker: how many distinct members traded it, and how much money moved.

    `weighted_net_flow` scales each member's dollars by their accuracy weight, so
    a ticker bought mostly by consistently well-timed traders outranks one bought
    by the same dollars from members whose timing is average.
    """
    where, params = _where(**kw)
    sql = f"""
    SELECT t.ticker,
           COUNT(DISTINCT t.member_id)                                      AS n_members,
           COUNT(*)                                                         AS n_trades,
           SUM(CASE WHEN t.direction > 0 THEN 1 ELSE 0 END)                 AS n_buys,
           SUM(CASE WHEN t.direction < 0 THEN 1 ELSE 0 END)                 AS n_sells,
           COUNT(DISTINCT CASE WHEN t.direction > 0 THEN t.member_id END)   AS n_members_buying,
           COUNT(DISTINCT CASE WHEN t.direction < 0 THEN t.member_id END)   AS n_members_selling,
           SUM(COALESCE(t.amount_est,0))                                    AS gross_volume,
           SUM(COALESCE(t.amount_est,0) * t.direction)                      AS net_flow,
           SUM(COALESCE(t.amount_est,0) * t.direction * COALESCE(s.weight,1.0)) AS weighted_net_flow,
           SUM(COALESCE(t.amount_low,0))                                    AS volume_low,
           SUM(COALESCE(t.amount_high,0))                                   AS volume_high,
           AVG(t.filing_delay_days)                                         AS avg_filing_delay,
           MIN(t.transaction_date)                                          AS first_trade,
           MAX(t.transaction_date)                                          AS last_trade
    FROM transactions t
    LEFT JOIN members m       ON m.member_id = t.member_id
    LEFT JOIN member_scores s ON s.member_id = t.member_id AND s.horizon_days = ?
    WHERE {where}
    GROUP BY t.ticker
    ORDER BY n_members DESC, gross_volume DESC
    LIMIT ?
    """
    return pd.read_sql_query(sql, con, params=[horizon] + params + [limit])


def member_leaderboard(con, *, horizon: int = 90, limit: int = 100, **kw) -> pd.DataFrame:
    where, params = _where(**kw)
    sql = f"""
    SELECT t.member_id, m.full_name, m.chamber, m.party, m.state, m.role_title,
           COUNT(*)                                   AS n_trades,
           COUNT(DISTINCT t.ticker)                   AS n_tickers,
           SUM(COALESCE(t.amount_est,0))              AS gross_volume,
           SUM(COALESCE(t.amount_est,0)*t.direction)  AS net_flow,
           AVG(t.filing_delay_days)                   AS avg_filing_delay,
           MAX(t.filing_delay_days)                   AS max_filing_delay,
           SUM(CASE WHEN t.filing_delay_days > 45 THEN 1 ELSE 0 END) AS n_late,
           s.hit_rate, s.mean_excess, s.shrunk_excess, s.weight, s.n_scored
    FROM transactions t
    LEFT JOIN members m       ON m.member_id = t.member_id
    LEFT JOIN member_scores s ON s.member_id = t.member_id AND s.horizon_days = ?
    WHERE {where} AND t.member_id IS NOT NULL
    GROUP BY t.member_id
    ORDER BY gross_volume DESC
    LIMIT ?
    """
    return pd.read_sql_query(sql, con, params=[horizon] + params + [limit])


def flow_timeseries(con, *, freq: str = "M", ticker: str | None = None, **kw) -> pd.DataFrame:
    where, params = _where(**kw)
    if ticker:
        where += " AND t.ticker = ?"
        params.append(ticker)
    fmt = {"D": "%Y-%m-%d", "W": "%Y-%W", "M": "%Y-%m", "Q": "%Y-%m", "Y": "%Y"}.get(freq, "%Y-%m")
    sql = f"""
    SELECT strftime('{fmt}', t.transaction_date) AS period,
           COUNT(*) AS n_trades,
           COUNT(DISTINCT t.member_id) AS n_members,
           SUM(CASE WHEN t.direction>0 THEN COALESCE(t.amount_est,0) ELSE 0 END) AS buy_volume,
           SUM(CASE WHEN t.direction<0 THEN COALESCE(t.amount_est,0) ELSE 0 END) AS sell_volume,
           SUM(COALESCE(t.amount_est,0)*t.direction) AS net_flow
    FROM transactions t LEFT JOIN members m ON m.member_id = t.member_id
    WHERE {where}
    GROUP BY period ORDER BY period
    """
    return pd.read_sql_query(sql, con, params=params)


def member_timeline(con, member_id: str) -> pd.DataFrame:
    return pd.read_sql_query("""
        SELECT t.transaction_date, t.filed_date, t.filing_delay_days, t.ticker,
               t.asset_name_raw, t.txn_type, t.direction, t.amount_low, t.amount_high,
               t.amount_est, t.owner, f.url,
               MAX(CASE WHEN r.horizon_days=90 THEN r.excess_return END) AS excess_90d
        FROM transactions t
        LEFT JOIN filings f ON f.filing_id = t.filing_id
        LEFT JOIN trade_returns r ON r.txn_id = t.txn_id
        WHERE t.member_id = ?
        GROUP BY t.txn_id
        ORDER BY t.transaction_date DESC""", con, params=[member_id])


def disclosure_delay_stats(con, **kw) -> pd.DataFrame:
    """How long the public actually waited to learn about each trade."""
    where, params = _where(**kw)
    return pd.read_sql_query(f"""
        SELECT m.chamber, m.party,
               COUNT(*) AS n,
               AVG(t.filing_delay_days) AS mean_delay,
               MIN(t.filing_delay_days) AS min_delay,
               MAX(t.filing_delay_days) AS max_delay,
               SUM(CASE WHEN t.filing_delay_days > 45 THEN 1 ELSE 0 END)*1.0/COUNT(*) AS pct_late
        FROM transactions t JOIN members m ON m.member_id=t.member_id
        WHERE {where} AND t.filing_delay_days IS NOT NULL
        GROUP BY m.chamber, m.party""", con, params=params)


def cluster_detector(con, *, window_days: int = 14, min_members: int = 4, horizon: int = 90,
                     limit: int = 40, **kw) -> pd.DataFrame:
    """Find tickers where an unusual number of distinct members traded the same
    way inside a short window — the pattern most worth a human look."""
    where, params = _where(**kw)
    df = pd.read_sql_query(f"""
        SELECT t.ticker, t.member_id, t.transaction_date, t.direction,
               COALESCE(t.amount_est,0) AS amount_est, COALESCE(s.weight,1.0) AS weight
        FROM transactions t
        LEFT JOIN members m ON m.member_id=t.member_id
        LEFT JOIN member_scores s ON s.member_id=t.member_id AND s.horizon_days=?
        WHERE {where} AND t.member_id IS NOT NULL AND t.direction != 0
    """, con, params=[horizon] + params)
    if df.empty:
        return df
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    out = []
    for (tic, side), grp in df.groupby(["ticker", df["direction"] > 0]):
        grp = grp.sort_values("transaction_date")
        for _, row in grp.iterrows():
            lo = row["transaction_date"]
            win = grp[(grp["transaction_date"] >= lo) &
                      (grp["transaction_date"] <= lo + pd.Timedelta(days=window_days))]
            nm = win["member_id"].nunique()
            if nm >= min_members:
                out.append({
                    "ticker": tic, "side": "buy" if side else "sell",
                    "window_start": lo.date().isoformat(),
                    "window_end": (lo + pd.Timedelta(days=window_days)).date().isoformat(),
                    "n_members": int(nm), "n_trades": int(len(win)),
                    "volume": float(win["amount_est"].sum()),
                    "weighted_volume": float((win["amount_est"] * win["weight"]).sum()),
                })
    if not out:
        return pd.DataFrame()
    res = pd.DataFrame(out).sort_values(["n_members", "weighted_volume"], ascending=False)
    # collapse overlapping windows for the same ticker/side
    res = res.drop_duplicates(subset=["ticker", "side", "n_members"], keep="first")
    return res.head(limit).reset_index(drop=True)
