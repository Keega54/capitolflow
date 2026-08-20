"""Shared query layer. The live API and the static export both call these, so a
GitHub Pages build and a locally served dashboard show identical numbers."""
from __future__ import annotations
import json
import pandas as pd

from ..analytics import aggregates, lobbying_join
from ..pipeline import health


def _recs(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


def summary(con) -> dict:
    h = health(con)
    row = con.execute("""
        SELECT COUNT(*) n, SUM(COALESCE(amount_est,0)) vol,
               COUNT(DISTINCT member_id) nm, COUNT(DISTINCT ticker) nt
        FROM transactions WHERE ticker IS NOT NULL""").fetchone()
    delay = con.execute("""
        SELECT AVG(filing_delay_days) mean_delay,
               SUM(CASE WHEN filing_delay_days > 45 THEN 1 ELSE 0 END)*1.0/COUNT(*) pct_late
        FROM transactions WHERE filing_delay_days IS NOT NULL""").fetchone()
    med = con.execute("""
        SELECT filing_delay_days FROM transactions WHERE filing_delay_days IS NOT NULL
        ORDER BY filing_delay_days
        LIMIT 1 OFFSET (SELECT COUNT(*)/2 FROM transactions WHERE filing_delay_days IS NOT NULL)
    """).fetchone()
    last = con.execute("SELECT source, MAX(finished_at) t, status FROM ingest_runs "
                       "GROUP BY source ORDER BY t DESC LIMIT 8").fetchall()
    return {
        "counts": h,
        "n_transactions": row["n"] or 0,
        "gross_volume": float(row["vol"] or 0),
        "n_members": row["nm"] or 0,
        "n_tickers": row["nt"] or 0,
        "mean_disclosure_delay": float(delay["mean_delay"] or 0),
        "median_disclosure_delay": (med["filing_delay_days"] if med else None),
        "pct_filed_late": float(delay["pct_late"] or 0),
        "last_runs": [dict(r) for r in last],
    }


def tickers(con, limit=40, **kw):
    return _recs(aggregates.ticker_leaderboard(con, limit=limit, **kw))


def members_view(con, limit=100, **kw):
    return _recs(aggregates.member_leaderboard(con, limit=limit, **kw))


def timeseries(con, freq="M", ticker=None, **kw):
    return _recs(aggregates.flow_timeseries(con, freq=freq, ticker=ticker, **kw))


def delays(con):
    hist = pd.read_sql_query("""
        SELECT CASE
                 WHEN filing_delay_days < 0  THEN '<0'
                 WHEN filing_delay_days <= 15 THEN '0-15'
                 WHEN filing_delay_days <= 30 THEN '16-30'
                 WHEN filing_delay_days <= 45 THEN '31-45'
                 WHEN filing_delay_days <= 90 THEN '46-90'
                 WHEN filing_delay_days <= 180 THEN '91-180'
                 ELSE '180+' END AS bucket,
               COUNT(*) AS n
        FROM transactions WHERE filing_delay_days IS NOT NULL
        GROUP BY bucket""", con)
    order = ['<0', '0-15', '16-30', '31-45', '46-90', '91-180', '180+']
    hist["sort"] = hist["bucket"].map({b: i for i, b in enumerate(order)})
    hist = hist.sort_values("sort").drop(columns="sort")
    return {"histogram": _recs(hist), "by_group": _recs(aggregates.disclosure_delay_stats(con))}


def clusters(con, window_days=14, min_members=4, limit=30):
    return _recs(aggregates.cluster_detector(con, window_days=window_days,
                                             min_members=min_members, limit=limit))


def lobbying(con):
    ov = lobbying_join.overlay(con)
    top = pd.DataFrame()
    if not ov.empty:
        top = (ov.groupby("ticker")
                 .agg(lobby_spend=("lobby_spend", "sum"), n_members=("n_members", "sum"),
                      gross_volume=("gross_volume", "sum"),
                      weighted_net_flow=("weighted_net_flow", "sum"))
                 .reset_index().sort_values("lobby_spend", ascending=False).head(40))
    try:
        corr = lobbying_join.correlation_report(con)
    except Exception:
        corr = pd.DataFrame()
    return {"by_ticker": _recs(top), "correlation": _recs(corr),
            "committee_alignment": _recs(lobbying_join.committee_alignment(con, limit=30))}


def member_detail(con, member_id: str):
    m = con.execute("SELECT * FROM members WHERE member_id=?", (member_id,)).fetchone()
    scores = con.execute("SELECT * FROM member_scores WHERE member_id=? ORDER BY horizon_days",
                         (member_id,)).fetchall()
    return {
        "member": dict(m) if m else None,
        "scores": [dict(r) for r in scores],
        "trades": _recs(aggregates.member_timeline(con, member_id)),
    }


def predictions(con):
    """Latest ranked picks plus everything needed to judge how much to trust them."""
    from ..db import get_kv
    bt = get_kv(con, "last_backtest") or {}
    as_of = con.execute("SELECT MAX(as_of) a FROM predictions").fetchone()
    as_of = as_of["a"] if as_of else None
    out = {"as_of": as_of, "horizons": {}, "backtest": {}, "timing": get_kv(con, "signal_decay"),
           "timing_summary": get_kv(con, "timing_summary")}
    if not as_of:
        return out
    for label, h in (("short_term", 21), ("long_term", 126)):
        rows = _recs(pd.read_sql_query(
            "SELECT ticker, rank, score, score_pctile, expected_excess, confidence,"
            " attribution, rationale FROM predictions WHERE as_of=? AND horizon_days=?"
            " ORDER BY rank", con, params=[as_of, h]))
        for r in rows:
            try:
                r["attribution"] = json.loads(r["attribution"]) if r["attribution"] else {}
            except Exception:
                r["attribution"] = {}
        w = _recs(pd.read_sql_query(
            "SELECT factor, weight, raw_ic, stability FROM factor_weights WHERE horizon_days=?"
            " AND as_of=(SELECT MAX(as_of) FROM factor_weights WHERE horizon_days=?)",
            con, params=[h, h]))
        bh = (bt.get("horizons") or {}).get(str(h)) or (bt.get("horizons") or {}).get(h) or {}
        out["horizons"][label] = {
            "horizon_days": h, "picks": rows, "weights": w,
            "mean_ic": bh.get("mean_ic"), "null_ic_p95": bh.get("null_ic_p95"),
            "beats_null": bh.get("beats_null"), "deflated_ic": bh.get("deflated_ic"),
            "n_independent_groups": bh.get("n_independent_groups"),
            "verdict": bh.get("verdict"),
            "folds": _recs(pd.read_sql_query(
                "SELECT fold, test_start, test_end, ic, long_short, null_ic_p95 "
                "FROM backtest_results WHERE horizon_days=? ORDER BY fold", con, params=[h])),
        }
    out["backtest"] = {"overall_verdict": bt.get("overall_verdict"),
                       "date_range": bt.get("date_range"), "n_rows": bt.get("n_rows"),
                       "n_tickers": bt.get("n_tickers")}
    return out


def events(con, limit=100):
    return _recs(pd.read_sql_query("""
        SELECT e.key, e.event_date, e.car, e.car_tstat, e.beta, e.r2, e.n_obs,
               t.ticker, t.direction, t.member_id, m.full_name
        FROM event_studies e
        LEFT JOIN transactions t ON t.txn_id = e.key
        LEFT JOIN members m ON m.member_id = t.member_id
        WHERE e.scope='txn' AND e.car_tstat IS NOT NULL
        ORDER BY ABS(e.car_tstat) DESC LIMIT ?""", con, params=[limit]))
