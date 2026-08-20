"""Standard market-model event study around trade dates and policy events.

For each event we fit  r_i,t = alpha + beta * r_mkt,t + e  over an estimation
window that ENDS before the event (default [-250, -30] trading days), then sum
the residuals over the event window to get a cumulative abnormal return (CAR).
The t-statistic uses the estimation-window residual standard deviation, which is
the conventional Brown-Warner approach and keeps event-window volatility from
flattering the result.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from ..config import SETTINGS

log = logging.getLogger(__name__)


def _returns_panel(con) -> pd.DataFrame:
    px = pd.read_sql_query("SELECT ticker, date, adj_close FROM prices", con)
    if px.empty:
        return px
    px["date"] = pd.to_datetime(px["date"])
    wide = px.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
    return wide.sort_index().pct_change(fill_method=None)


def car_for_event(rets: pd.DataFrame, ticker: str, event_date, *,
                  bench: str | None = None, est=(-250, -30), win=(0, 20)) -> dict | None:
    bench = bench or SETTINGS.benchmark
    if ticker not in rets.columns or bench not in rets.columns:
        return None
    idx = rets.index
    ev = pd.Timestamp(event_date)
    pos = idx.searchsorted(ev, side="left")
    if pos >= len(idx):
        return None

    e0, e1 = pos + est[0], pos + est[1]
    w0, w1 = pos + win[0], pos + win[1] + 1
    if e0 < 0 or w1 > len(idx) or e1 <= e0 + 30:
        return None

    est_df = pd.concat([rets[ticker].iloc[e0:e1], rets[bench].iloc[e0:e1]], axis=1).dropna()
    est_df.columns = ["y", "x"]
    if len(est_df) < 40:
        return None
    x = np.c_[np.ones(len(est_df)), est_df["x"].to_numpy()]
    y = est_df["y"].to_numpy()
    try:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y - x @ coef
    sd = float(resid.std(ddof=2))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan

    win_df = pd.concat([rets[ticker].iloc[w0:w1], rets[bench].iloc[w0:w1]], axis=1).dropna()
    win_df.columns = ["y", "x"]
    if win_df.empty:
        return None
    ar = win_df["y"] - (alpha + beta * win_df["x"])
    car = float(ar.sum())
    n = len(ar)
    tstat = car / (sd * np.sqrt(n)) if sd > 0 else np.nan
    return {"alpha": alpha, "beta": beta, "r2": r2, "car": car,
            "car_tstat": float(tstat), "n_obs": int(n)}


def study_trades(con, *, min_conf: float = 0.7, win=(0, 20), est=(-250, -30),
                 limit: int | None = None, store: bool = True) -> pd.DataFrame:
    rets = _returns_panel(con)
    if rets.empty:
        return pd.DataFrame()
    q = """SELECT txn_id, member_id, ticker, transaction_date, direction
           FROM transactions
           WHERE ticker IS NOT NULL AND ticker_confidence >= ? AND direction != 0
             AND transaction_date IS NOT NULL AND asset_type IN ('stock','fund','option')"""
    txns = pd.read_sql_query(q, con, params=[min_conf])
    if limit:
        txns = txns.head(limit)
    out = []
    for r in txns.itertuples(index=False):
        res = car_for_event(rets, r.ticker, r.transaction_date, est=est, win=win)
        if not res:
            continue
        res.update({"scope": "txn", "key": r.txn_id, "event_date": r.transaction_date,
                    "window_start": win[0], "window_end": win[1],
                    "ticker": r.ticker, "member_id": r.member_id, "direction": r.direction})
        res["signed_car"] = res["car"] * r.direction
        out.append(res)
    df = pd.DataFrame(out)
    if store and not df.empty:
        _store(con, df)
    return df


def study_policy_events(con, *, win=(-5, 20), est=(-250, -30), store: bool = True) -> pd.DataFrame:
    import json
    rets = _returns_panel(con)
    evs = pd.read_sql_query("SELECT * FROM policy_events", con)
    if rets.empty or evs.empty:
        return pd.DataFrame()
    out = []
    for e in evs.itertuples(index=False):
        try:
            tickers = json.loads(e.tickers or "[]")
        except Exception:
            tickers = []
        for t in tickers:
            res = car_for_event(rets, t, e.event_date, est=est, win=win)
            if not res:
                continue
            res.update({"scope": "policy_event", "key": f"{e.event_id}:{t}",
                        "event_date": e.event_date, "window_start": win[0],
                        "window_end": win[1], "ticker": t, "category": e.category,
                        "title": e.title})
            out.append(res)
    df = pd.DataFrame(out)
    if store and not df.empty:
        _store(con, df)
    return df


def _store(con, df: pd.DataFrame) -> None:
    cols = ["scope", "key", "event_date", "window_start", "window_end",
            "alpha", "beta", "r2", "car", "car_tstat", "n_obs"]
    con.executemany(
        f"INSERT INTO event_studies ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
        df[cols].astype(object).where(pd.notna(df[cols]), None).values.tolist())


def summarize(df: pd.DataFrame, by: str = "direction") -> pd.DataFrame:
    """Aggregate CARs with the cross-sectional t-test that the literature uses."""
    if df.empty or "signed_car" not in df.columns:
        return pd.DataFrame()
    g = df.groupby(by)["signed_car"]
    res = pd.DataFrame({"n": g.size(), "mean_car": g.mean(), "median_car": g.median(),
                        "sd": g.std(ddof=1)})
    res["t_stat"] = res["mean_car"] / (res["sd"] / np.sqrt(res["n"]))
    res["pct_positive"] = g.apply(lambda x: float((x > 0).mean()))
    return res.reset_index()
