"""Disclosure-lag analysis: how much of a politician's edge is still on the table
by the time the public can act on it?

A member buys on day 0 and the filing appears on day 34. Any move between those
two dates is unreachable — nobody outside the filer could have traded it. Only
the move *after* the disclosure date is capturable, and that is the only part a
signal built on this data is entitled to claim.

So every trade's excess return is split in two:

    total = pre_disclosure (unreachable) + post_disclosure (capturable)

and the ratio post/total is the `capturable_share`. If it turns out that the
edge is concentrated before the filing hits — which is what you would expect if
the information advantage is real and the market absorbs it quickly — then the
honest conclusion is that following disclosures is close to worthless, and this
module is what will tell you so rather than hiding it inside a backtest.

It also fits the signal's decay half-life from the data instead of assuming one,
so downstream code can age a stale disclosure correctly.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from ..config import SETTINGS

log = logging.getLogger(__name__)

DEFAULT_HORIZON = 90


def _panel(con) -> pd.DataFrame:
    px = pd.read_sql_query("SELECT ticker, date, adj_close FROM prices", con)
    if px.empty:
        return px
    px["date"] = pd.to_datetime(px["date"])
    wide = px.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    return wide.reindex(full).ffill()


def _px(panel: pd.DataFrame, tic: str, when) -> float:
    if tic not in panel.columns:
        return np.nan
    col = panel[tic]
    i = col.index.searchsorted(pd.Timestamp(when), side="left")
    if i >= len(col):
        return np.nan
    v = col.iloc[i]
    return float(v) if pd.notna(v) else np.nan


def decompose(con, horizon: int = DEFAULT_HORIZON, min_confidence: float = 0.7) -> pd.DataFrame:
    """Split every trade's excess return at the disclosure date."""
    bench = SETTINGS.benchmark
    panel = _panel(con)
    if panel.empty or bench not in panel.columns:
        log.warning("no price panel or benchmark; run `capitolflow prices` first")
        return pd.DataFrame()

    txns = pd.read_sql_query("""
        SELECT txn_id, member_id, ticker, transaction_date, filed_date, direction,
               filing_delay_days, COALESCE(amount_est,0) AS amount_est
        FROM transactions
        WHERE ticker IS NOT NULL AND ticker_confidence >= ? AND direction != 0
          AND transaction_date IS NOT NULL AND filed_date IS NOT NULL
          AND asset_type IN ('stock','fund','option')""", con, params=[min_confidence])
    if txns.empty:
        return pd.DataFrame()
    txns["transaction_date"] = pd.to_datetime(txns["transaction_date"], errors="coerce")
    txns["filed_date"] = pd.to_datetime(txns["filed_date"], errors="coerce")
    txns = txns.dropna(subset=["transaction_date", "filed_date"])
    # Guard against filings dated before the trade (data-entry noise).
    txns = txns[txns["filed_date"] >= txns["transaction_date"]]

    last = panel.index[-1]
    rows = []
    for r in txns.itertuples(index=False):
        t0, td = r.transaction_date, r.filed_date
        t1 = td + pd.Timedelta(days=horizon)
        if t1 > last:
            continue                                  # horizon incomplete: skip, never impute
        p0, pd_, p1 = _px(panel, r.ticker, t0), _px(panel, r.ticker, td), _px(panel, r.ticker, t1)
        b0, bd_, b1 = _px(panel, bench, t0), _px(panel, bench, td), _px(panel, bench, t1)
        vals = [p0, pd_, p1, b0, bd_, b1]
        if any(not np.isfinite(v) or v <= 0 for v in vals):
            continue
        d = r.direction
        pre = ((pd_ / p0 - 1) - (bd_ / b0 - 1)) * d
        post = ((p1 / pd_ - 1) - (b1 / bd_ - 1)) * d
        total = pre + post
        rows.append({
            "txn_id": r.txn_id, "member_id": r.member_id, "ticker": r.ticker,
            "horizon_days": horizon, "lag_days": int((td - t0).days),
            "pre_disclosure_excess": float(pre),
            "post_disclosure_excess": float(post),
            "total_excess": float(total),
            "capturable_share": float(post / total) if abs(total) > 1e-9 else np.nan,
            "amount_est": float(r.amount_est), "direction": int(d),
        })
    return pd.DataFrame(rows)


def store(con, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    cols = ["txn_id", "horizon_days", "pre_disclosure_excess", "post_disclosure_excess",
            "total_excess", "capturable_share", "lag_days"]
    con.executemany(
        f"INSERT OR REPLACE INTO trade_timing ({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})",
        df[cols].astype(object).where(pd.notna(df[cols]), None).values.tolist())
    return len(df)


def summary(df: pd.DataFrame) -> dict:
    """The headline honesty numbers."""
    if df is None or df.empty:
        return {"status": "no_data"}
    n = len(df)
    pre, post = df["pre_disclosure_excess"], df["post_disclosure_excess"]

    def t_of(x):
        s = x.std(ddof=1)
        return float(x.mean() / (s / np.sqrt(len(x)))) if s and len(x) > 1 else np.nan

    return {
        "n_trades": int(n),
        "mean_pre_disclosure_excess": float(pre.mean()),
        "mean_post_disclosure_excess": float(post.mean()),
        "t_pre": t_of(pre),
        "t_post": t_of(post),
        "pct_of_edge_before_disclosure": (
            float(pre.mean() / (pre.mean() + post.mean()))
            if abs(pre.mean() + post.mean()) > 1e-12 else np.nan),
        "post_hit_rate": float((post > 0).mean()),
        "median_lag_days": float(df["lag_days"].median()),
        "verdict": _verdict(pre, post),
    }


def _verdict(pre, post) -> str:
    tp = post.mean() / (post.std(ddof=1) / np.sqrt(len(post))) if len(post) > 1 and post.std(ddof=1) else 0
    if tp > 2:
        return ("Post-disclosure excess return is positive and statistically distinguishable "
                "from zero. A public follower could plausibly have captured some of this.")
    if tp < -2:
        return ("Post-disclosure excess return is significantly NEGATIVE — following these "
                "disclosures would have lost money against the benchmark.")
    if pre.mean() > 0 and post.mean() <= 0:
        return ("The edge sits entirely before the filing becomes public. By the time you can "
                "see the trade, the move has already happened. Treat rankings with heavy scepticism.")
    return ("Post-disclosure excess return is not distinguishable from zero. Any ranking built "
            "on this data should be read as a research artifact, not a signal.")


# ------------------------------------------------------------------ decay
def fit_decay(con, horizon: int = DEFAULT_HORIZON, max_age: int = 180,
              bucket: int = 15) -> dict:
    """Estimate how fast a disclosure goes stale.

    Buckets trades by how old the disclosure was and measures forward excess
    return from each age point. Fits ``ic(age) = a * exp(-age / tau)`` by
    least squares on the log of positive values, and returns the half-life.
    A fitted half-life is honest; a hard-coded 30-day decay is a guess.
    """
    df = pd.read_sql_query("""
        SELECT t.ticker, t.filed_date, t.direction, COALESCE(t.amount_est,0) amount_est
        FROM transactions t
        WHERE t.ticker IS NOT NULL AND t.ticker_confidence>=0.7 AND t.direction!=0
          AND t.filed_date IS NOT NULL""", con)
    panel = _panel(con)
    if df.empty or panel.empty or SETTINGS.benchmark not in panel.columns:
        return {"status": "no_data"}
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
    df = df.dropna(subset=["filed_date"])
    bench = SETTINGS.benchmark
    last = panel.index[-1]

    ages = list(range(0, max_age + 1, bucket))
    out = []
    for age in ages:
        vals = []
        for r in df.itertuples(index=False):
            start = r.filed_date + pd.Timedelta(days=age)
            end = start + pd.Timedelta(days=horizon)
            if end > last:
                continue
            p0, p1 = _px(panel, r.ticker, start), _px(panel, r.ticker, end)
            b0, b1 = _px(panel, bench, start), _px(panel, bench, end)
            if any(not np.isfinite(v) or v <= 0 for v in (p0, p1, b0, b1)):
                continue
            vals.append(((p1 / p0 - 1) - (b1 / b0 - 1)) * r.direction)
        if len(vals) >= 30:
            v = np.array(vals)
            out.append({"age_days": age, "n": len(v), "mean_excess": float(v.mean()),
                        "t_stat": float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
                                  if v.std(ddof=1) else np.nan})
    if len(out) < 3:
        return {"status": "insufficient_data", "curve": out}

    curve = pd.DataFrame(out)
    pos = curve[curve["mean_excess"] > 0]
    tau = half_life = np.nan
    if len(pos) >= 3:
        try:
            slope, intercept = np.polyfit(pos["age_days"], np.log(pos["mean_excess"]), 1)
            if slope < 0:
                tau = float(-1.0 / slope)
                half_life = float(tau * np.log(2))
        except Exception as e:                                  # degenerate fit
            log.debug("decay fit failed: %s", e)
    return {
        "status": "ok",
        "curve": curve.to_dict("records"),
        "tau_days": tau,
        "half_life_days": half_life,
        "note": ("Excess return available to someone acting N days after the filing. "
                 "A decaying curve means freshness matters; a flat curve near zero means "
                 "the disclosure never carried tradeable information."),
    }


def staleness_weight(age_days, half_life: float | None) -> float:
    """Age a disclosure using the fitted half-life. Falls back to no decay when
    the fit failed, because inventing a decay rate would be worse than none."""
    if half_life is None or not np.isfinite(half_life) or half_life <= 0:
        return 1.0
    return float(0.5 ** (np.asarray(age_days, dtype=float) / half_life))


def capturable_member_scores(con, horizon: int = DEFAULT_HORIZON,
                             min_trades: int = 5) -> pd.DataFrame:
    """Rank members by POST-disclosure edge only — the version of skill that a
    public follower could actually have acted on."""
    df = pd.read_sql_query("""
        SELECT tt.*, t.member_id, COALESCE(t.amount_est,0) amount_est
        FROM trade_timing tt JOIN transactions t USING (txn_id)
        WHERE t.member_id IS NOT NULL AND tt.horizon_days = ?""", con, params=[horizon])
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("member_id")
    res = pd.DataFrame({
        "n_scored": g.size(),
        "mean_pre": g["pre_disclosure_excess"].mean(),
        "mean_post": g["post_disclosure_excess"].mean(),
        "sd_post": g["post_disclosure_excess"].std(ddof=1),
        "post_hit_rate": g["post_disclosure_excess"].apply(lambda x: float((x > 0).mean())),
        "median_lag": g["lag_days"].median(),
    }).reset_index()
    res = res[res["n_scored"] >= min_trades]
    if res.empty:
        return res
    res["t_post"] = res["mean_post"] / (res["sd_post"] / np.sqrt(res["n_scored"]))
    # Same empirical-Bayes shrink as the headline accuracy score.
    mu = float(np.average(res["mean_post"], weights=res["n_scored"]))
    sd = res["sd_post"].fillna(res["sd_post"].median())
    sigma2 = float(np.nanmean((sd ** 2) / res["n_scored"].clip(lower=1)))
    tau2 = max(float(np.nanvar(res["mean_post"], ddof=1)) - sigma2, 1e-9) if len(res) > 1 else 1e-9
    var_i = (sd ** 2) / res["n_scored"].clip(lower=1)
    B = tau2 / (tau2 + var_i)
    res["shrunk_post"] = mu + B * (res["mean_post"] - mu)
    spread = float(res["shrunk_post"].std(ddof=1)) or 1.0
    res["capturable_weight"] = (1.0 + (res["shrunk_post"] - mu) / (2 * spread)).clip(0.25, 2.0)
    return res.sort_values("shrunk_post", ascending=False).reset_index(drop=True)
