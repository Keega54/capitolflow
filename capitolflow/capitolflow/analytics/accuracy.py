"""Per-member trading accuracy, shrunk so small samples cannot top the board.

A member with four lucky trades will show a huge raw mean excess return. Ranking
on that is noise. We use an empirical-Bayes (James-Stein style) shrink toward the
population mean, with the shrink factor driven by how much of the observed
spread is signal versus within-member sampling error:

    shrunk_i = mu + B_i * (xbar_i - mu),   B_i = tau^2 / (tau^2 + sigma^2 / n_i)

tau^2 is the between-member variance of true skill, estimated by subtracting the
average sampling variance from the observed cross-sectional variance. When a
member has few trades, B_i -> 0 and they are pulled to the average, which is the
honest answer.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..config import SETTINGS

MIN_TRADES = 5


def compute_member_scores(con, horizons=None, min_trades: int = MIN_TRADES) -> pd.DataFrame:
    horizons = list(horizons or SETTINGS.horizons)
    df = pd.read_sql_query("""
        SELECT r.txn_id, r.horizon_days, r.excess_return, t.member_id, t.amount_est
        FROM trade_returns r JOIN transactions t USING (txn_id)
        WHERE t.member_id IS NOT NULL
    """, con)
    if df.empty:
        return pd.DataFrame()

    frames = []
    for h in horizons:
        sub = df[df["horizon_days"] == h].dropna(subset=["excess_return"])
        if sub.empty:
            continue
        g = sub.groupby("member_id")["excess_return"]
        stats = pd.DataFrame({
            "n_scored": g.size(),
            "mean_excess": g.mean(),
            "sd_excess": g.std(ddof=1),
            "hit_rate": g.apply(lambda x: float((x > 0).mean())),
        })
        # dollar-weighted mean
        w = sub.copy()
        w["amount_est"] = w["amount_est"].fillna(w["amount_est"].median() or 1.0).clip(lower=1.0)
        dw = (w.assign(num=w["excess_return"] * w["amount_est"])
                .groupby("member_id")
                .apply(lambda d: d["num"].sum() / d["amount_est"].sum(), include_groups=False))
        stats["dollar_wtd_excess"] = dw

        elig = stats[stats["n_scored"] >= min_trades].copy()
        if elig.empty:
            elig = stats.copy()

        mu = float(np.average(elig["mean_excess"], weights=elig["n_scored"]))
        # average within-member sampling variance
        sd = elig["sd_excess"].fillna(elig["sd_excess"].median())
        sigma2 = float(np.nanmean((sd ** 2) / elig["n_scored"].clip(lower=1)))
        observed_var = float(np.nanvar(elig["mean_excess"], ddof=1)) if len(elig) > 1 else 0.0
        tau2 = max(observed_var - sigma2, 1e-9)

        var_i = (stats["sd_excess"].fillna(sd.median()) ** 2) / stats["n_scored"].clip(lower=1)
        B = tau2 / (tau2 + var_i)
        stats["shrunk_excess"] = mu + B * (stats["mean_excess"] - mu)
        stats["t_stat"] = (stats["mean_excess"] /
                           (stats["sd_excess"] / np.sqrt(stats["n_scored"].clip(lower=1)))).replace(
                               [np.inf, -np.inf], np.nan)

        # Weight used in weighted aggregates: centred on 1.0, bounded, driven by
        # shrunk skill scaled by the cross-sectional spread of shrunk skill.
        spread = float(stats["shrunk_excess"].std(ddof=1)) or 1.0
        stats["weight"] = (1.0 + (stats["shrunk_excess"] - mu) / (2 * spread)).clip(0.25, 2.0)
        stats["horizon_days"] = h
        frames.append(stats.reset_index())

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)

    counts = pd.read_sql_query(
        "SELECT member_id, COUNT(*) n_trades FROM transactions WHERE member_id IS NOT NULL"
        " GROUP BY member_id", con)
    out = out.merge(counts, on="member_id", how="left")
    return out


def store_member_scores(con, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    cols = ["member_id", "horizon_days", "n_trades", "n_scored", "hit_rate", "mean_excess",
            "dollar_wtd_excess", "shrunk_excess", "weight", "t_stat"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    rows = df[cols].astype(object).where(pd.notna(df[cols]), None).values.tolist()
    con.executemany(
        "INSERT OR REPLACE INTO member_scores (member_id,horizon_days,n_trades,n_scored,hit_rate,"
        "mean_excess,dollar_wtd_excess,shrunk_excess,weight,t_stat,computed_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'))", rows)
    return len(rows)


def member_weights(con, horizon: int = 90) -> dict:
    return {r["member_id"]: r["weight"] for r in con.execute(
        "SELECT member_id, weight FROM member_scores WHERE horizon_days=?", (horizon,))}
