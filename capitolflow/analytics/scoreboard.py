"""The track record: what the model picked, and what those picks actually did.

Two kinds of record live here, and keeping them apart is the whole point.

**Backtested** — picks reconstructed for historical dates using only data that
existed on each of those dates. Weights are refit at every step from the past
alone. This is honest simulation, and it is available immediately, but it is
still simulation: it knows which companies exist today, which survived, and it
never had to place a real order.

**Live** — picks that were written into the database *before* the outcome was
knowable, then scored later. This is the only real evidence, and on a new
install it is empty until enough time passes. That emptiness is not a bug and
must not be papered over with the backtested curve.

The dashboard shows both, labelled, never merged into one number.
"""
from __future__ import annotations
import logging
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from ..config import SETTINGS
from .backtest import FACTORS, fit_weights
from .features import LONG_HORIZON, SHORT_HORIZON, factor_scores

log = logging.getLogger(__name__)

TOP_N = 10
MIN_TRAIN_DATES = 30
# Trading days -> calendar days, for locating the exit price.
CAL = 1.45


def _price_panel(con):
    px = pd.read_sql_query("SELECT ticker, date, adj_close FROM prices", con)
    if px.empty:
        return None
    px["date"] = pd.to_datetime(px["date"])
    wide = px.pivot_table(index="date", columns="ticker", values="adj_close",
                          aggfunc="last").sort_index()
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    return wide.reindex(full).ffill()


def _at(panel, tic, when):
    if panel is None or tic not in panel.columns:
        return np.nan
    i = panel.index.searchsorted(pd.Timestamp(when), side="left")
    if i >= len(panel):
        return np.nan
    v = panel[tic].iloc[i]
    return float(v) if pd.notna(v) else np.nan


# ------------------------------------------------------------------ simulation
def simulate(con, panel: pd.DataFrame | None = None, *, horizons=(SHORT_HORIZON, LONG_HORIZON),
             top_n: int = TOP_N, step: int = 4, store_rows: bool = True) -> dict:
    """Walk history forward, picking a top-N at each step from past data only.

    At each rebalance date the weights are refit on everything strictly before
    that date — no peeking. The resulting picks are then scored against the
    prices that followed. `step` is in panel periods (weeks), so 4 is monthly.
    """
    if panel is None:
        from .features import build
        panel = build(con)
    if panel is None or panel.empty:
        return {"status": "no_data"}

    scores = factor_scores(panel)
    prices = _price_panel(con)
    bench = SETTINGS.benchmark
    if prices is None or bench not in prices.columns:
        return {"status": "no_prices"}

    out = {"status": "ok", "horizons": {}}
    all_rows = []

    for h in horizons:
        ycol = f"y_{h}"
        sub = scores.dropna(subset=[ycol]) if ycol in scores.columns else scores
        dates = sorted(pd.unique(scores["date"]))
        rebal = dates[MIN_TRAIN_DATES::step]
        rows = []
        for d in rebal:
            d = pd.Timestamp(d)
            # Train strictly on rows whose forward window closed before d.
            train = sub[sub["date"] <= d - pd.Timedelta(days=int(h * CAL) + 5)]
            if len(train) < 100:
                continue
            w, _ = fit_weights(train[FACTORS].to_numpy(dtype=float),
                               train[ycol].to_numpy(dtype=float),
                               n_boot=12, seed=int(d.toordinal() % 100000),
                               groups=train["ticker"].to_numpy())
            today = scores[scores["date"] == d]
            if today.empty:
                continue
            comp = today[FACTORS].to_numpy(dtype=float) @ w
            pick = today.assign(_s=comp).sort_values("_s", ascending=False).head(top_n)

            exit_d = d + pd.Timedelta(days=int(h * CAL))
            resolved = exit_d <= prices.index[-1]
            b0, b1 = _at(prices, bench, d), _at(prices, bench, exit_d)
            for i, (_, r) in enumerate(pick.iterrows()):
                p0, p1 = _at(prices, r["ticker"], d), _at(prices, r["ticker"], exit_d)
                ok = resolved and all(np.isfinite(v) and v > 0 for v in (p0, p1, b0, b1))
                rows.append({
                    "as_of": str(d.date()), "horizon_days": h, "ticker": r["ticker"],
                    "rank": i + 1, "score": float(r["_s"]), "mode": "backtested",
                    "realised_return": float(p1 / p0 - 1) if ok else None,
                    "benchmark_return": float(b1 / b0 - 1) if ok else None,
                    "excess_return": float((p1 / p0 - 1) - (b1 / b0 - 1)) if ok else None,
                    "resolved": 1 if ok else 0,
                })
        all_rows += rows
        out["horizons"][h] = summarize(pd.DataFrame(rows), horizon=h)

    if store_rows and all_rows:
        from ..db import upsert_many
        con.execute("DELETE FROM pick_history WHERE mode='backtested'")
        upsert_many(con, "pick_history", all_rows, mode="REPLACE")
    return out


# ------------------------------------------------------------------ scoring
def resolve_live(con, horizons=(SHORT_HORIZON, LONG_HORIZON)) -> int:
    """Score stored live predictions whose horizon has now elapsed.

    Predictions are written the day they are made; this fills in what happened
    once enough time has passed. Nothing here can alter a pick — only record its
    outcome — which is what makes the live record trustworthy.
    """
    prices = _price_panel(con)
    bench = SETTINGS.benchmark
    if prices is None or bench not in prices.columns:
        return 0
    last = prices.index[-1]
    n = 0
    for h in horizons:
        preds = pd.read_sql_query(
            "SELECT as_of, ticker, rank, score FROM predictions WHERE horizon_days=?",
            con, params=[h])
        for r in preds.itertuples(index=False):
            d0 = pd.Timestamp(r.as_of)
            d1 = d0 + pd.Timedelta(days=int(h * CAL))
            if d1 > last:
                continue                                   # not yet resolvable
            p0, p1 = _at(prices, r.ticker, d0), _at(prices, r.ticker, d1)
            b0, b1 = _at(prices, bench, d0), _at(prices, bench, d1)
            if not all(np.isfinite(v) and v > 0 for v in (p0, p1, b0, b1)):
                continue
            con.execute("""
                INSERT OR REPLACE INTO pick_history
                (as_of,horizon_days,ticker,rank,score,mode,realised_return,
                 benchmark_return,excess_return,resolved)
                VALUES (?,?,?,?,?, 'live', ?,?,?,1)""",
                (r.as_of, h, r.ticker, r.rank, r.score,
                 float(p1 / p0 - 1), float(b1 / b0 - 1),
                 float((p1 / p0 - 1) - (b1 / b0 - 1))))
            n += 1
    return n


def _non_overlapping(by_date: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Keep only rebalance dates whose holding periods do not overlap.

    Rebalancing monthly while holding for six months means consecutive periods
    share five months of the same market move. Chaining those returns compounds
    one move six times and produces a curve that is pure arithmetic fantasy — a
    six-month strategy showed +350,000% before this was enforced. A cumulative
    return is only meaningful over periods you could actually have held in
    sequence, so the chain walks forward taking the next date that starts after
    the previous position closed.
    """
    if by_date.empty:
        return by_date
    need = pd.Timedelta(days=int(horizon * CAL))
    keep, last_exit = [], None
    for row in by_date.to_dict("records"):
        d = pd.Timestamp(row["as_of"])
        if last_exit is None or d >= last_exit:
            keep.append(row)
            last_exit = d + need
    return pd.DataFrame(keep)


def summarize(df: pd.DataFrame, horizon: int | None = None) -> dict:
    """Headline accuracy and return figures for one horizon and mode."""
    if df is None or df.empty:
        return {"status": "no_picks", "n_picks": 0}
    r = df[df["resolved"] == 1].dropna(subset=["excess_return"])
    if r.empty:
        return {"status": "unresolved", "n_picks": int(len(df)),
                "note": "picks exist but their horizons have not elapsed yet"}
    if horizon is None:
        horizon = int(df["horizon_days"].iloc[0])

    # Equal-weight basket per rebalance date...
    by_date = r.groupby("as_of").agg(
        pick=("realised_return", "mean"),
        bench=("benchmark_return", "mean"),
        n=("ticker", "size")).reset_index().sort_values("as_of")
    # ...then chain ONLY the periods that could be held back to back.
    chain_src = _non_overlapping(by_date, horizon)
    chain = float(np.prod(1 + chain_src["pick"].to_numpy()) - 1)
    chain_b = float(np.prod(1 + chain_src["bench"].to_numpy()) - 1)

    exc = r["excess_return"].to_numpy(dtype=float)
    sd = float(exc.std(ddof=1)) if len(exc) > 1 else np.nan
    t = float(exc.mean() / (sd / np.sqrt(len(exc)))) if sd and len(exc) > 1 else np.nan

    return {
        "status": "ok",
        "n_picks": int(len(r)),
        "n_periods": int(len(by_date)),
        "n_chained_periods": int(len(chain_src)),
        "first_period": str(by_date["as_of"].iloc[0]),
        "last_period": str(by_date["as_of"].iloc[-1]),
        # Accuracy: share of individual picks that beat the benchmark.
        "hit_rate": float((exc > 0).mean()),
        "mean_excess_per_pick": float(exc.mean()),
        "median_excess_per_pick": float(np.median(exc)),
        "t_stat": t,
        # Compounded basket return vs simply holding the benchmark.
        "cumulative_return": chain,
        "benchmark_cumulative_return": chain_b,
        "cumulative_excess": chain - chain_b,
        "periods_beating_benchmark": float((by_date["pick"] > by_date["bench"]).mean()),
        "annualized_excess": _annualize(chain, chain_b, chain_src, horizon),
        "best_pick": _extreme(r, True),
        "worst_pick": _extreme(r, False),
        "curve": _curve(chain_src),
    }


MIN_PERIODS_TO_ANNUALIZE = 3


def _annualize(chain: float, chain_b: float, chain_src: pd.DataFrame,
               horizon: int) -> float:
    """Excess return per year, so horizons can be compared on equal footing.

    Refuses to extrapolate from a handful of periods. Scaling one 21-day result
    up to an annual rate turns a rounding error into a headline number, and a
    brand-new install has exactly one period — which is precisely when a
    confident-looking annual figure would do the most damage.
    """
    if chain_src is None or len(chain_src) < MIN_PERIODS_TO_ANNUALIZE:
        return float("nan")
    years = len(chain_src) * horizon * CAL / 365.25
    if years <= 0:
        return float("nan")
    try:
        a = (1 + chain) ** (1 / years) - 1
        b = (1 + chain_b) ** (1 / years) - 1
        return float(a - b)
    except (ValueError, ZeroDivisionError):
        return float("nan")


def _extreme(r, best: bool) -> dict:
    i = r["excess_return"].idxmax() if best else r["excess_return"].idxmin()
    row = r.loc[i]
    return {"ticker": row["ticker"], "as_of": row["as_of"],
            "excess_return": float(row["excess_return"])}


def _curve(by_date: pd.DataFrame) -> list[dict]:
    """Growth of 1.0 in the picks vs the benchmark, period by period."""
    out, p, b = [], 1.0, 1.0
    for r in by_date.to_dict("records"):
        p *= (1 + r["pick"])
        b *= (1 + r["bench"])
        out.append({"period": r["as_of"], "picks": round(p, 6),
                    "benchmark": round(b, 6), "n_picks": int(r["n"]),
                    "n_members": None})
    return out


def report(con, horizons=(SHORT_HORIZON, LONG_HORIZON)) -> dict:
    """Everything the dashboard needs, with backtested and live kept separate."""
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "backtested": {}, "live": {}}
    for mode in ("backtested", "live"):
        for h in horizons:
            df = pd.read_sql_query(
                "SELECT * FROM pick_history WHERE mode=? AND horizon_days=?",
                con, params=[mode, h])
            out[mode][h] = summarize(df, horizon=h)
    out["caveat"] = (
        "Backtested results are a simulation run on history the model can see in "
        "full; they are not a track record. The live section is the only forward "
        "evidence, and it stays empty until predictions made today have had their "
        "full horizon to play out. Neither figure includes trading costs, "
        "slippage, or taxes.")
    return out
