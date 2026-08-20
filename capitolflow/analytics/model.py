"""Predictive layer: does congressional trading + lobbying carry any forward signal?

Design choices that matter more than the algorithm:
  * Panel is (ticker, month). Target is the ticker's forward 63-trading-day
    excess return over the benchmark.
  * Features are computed from a trailing window that ends at the feature date,
    and — critically — only from trades that had ALREADY BEEN DISCLOSED by then.
    A backtest that uses trade date for feature timing is using information no
    investor had, and will look brilliant and be worthless. Both variants are
    computed so the gap between them can be measured directly.
  * Validation is a purged, embargoed walk-forward split. Random K-fold on
    overlapping forward returns leaks the future and is the single most common
    way this kind of study fools itself.
Treat the output as exploratory. A positive IC on this data is a hypothesis.
"""
from __future__ import annotations
import json, logging
import numpy as np
import pandas as pd

from ..config import SETTINGS

log = logging.getLogger(__name__)

FWD_DAYS = 63              # ~3 trading months
FEATURE_LOOKBACKS = (30, 90, 180)


# ------------------------------------------------------------------ panel
def build_panel(con, *, use_disclosure_timing: bool = True, freq: str = "M") -> pd.DataFrame:
    px = pd.read_sql_query("SELECT ticker, date, adj_close FROM prices", con)
    if px.empty:
        return pd.DataFrame()
    px["date"] = pd.to_datetime(px["date"])
    wide = px.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
    bench = SETTINGS.benchmark
    if bench not in wide.columns:
        log.warning("benchmark %s missing", bench)
        return pd.DataFrame()

    fwd = wide.shift(-FWD_DAYS) / wide - 1.0
    fwd_b = (wide[bench].shift(-FWD_DAYS) / wide[bench] - 1.0)
    excess = fwd.sub(fwd_b, axis=0)

    txns = pd.read_sql_query("""
        SELECT t.ticker, t.member_id, t.transaction_date, t.filed_date, t.direction,
               COALESCE(t.amount_est,0) AS amount_est, t.filing_delay_days,
               COALESCE(s.weight,1.0) AS mweight
        FROM transactions t
        LEFT JOIN member_scores s ON s.member_id=t.member_id AND s.horizon_days=90
        WHERE t.ticker IS NOT NULL AND t.ticker_confidence>=0.7 AND t.direction!=0
          AND t.transaction_date IS NOT NULL""", con)
    if txns.empty:
        return pd.DataFrame()
    txns["transaction_date"] = pd.to_datetime(txns["transaction_date"], errors="coerce")
    txns["filed_date"] = pd.to_datetime(txns["filed_date"], errors="coerce")
    # The date at which a feature may legitimately observe this trade.
    txns["known_at"] = (txns["filed_date"] if use_disclosure_timing
                        else txns["transaction_date"])
    txns["known_at"] = txns["known_at"].fillna(txns["transaction_date"])
    txns = txns.dropna(subset=["known_at"])

    lob = pd.read_sql_query("""
        SELECT ticker, period_end, COALESCE(amount,0) AS amount FROM lobbying_filings
        WHERE ticker IS NOT NULL AND ticker_confidence>=0.6 AND period_end IS NOT NULL""", con)
    if not lob.empty:
        lob["period_end"] = pd.to_datetime(lob["period_end"], errors="coerce")
        lob = lob.dropna(subset=["period_end"])

    period = wide.index.to_period(freq.rstrip("E") or "M")
    dates = pd.Series(wide.index).groupby(period.to_numpy()).last().tolist()
    tickers = [t for t in wide.columns if t != bench]

    rows = []
    for d in dates:
        d = pd.Timestamp(d)
        for t in tickers:
            y = excess.at[d, t] if (d in excess.index and t in excess.columns) else np.nan
            if not np.isfinite(y):
                continue
            rec = {"date": d, "ticker": t, "y_fwd_excess": float(y)}
            sub_all = txns[(txns["ticker"] == t) & (txns["known_at"] <= d)]
            for lb in FEATURE_LOOKBACKS:
                w = sub_all[sub_all["known_at"] > d - pd.Timedelta(days=lb)]
                buys = w[w["direction"] > 0]
                sells = w[w["direction"] < 0]
                rec[f"n_members_{lb}"] = w["member_id"].nunique()
                rec[f"n_trades_{lb}"] = len(w)
                rec[f"n_buyers_{lb}"] = buys["member_id"].nunique()
                rec[f"n_sellers_{lb}"] = sells["member_id"].nunique()
                gross = float(w["amount_est"].sum())
                net = float((w["amount_est"] * w["direction"]).sum())
                wnet = float((w["amount_est"] * w["direction"] * w["mweight"]).sum())
                rec[f"log_gross_{lb}"] = float(np.log1p(gross))
                rec[f"net_ratio_{lb}"] = net / gross if gross > 0 else 0.0
                rec[f"wnet_ratio_{lb}"] = wnet / gross if gross > 0 else 0.0
                rec[f"buy_share_{lb}"] = (len(buys) / len(w)) if len(w) else 0.5
                rec[f"mean_delay_{lb}"] = float(w["filing_delay_days"].mean()) if len(w) else np.nan
                rec[f"max_mweight_{lb}"] = float(w["mweight"].max()) if len(w) else 1.0
            # momentum / vol controls, so the model cannot just relearn price trend
            hist = wide[t].loc[:d].tail(130)
            if len(hist) > 30:
                rec["mom_20"] = float(hist.iloc[-1] / hist.iloc[-21] - 1) if len(hist) > 21 else 0.0
                rec["mom_120"] = float(hist.iloc[-1] / hist.iloc[0] - 1)
                rec["vol_60"] = float(hist.pct_change(fill_method=None).tail(60).std())
            if not lob.empty:
                lw = lob[(lob["ticker"] == t) & (lob["period_end"] <= d) &
                         (lob["period_end"] > d - pd.Timedelta(days=365))]
                rec["log_lobby_12m"] = float(np.log1p(lw["amount"].sum()))
                prev = lob[(lob["ticker"] == t) & (lob["period_end"] <= d - pd.Timedelta(days=365)) &
                           (lob["period_end"] > d - pd.Timedelta(days=730))]
                a, b = lw["amount"].sum(), prev["amount"].sum()
                rec["lobby_yoy"] = float((a - b) / b) if b > 0 else 0.0
            rows.append(rec)
    df = pd.DataFrame(rows)
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


# ------------------------------------------------------------------ CV
def purged_walk_forward(dates: pd.Series, n_splits: int = 5, embargo_days: int = FWD_DAYS + 5):
    """Expanding-window splits with a gap that covers the forward-return horizon."""
    uniq = np.array(sorted(pd.unique(dates)))
    if len(uniq) < n_splits + 2:
        return
    bounds = np.array_split(np.arange(len(uniq)), n_splits + 1)
    for k in range(1, n_splits + 1):
        test_idx = bounds[k]
        test_start = uniq[test_idx[0]]
        cutoff = pd.Timestamp(test_start) - pd.Timedelta(days=embargo_days)
        train_mask = dates <= cutoff
        test_mask = dates.isin(uniq[test_idx])
        if train_mask.sum() < 50 or test_mask.sum() < 10:
            continue
        yield np.where(train_mask)[0], np.where(test_mask)[0]


def _spearman(a, b) -> float:
    from scipy import stats
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    return float(stats.spearmanr(a[m], b[m]).statistic)


def train(con=None, panel: pd.DataFrame | None = None, *, n_splits: int = 5,
          use_disclosure_timing: bool = True, model_out: str | None = None) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.dummy import DummyRegressor

    if panel is None:
        panel = build_panel(con, use_disclosure_timing=use_disclosure_timing)
    if panel is None or panel.empty or len(panel) < 200:
        return {"status": "insufficient_data", "n_rows": 0 if panel is None else len(panel)}

    feats = [c for c in panel.columns if c not in ("date", "ticker", "y_fwd_excess")]
    X = panel[feats].to_numpy(dtype=float)
    y = panel["y_fwd_excess"].to_numpy(dtype=float)
    dates = panel["date"]

    fold_rows, oof = [], np.full(len(panel), np.nan)
    for k, (tr, te) in enumerate(purged_walk_forward(dates, n_splits)):
        m = HistGradientBoostingRegressor(
            max_depth=3, max_iter=250, learning_rate=0.05, min_samples_leaf=40,
            l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        oof[te] = p
        base = DummyRegressor(strategy="mean").fit(X[tr], y[tr]).predict(X[te])
        fold_rows.append({
            "fold": k, "n_train": len(tr), "n_test": len(te),
            "test_start": str(pd.Timestamp(dates.iloc[te].min()).date()),
            "ic_spearman": _spearman(p, y[te]),
            "rmse": float(np.sqrt(np.mean((p - y[te]) ** 2))),
            "rmse_baseline": float(np.sqrt(np.mean((base - y[te]) ** 2))),
            "long_short_spread": _decile_spread(p, y[te]),
        })
    if not fold_rows:
        return {"status": "insufficient_folds", "n_rows": len(panel)}

    folds = pd.DataFrame(fold_rows)
    final = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.05,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          random_state=0).fit(X, y)
    imp = permutation_importance_safe(final, X, y, feats)

    if model_out:
        import pickle
        from pathlib import Path
        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        with open(model_out, "wb") as fh:
            pickle.dump({"model": final, "features": feats,
                         "use_disclosure_timing": use_disclosure_timing}, fh)

    return {
        "status": "ok",
        "n_rows": int(len(panel)), "n_features": len(feats),
        "use_disclosure_timing": use_disclosure_timing,
        "mean_ic": float(np.nanmean(folds["ic_spearman"])),
        "ic_std": float(np.nanstd(folds["ic_spearman"])),
        "mean_long_short_spread": float(np.nanmean(folds["long_short_spread"])),
        "beats_baseline_folds": int((folds["rmse"] < folds["rmse_baseline"]).sum()),
        "n_folds": int(len(folds)),
        "folds": folds.round(5).to_dict("records"),
        "top_features": imp[:15],
    }


def _decile_spread(pred, actual, q: int = 5) -> float:
    """Mean actual return of the top predicted bucket minus the bottom bucket."""
    if len(pred) < q * 4:
        return np.nan
    df = pd.DataFrame({"p": pred, "a": actual}).dropna()
    if len(df) < q * 4:
        return np.nan
    try:
        df["b"] = pd.qcut(df["p"].rank(method="first"), q, labels=False)
    except ValueError:
        return np.nan
    return float(df[df.b == q - 1]["a"].mean() - df[df.b == 0]["a"].mean())


def permutation_importance_safe(model, X, y, feats, n_repeats: int = 5) -> list[dict]:
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=0,
                                   scoring="neg_mean_squared_error")
        order = np.argsort(-r.importances_mean)
        return [{"feature": feats[i], "importance": float(r.importances_mean[i]),
                 "std": float(r.importances_std[i])} for i in order]
    except Exception as e:
        log.warning("permutation importance failed: %s", e)
        return []


def leakage_check(con, **kw) -> dict:
    """Train twice — once with honest disclosure timing, once with trade-date
    timing — and report the gap. A large gap is the size of the information
    advantage that is NOT available to the public in real time."""
    honest = train(con, use_disclosure_timing=True, **kw)
    oracle = train(con, use_disclosure_timing=False, **kw)
    return {
        "disclosure_timed_ic": honest.get("mean_ic"),
        "trade_date_timed_ic": oracle.get("mean_ic"),
        "ic_gap": (None if honest.get("mean_ic") is None or oracle.get("mean_ic") is None
                   else oracle["mean_ic"] - honest["mean_ic"]),
        "honest": honest, "oracle": oracle,
    }
