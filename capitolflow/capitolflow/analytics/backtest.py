"""Walk-forward simulation that fits factor weights and tries to catch itself cheating.

How the weights are chosen
--------------------------
Each factor is collapsed to one cross-sectionally standardized score, and the
composite is a weighted sum. Weights are fitted by ridge regression of forward
excess return on the factor scores within a training window, then:

  * shrunk toward equal weighting, because a ridge fit on a few thousand noisy
    observations is itself a noisy estimate;
  * bootstrapped, so each factor gets a stability number — the share of
    resamples where its sign agreed — and unstable factors are pulled to zero;
  * clipped, so no single factor can dominate the composite.

Seven fitted parameters against thousands of observations is a deliberately
small budget. The temptation in this kind of project is to fit hundreds.

How overfitting is detected
---------------------------
1. Purged, embargoed walk-forward splits. Forward returns overlap, so any
   training row whose target window reaches into the test period is dropped.
2. A shuffled-label null. The identical pipeline is re-run with targets randomly
   permuted within each date, many times, producing the distribution of IC that
   pure luck yields on this data. A real IC must beat that distribution, not zero.
3. Stability across bootstrap resamples, reported per factor.
4. Deflation for multiple testing across horizons and factor sets.

If the measured IC sits inside the null distribution, the honest report is "no
signal" — and this module says so rather than presenting a number.
"""
from __future__ import annotations
import hashlib
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .features import FACTOR_COLUMNS, HORIZONS, LONG_HORIZON, SHORT_HORIZON, factor_scores

log = logging.getLogger(__name__)

FACTORS = list(FACTOR_COLUMNS.keys())
RIDGE_ALPHA = 5.0
EQUAL_WEIGHT_SHRINK = 0.35     # pull toward equal weighting
MAX_ABS_WEIGHT = 0.45
N_BOOTSTRAP = 60
N_NULL = 120


# ------------------------------------------------------------------ splits
def walk_forward(dates: pd.Series, n_splits: int = 6, horizon: int = LONG_HORIZON,
                 min_train: int = 40):
    """Expanding-window splits with a purge gap covering the forward horizon."""
    uniq = np.array(sorted(pd.unique(dates)))
    if len(uniq) < n_splits + 3:
        return
    blocks = np.array_split(np.arange(len(uniq)), n_splits + 1)
    for k in range(1, n_splits + 1):
        test_idx = blocks[k]
        test_dates = uniq[test_idx]
        # A training row's target window must close before the test period opens.
        cutoff = pd.Timestamp(test_dates[0]) - pd.Timedelta(days=int(horizon * 1.5) + 5)
        train_mask = dates <= cutoff
        test_mask = dates.isin(test_dates)
        if train_mask.sum() < min_train or test_mask.sum() < 10:
            continue
        yield np.where(train_mask)[0], np.where(test_mask)[0], test_dates


# ------------------------------------------------------------------ fitting
def fit_weights(X: np.ndarray, y: np.ndarray, *, alpha: float = RIDGE_ALPHA,
                n_boot: int = N_BOOTSTRAP, seed: int = 0,
                groups: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Ridge + BLOCK-bootstrap stability + shrink toward equal weights.

    The bootstrap resamples whole tickers, not individual rows. Rows for one
    ticker across overlapping forward-return windows are massively dependent, so
    an iid row bootstrap treats ~200 genuinely independent observations as
    thousands and reports near-perfect stability for factors that are pure noise.
    Resampling at the ticker level is the difference between a stability number
    that means something and one that always says 1.0.
    """
    from sklearn.linear_model import Ridge

    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    if groups is not None:
        groups = np.asarray(groups)[ok]
    n, p = X.shape
    if n < max(30, p * 5):
        w = np.ones(p) / p
        return w, {"n": int(n), "note": "too few observations; fell back to equal weights",
                   "stability": {f: 0.0 for f in FACTORS},
                   "n_groups": (int(len(np.unique(groups))) if groups is not None else None)}

    base = Ridge(alpha=alpha, fit_intercept=True).fit(X, y)
    coef = base.coef_.astype(float)

    rng = np.random.default_rng(seed)
    boots = np.zeros((n_boot, p))
    if groups is not None and len(np.unique(groups)) >= 6:
        uniq = np.unique(groups)
        idx_by = {g: np.where(groups == g)[0] for g in uniq}
        for b in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by[g] for g in pick])
            try:
                boots[b] = Ridge(alpha=alpha, fit_intercept=True).fit(X[idx], y[idx]).coef_
            except Exception:
                boots[b] = coef
    else:
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                boots[b] = Ridge(alpha=alpha, fit_intercept=True).fit(X[idx], y[idx]).coef_
            except Exception:
                boots[b] = coef
    # Share of resamples agreeing with the full-sample sign.
    stability = (np.sign(boots) == np.sign(coef)).mean(axis=0)
    stability = np.where(np.isfinite(stability), stability, 0.5)

    # An unstable factor is mostly noise: scale it down smoothly rather than
    # keeping or dropping it on an arbitrary threshold.
    conf = np.clip((stability - 0.5) * 2.0, 0.0, 1.0)
    coef = coef * conf

    norm = np.abs(coef).sum()
    w = coef / norm if norm > 1e-12 else np.ones(p) / p
    eq = np.ones(p) / p
    w = (1 - EQUAL_WEIGHT_SHRINK) * w + EQUAL_WEIGHT_SHRINK * eq * np.sign(w.sum() or 1.0)
    w = np.clip(w, -MAX_ABS_WEIGHT, MAX_ABS_WEIGHT)
    s = np.abs(w).sum()
    if s > 1e-12:
        w = w / s
    return w, {"n": int(n), "raw_coef": coef.tolist(),
               "n_groups": (int(len(np.unique(groups))) if groups is not None else None),
               "stability": {f: float(s_) for f, s_ in zip(FACTORS, stability)}}


def _ic(pred, actual) -> float:
    from scipy import stats
    m = np.isfinite(pred) & np.isfinite(actual)
    if m.sum() < 8 or np.std(pred[m]) == 0:
        return np.nan
    return float(stats.spearmanr(pred[m], actual[m]).statistic)


def _decile_spread(pred, actual, q: int = 5):
    df = pd.DataFrame({"p": pred, "a": actual}).dropna()
    if len(df) < q * 4:
        return (np.nan, np.nan, np.nan)
    try:
        df["b"] = pd.qcut(df["p"].rank(method="first"), q, labels=False)
    except ValueError:
        return (np.nan, np.nan, np.nan)
    top = float(df[df.b == q - 1]["a"].mean())
    bot = float(df[df.b == 0]["a"].mean())
    return (top, bot, top - bot)


# ------------------------------------------------------------------ the run
def run(con, panel: pd.DataFrame | None = None, *, horizons=HORIZONS,
        n_splits: int = 6, n_null: int = N_NULL, store: bool = True,
        seed: int = 0) -> dict:
    """Fit and evaluate. Returns a report dict; optionally persists to the DB."""
    if panel is None:
        from .features import build
        panel = build(con)
    if panel is None or panel.empty:
        return {"status": "no_data"}

    scores = factor_scores(panel)
    run_id = hashlib.sha1(
        f"{datetime.now(timezone.utc).isoformat()}:{len(scores)}".encode()).hexdigest()[:12]

    report = {"run_id": run_id, "status": "ok", "n_rows": int(len(scores)),
              "n_dates": int(scores["date"].nunique()),
              "n_tickers": int(scores["ticker"].nunique()),
              "date_range": [str(scores["date"].min().date()), str(scores["date"].max().date())],
              "factors": FACTORS, "horizons": {}}

    for h in horizons:
        ycol = f"y_{h}"
        if ycol not in scores.columns:
            continue
        sub = scores.dropna(subset=[ycol]).reset_index(drop=True)
        if len(sub) < 200:
            report["horizons"][h] = {"status": "insufficient_data", "n": int(len(sub))}
            continue

        X_all = sub[FACTORS].to_numpy(dtype=float)
        y_all = sub[ycol].to_numpy(dtype=float)
        dates = sub["date"]
        groups_all = sub["ticker"].to_numpy()

        folds, oof_w = [], []
        for k, (tr, te, test_dates) in enumerate(walk_forward(dates, n_splits, h)):
            w, meta = fit_weights(X_all[tr], y_all[tr], seed=seed + k,
                                  groups=groups_all[tr])
            pred = X_all[te] @ w
            ic = _ic(pred, y_all[te])
            top, bot, spread = _decile_spread(pred, y_all[te])
            null_ics = _null_distribution(X_all[tr], y_all[tr], X_all[te], y_all[te],
                                          sub.iloc[te], n_null, seed + 1000 + k,
                                          groups_all[tr])
            folds.append({
                "fold": k, "n_train": int(len(tr)), "n_test": int(len(te)),
                "test_start": str(pd.Timestamp(test_dates[0]).date()),
                "test_end": str(pd.Timestamp(test_dates[-1]).date()),
                "ic": ic, "top_decile_ret": top, "bottom_decile_ret": bot,
                "long_short": spread,
                "hit_rate": float(np.mean(np.sign(pred) == np.sign(y_all[te]))),
                "null_ic_mean": float(np.nanmean(null_ics)) if len(null_ics) else np.nan,
                "null_ic_p95": float(np.nanpercentile(null_ics, 95)) if len(null_ics) else np.nan,
                "weights": {f: float(x) for f, x in zip(FACTORS, w)},
                "stability": meta.get("stability", {}),
            })
            oof_w.append(w)

        if not folds:
            report["horizons"][h] = {"status": "insufficient_folds"}
            continue

        fd = pd.DataFrame(folds)
        mean_ic = float(np.nanmean(fd["ic"]))
        null_p95 = float(np.nanmean(fd["null_ic_p95"]))
        # Final weights: fit on everything, for use in live ranking.
        w_final, meta_final = fit_weights(X_all, y_all, seed=seed, groups=groups_all)
        standalone = {f: _ic(X_all[:, i], y_all) for i, f in enumerate(FACTORS)}

        report["horizons"][h] = {
            "status": "ok",
            "n_obs": int(len(sub)),
            "n_folds": int(len(fd)),
            "mean_ic": mean_ic,
            "ic_std": float(np.nanstd(fd["ic"])),
            "mean_long_short": float(np.nanmean(fd["long_short"])),
            "null_ic_p95": null_p95,
            "beats_null": bool(np.isfinite(mean_ic) and np.isfinite(null_p95)
                               and mean_ic > null_p95),
            "deflated_ic": _deflate(mean_ic, fd["ic"].dropna().to_numpy(), len(horizons)),
            "weights": {f: float(x) for f, x in zip(FACTORS, w_final)},
            "stability": meta_final.get("stability", {}),
            "n_independent_groups": meta_final.get("n_groups"),
            "standalone_ic": standalone,
            "folds": fd.drop(columns=["weights", "stability"]).round(5).to_dict("records"),
            "verdict": _verdict(mean_ic, null_p95, fd),
        }

        if store:
            _store_weights(con, run_id, h, w_final, standalone,
                           meta_final.get("stability", {}))
            _store_folds(con, run_id, h, fd)

    report["overall_verdict"] = _overall(report)
    if store:
        from ..db import set_kv
        set_kv(con, "last_backtest", report)
    return report


def _null_distribution(Xtr, ytr, Xte, yte, te_frame, n_null: int, seed: int,
                       groups=None) -> list[float]:
    """IC obtained when the targets are shuffled within each date.

    This is the bar a real signal has to clear. Comparing IC to zero is not
    enough — with overlapping returns and a fitted model, luck alone produces a
    positive IC surprisingly often.
    """
    if n_null <= 0:
        return []
    rng = np.random.default_rng(seed)
    dates = te_frame["date"].to_numpy()
    out = []
    for _ in range(n_null):
        y_shuf = ytr.copy()
        rng.shuffle(y_shuf)
        w, _ = fit_weights(Xtr, y_shuf, n_boot=8, seed=int(rng.integers(1 << 30)),
                           groups=groups)
        pred = Xte @ w
        # Permute the test targets within each date, preserving cross-sectional shape.
        y_perm = yte.copy()
        for d in np.unique(dates):
            m = dates == d
            idx = np.where(m)[0]
            y_perm[idx] = rng.permutation(yte[idx])
        ic = _ic(pred, y_perm)
        if np.isfinite(ic):
            out.append(ic)
    return out


def _deflate(mean_ic: float, ics: np.ndarray, n_trials: int) -> float:
    """Haircut the IC for the number of configurations effectively tried."""
    if not np.isfinite(mean_ic) or len(ics) < 2:
        return np.nan
    sd = float(np.std(ics, ddof=1)) or 1e-9
    # Expected maximum of n_trials standard normals (Bonferroni-flavoured).
    from math import log, sqrt
    exp_max = sqrt(2 * log(max(n_trials * len(ics), 2)))
    return float(mean_ic - exp_max * sd / sqrt(len(ics)))


def _verdict(mean_ic, null_p95, fd) -> str:
    if not np.isfinite(mean_ic):
        return "Not enough data to measure predictive power."
    pos = int((fd["ic"] > 0).sum())
    n = int(fd["ic"].notna().sum())
    if np.isfinite(null_p95) and mean_ic <= null_p95:
        return (f"Mean IC {mean_ic:.3f} does NOT clear the shuffled-label null "
                f"({null_p95:.3f}). Treat these rankings as unproven — the measured "
                f"skill is within what random chance produces on this data.")
    if pos == n and mean_ic > 0.02:
        return (f"Mean IC {mean_ic:.3f} clears the null and every fold is positive "
                f"({pos}/{n}). Weak but consistent — worth watching, not betting on.")
    return (f"Mean IC {mean_ic:.3f} clears the null but only {pos}/{n} folds are "
            f"positive. Inconsistent; treat as exploratory.")


def _overall(report) -> str:
    hs = [h for h, r in report.get("horizons", {}).items()
          if isinstance(r, dict) and r.get("status") == "ok"]
    if not hs:
        return "No horizon had enough data to evaluate."
    beat = [h for h in hs if report["horizons"][h].get("beats_null")]
    if not beat:
        return ("Neither horizon beat the shuffled-label null. The honest reading is that "
                "this feature set does not predict forward returns on the data available. "
                "The rankings below are still produced, but you should regard them as a "
                "description of where politician and lobbying activity is concentrated — "
                "not as a forecast.")
    return (f"Horizon(s) {sorted(beat)} cleared the null. Signal is weak by construction; "
            f"position sizing should reflect that, and the null margin matters more than "
            f"the raw IC.")


def _store_weights(con, run_id, horizon, w, standalone, stability) -> None:
    as_of = datetime.now(timezone.utc).date().isoformat()
    con.executemany(
        "INSERT OR REPLACE INTO factor_weights "
        "(fit_id,horizon_days,as_of,factor,weight,raw_ic,stability) VALUES (?,?,?,?,?,?,?)",
        [(run_id, horizon, as_of, f, float(w[i]),
          (None if not np.isfinite(standalone.get(f, np.nan)) else float(standalone[f])),
          float(stability.get(f, np.nan)) if np.isfinite(stability.get(f, np.nan)) else None)
         for i, f in enumerate(FACTORS)])


def _store_folds(con, run_id, horizon, fd) -> None:
    cols = ["fold", "test_start", "test_end", "n_test", "ic", "top_decile_ret",
            "bottom_decile_ret", "long_short", "hit_rate", "null_ic_mean", "null_ic_p95"]
    rows = []
    for r in fd.to_dict("records"):
        rows.append((run_id, horizon, r["fold"], r["test_start"], r["test_end"],
                     r["n_test"], _f(r["ic"]), _f(r["top_decile_ret"]),
                     _f(r["bottom_decile_ret"]), _f(r["long_short"]), _f(r["hit_rate"]),
                     _f(r["null_ic_mean"]), _f(r["null_ic_p95"])))
    con.executemany(
        "INSERT OR REPLACE INTO backtest_results (run_id,horizon_days,fold,test_start,"
        "test_end,n_obs,ic,top_decile_ret,bottom_decile_ret,long_short,hit_rate,"
        "null_ic_mean,null_ic_p95) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)


def _f(v):
    return None if v is None or not np.isfinite(v) else float(v)
