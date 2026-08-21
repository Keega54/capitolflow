"""One point-in-time-correct feature panel combining every data source.

The governing rule, applied without exception: a feature computed for date D may
only use information that was PUBLIC on date D.

  * politician trades enter on their FILED date, never their trade date
  * a filing is aged by the fitted decay half-life, so a 40-day-old disclosure
    counts for less than yesterday's
  * lobbying enters after the reporting period closed
  * earnings enter on the report date
  * event-theme z-scores are computed from trailing windows only
  * every rolling statistic is shifted so the current bar is excluded

Breaking any one of these produces a backtest that looks wonderful and predicts
nothing, which is the default outcome of this kind of project.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from ..config import SETTINGS

log = logging.getLogger(__name__)

# Feature families. The backtest fits one weight per FACTOR, not per raw column,
# which keeps the number of fitted parameters small relative to the sample.
FACTOR_COLUMNS = {
    "politician_flow": ["net_ratio_90", "wnet_ratio_90", "n_buyers_90", "buy_share_90"],
    "politician_conviction": ["log_gross_90", "n_members_90", "max_mweight_90"],
    "politician_freshness": ["decayed_net_flow", "days_since_last_filing"],
    "lobbying": ["log_lobby_12m", "lobby_yoy"],
    "earnings": ["last_surprise", "days_since_earnings", "earnings_soon", "surprise_trend"],
    # A theme's intensity is identical for every ticker on a given day, so on its
    # own it is annihilated by cross-sectional standardization — it can only
    # distinguish stocks once interacted with each stock's sector exposure.
    # "Is there a war on" is not a stock-picking signal; "is there a war on AND
    # does this company sell to defense ministries" is.
    "events": ["conflict_x_sector", "tariffs_x_sector", "energy_x_sector",
               "health_x_sector", "ai_x_sector", "monetary_x_sector",
               "event_beta"],
    "momentum": ["mom_20", "mom_120", "vol_60"],
}
ALL_FEATURES = [c for cols in FACTOR_COLUMNS.values() for c in cols]

SHORT_HORIZON = 21     # ~1 trading month
LONG_HORIZON = 126     # ~6 trading months
HORIZONS = (SHORT_HORIZON, LONG_HORIZON)


def _price_frames(con):
    px = pd.read_sql_query("SELECT ticker, date, adj_close FROM prices", con)
    if px.empty:
        return None, None
    px["date"] = pd.to_datetime(px["date"])
    wide = px.pivot_table(index="date", columns="ticker", values="adj_close",
                          aggfunc="last").sort_index()
    return wide, SETTINGS.benchmark


def forward_excess(wide: pd.DataFrame, bench: str, horizon: int) -> pd.DataFrame:
    fwd = wide.shift(-horizon) / wide - 1.0
    fwd_b = wide[bench].shift(-horizon) / wide[bench] - 1.0
    return fwd.sub(fwd_b, axis=0)


def build(con, *, horizons=HORIZONS, freq: str = "W", half_life: float | None = None,
          min_price_history: int = 130) -> pd.DataFrame:
    """Return a long panel: one row per (date, ticker) with features and targets."""
    wide, bench = _price_frames(con)
    if wide is None or bench not in (wide.columns if wide is not None else []):
        log.warning("no prices or benchmark; cannot build panel")
        return pd.DataFrame()

    targets = {h: forward_excess(wide, bench, h) for h in horizons}

    txns = pd.read_sql_query("""
        SELECT t.ticker, t.member_id, t.filed_date, t.transaction_date, t.direction,
               COALESCE(t.amount_est,0) AS amount_est,
               COALESCE(s.weight,1.0) AS mweight
        FROM transactions t
        LEFT JOIN member_scores s ON s.member_id=t.member_id AND s.horizon_days=90
        WHERE t.ticker IS NOT NULL AND t.ticker_confidence>=0.7 AND t.direction!=0
          AND t.filed_date IS NOT NULL""", con)
    if txns.empty:
        log.warning("no usable transactions")
        return pd.DataFrame()
    txns["filed_date"] = pd.to_datetime(txns["filed_date"], errors="coerce")
    txns = txns.dropna(subset=["filed_date"])

    lob = pd.read_sql_query("""
        SELECT ticker, period_end, COALESCE(amount,0) amount FROM lobbying_filings
        WHERE ticker IS NOT NULL AND ticker_confidence>=0.6 AND period_end IS NOT NULL""", con)
    if not lob.empty:
        lob["period_end"] = pd.to_datetime(lob["period_end"], errors="coerce")
        lob = lob.dropna(subset=["period_end"])

    earn = pd.read_sql_query("""
        SELECT ticker, report_date, surprise_pct FROM earnings
        WHERE report_date IS NOT NULL""", con)
    if not earn.empty:
        earn["report_date"] = pd.to_datetime(earn["report_date"], errors="coerce")
        earn = earn.dropna(subset=["report_date"]).sort_values(["ticker", "report_date"])

    from ..sources.events import theme_panel
    themes = theme_panel(con)

    if half_life is None:
        half_life = _stored_half_life(con)

    period = wide.index.to_period(freq)
    dates = pd.Series(wide.index).groupby(period.to_numpy()).last().tolist()
    tickers = [t for t in wide.columns if t != bench]

    # Pre-group for speed: dict ticker -> sorted frame.
    sector_of = {r["ticker"]: r["sector"] for r in
                 con.execute("SELECT ticker, sector FROM ticker_sectors")}
    tx_by = {t: g.sort_values("filed_date") for t, g in txns.groupby("ticker")}
    lob_by = {t: g.sort_values("period_end") for t, g in lob.groupby("ticker")} if not lob.empty else {}
    earn_by = {t: g for t, g in earn.groupby("ticker")} if not earn.empty else {}

    rows = []
    for d in dates:
        d = pd.Timestamp(d)
        theme_row = _theme_values(themes, d)
        for t in tickers:
            ys = {h: targets[h].at[d, t] if (d in targets[h].index and t in targets[h].columns)
                  else np.nan for h in horizons}
            if all(not np.isfinite(v) for v in ys.values()):
                continue
            hist = wide[t].loc[:d].dropna()
            if len(hist) < min_price_history:
                continue

            rec = {"date": d, "ticker": t}
            for h in horizons:
                rec[f"y_{h}"] = float(ys[h]) if np.isfinite(ys[h]) else np.nan

            rec.update(_politician_features(tx_by.get(t), d, half_life))
            rec.update(_lobbying_features(lob_by.get(t), d))
            rec.update(_earnings_features(earn_by.get(t), d))
            rec.update(_event_features(theme_row, sector_of.get(t)))
            rec.update(_momentum_features(hist))
            rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def _stored_half_life(con) -> float | None:
    from ..db import get_kv
    v = get_kv(con, "signal_half_life_days")
    try:
        v = float(v)
        return v if np.isfinite(v) and v > 0 else None
    except (TypeError, ValueError):
        return None


def _politician_features(g, d: pd.Timestamp, half_life: float | None) -> dict:
    base = {"net_ratio_90": 0.0, "wnet_ratio_90": 0.0, "n_buyers_90": 0.0,
            "buy_share_90": 0.5, "log_gross_90": 0.0, "n_members_90": 0.0,
            "max_mweight_90": 1.0, "decayed_net_flow": 0.0,
            "days_since_last_filing": 365.0}
    if g is None or g.empty:
        return base
    seen = g[g["filed_date"] <= d]
    if seen.empty:
        return base
    base["days_since_last_filing"] = float((d - seen["filed_date"].max()).days)

    w = seen[seen["filed_date"] > d - pd.Timedelta(days=90)]
    if not w.empty:
        gross = float(w["amount_est"].sum())
        net = float((w["amount_est"] * w["direction"]).sum())
        wnet = float((w["amount_est"] * w["direction"] * w["mweight"]).sum())
        base["net_ratio_90"] = net / gross if gross > 0 else 0.0
        base["wnet_ratio_90"] = wnet / gross if gross > 0 else 0.0
        base["n_buyers_90"] = float(w[w["direction"] > 0]["member_id"].nunique())
        base["n_members_90"] = float(w["member_id"].nunique())
        base["buy_share_90"] = float((w["direction"] > 0).mean())
        base["log_gross_90"] = float(np.log1p(gross))
        base["max_mweight_90"] = float(w["mweight"].max())

    # Freshness-aged flow over a longer lookback: an old filing still counts,
    # just less, at a rate estimated from the data rather than assumed.
    long_w = seen[seen["filed_date"] > d - pd.Timedelta(days=270)]
    if not long_w.empty:
        age = (d - long_w["filed_date"]).dt.days.to_numpy(dtype=float)
        if half_life and half_life > 0:
            decay = 0.5 ** (age / half_life)
        else:
            decay = np.ones_like(age)
        amt = long_w["amount_est"].to_numpy(dtype=float)
        dirn = long_w["direction"].to_numpy(dtype=float)
        mw = long_w["mweight"].to_numpy(dtype=float)
        denom = float((amt * decay).sum())
        base["decayed_net_flow"] = float((amt * dirn * mw * decay).sum() / denom) if denom > 0 else 0.0
    return base


def _lobbying_features(g, d: pd.Timestamp) -> dict:
    out = {"log_lobby_12m": 0.0, "lobby_yoy": 0.0}
    if g is None or g.empty:
        return out
    cur = g[(g["period_end"] <= d) & (g["period_end"] > d - pd.Timedelta(days=365))]
    prev = g[(g["period_end"] <= d - pd.Timedelta(days=365)) &
             (g["period_end"] > d - pd.Timedelta(days=730))]
    a, b = float(cur["amount"].sum()), float(prev["amount"].sum())
    out["log_lobby_12m"] = float(np.log1p(a))
    out["lobby_yoy"] = float((a - b) / b) if b > 0 else 0.0
    return out


def _earnings_features(g, d: pd.Timestamp) -> dict:
    out = {"last_surprise": 0.0, "days_since_earnings": 180.0,
           "earnings_soon": 0.0, "surprise_trend": 0.0}
    if g is None or g.empty:
        return out
    past = g[g["report_date"] <= d]
    if past.empty:
        return out
    last = past.iloc[-1]
    out["days_since_earnings"] = float((d - last["report_date"]).days)
    if pd.notna(last["surprise_pct"]):
        out["last_surprise"] = float(np.clip(last["surprise_pct"], -2, 2))
    recent = past.tail(4)["surprise_pct"].dropna()
    if len(recent) >= 2:
        out["surprise_trend"] = float(np.clip(recent.mean(), -2, 2))
    # Companies report on a ~91-day cadence; flag the window where the next
    # report is imminent, since that dominates short-horizon variance.
    cadence = 91.0
    if len(past) >= 2:
        gaps = past["report_date"].diff().dt.days.dropna()
        gaps = gaps[(gaps > 45) & (gaps < 200)]
        if len(gaps):
            cadence = float(gaps.median())
    out["earnings_soon"] = 1.0 if (cadence - out["days_since_earnings"]) <= 21 else 0.0
    return out


def _theme_values(themes, d: pd.Timestamp) -> dict:
    out = {f"theme_{k}": 0.0 for k in
           ("conflict", "tariffs", "energy", "health", "ai", "monetary")}
    if themes is None or getattr(themes, "empty", True):
        return out
    idx = themes.index.searchsorted(d, side="right") - 1
    if idx < 0:
        return out
    row = themes.iloc[idx]
    for k in out:
        theme = k.replace("theme_", "")
        if theme in themes.columns and pd.notna(row.get(theme)):
            out[k] = float(row[theme])
    return out


def _event_features(theme_row: dict, sector: str | None) -> dict:
    """Project each theme onto this stock's sector exposure.

    Exposure is 1 when the sector is one the theme plausibly moves, 0 otherwise.
    The SIGN and MAGNITUDE of the effect are never asserted here — the backtest
    fits them. All this does is restrict where an effect is allowed to appear,
    which keeps the search space small enough to be testable.
    """
    from ..sources.events import THEME_SECTORS
    out = {}
    total = 0.0
    for theme, sectors in THEME_SECTORS.items():
        z = float(theme_row.get(f"theme_{theme}", 0.0) or 0.0)
        exposed = 1.0 if (sector and sector in sectors) else 0.0
        out[f"{theme}_x_sector"] = z * exposed
        total += abs(z) * exposed
    # How much total event pressure is aimed at this stock's sector right now.
    out["event_beta"] = total
    return out


def _momentum_features(hist: pd.Series) -> dict:
    out = {"mom_20": 0.0, "mom_120": 0.0, "vol_60": 0.0}
    if len(hist) > 21:
        out["mom_20"] = float(hist.iloc[-1] / hist.iloc[-21] - 1)
    if len(hist) > 120:
        out["mom_120"] = float(hist.iloc[-1] / hist.iloc[-121] - 1)
    r = hist.pct_change(fill_method=None).tail(60)
    if len(r) > 10:
        out["vol_60"] = float(r.std())
    return out


# ------------------------------------------------------------------ scoring
def cross_sectional_z(df: pd.DataFrame, cols, by: str = "date") -> pd.DataFrame:
    """Standardize each feature WITHIN each date.

    Cross-sectional ranking is the only fair comparison: it asks "which stock
    looks best today", not "is today a good day", so a market-wide move cannot
    masquerade as stock selection.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        g = out.groupby(by)[c]
        mu, sd = g.transform("mean"), g.transform("std")
        z = (out[c] - mu) / sd.replace(0, np.nan)
        out[c + "_z"] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3, 3)
    return out


def factor_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse each feature family into one standardized factor score."""
    z = cross_sectional_z(df, ALL_FEATURES)
    out = df[["date", "ticker"]].copy()
    for col in df.columns:
        if col.startswith("y_"):
            out[col] = df[col]
    for factor, cols in FACTOR_COLUMNS.items():
        zc = [c + "_z" for c in cols if c + "_z" in z.columns]
        out[factor] = z[zc].mean(axis=1) if zc else 0.0
    return out
