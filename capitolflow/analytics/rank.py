"""Produce the ranked short-term and long-term lists.

Each pick carries its factor attribution — which of the seven factors pushed it
up, and by how much — so a ranking is never a bare number you have to trust. It
also carries a confidence that folds in the backtest's null-test result, so when
the model failed to beat random chance, the confidence reported is low and the
dashboard says so in words next to the table.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .backtest import FACTORS
from .features import LONG_HORIZON, SHORT_HORIZON, factor_scores

log = logging.getLogger(__name__)

TOP_N = 10


def latest_weights(con, horizon: int) -> tuple[dict, str | None]:
    rows = con.execute("""
        SELECT factor, weight, as_of, fit_id FROM factor_weights
        WHERE horizon_days=? AND as_of=(SELECT MAX(as_of) FROM factor_weights WHERE horizon_days=?)
    """, (horizon, horizon)).fetchall()
    if not rows:
        return ({f: 1.0 / len(FACTORS) for f in FACTORS}, None)
    return ({r["factor"]: r["weight"] for r in rows}, rows[0]["as_of"])


def _confidence(con, horizon: int) -> tuple[float, str]:
    """Confidence is driven by whether the backtest cleared its null, not by fit quality."""
    from ..db import get_kv
    bt = get_kv(con, "last_backtest") or {}
    h = (bt.get("horizons") or {}).get(str(horizon)) or (bt.get("horizons") or {}).get(horizon)
    if not isinstance(h, dict) or h.get("status") != "ok":
        return (0.1, "No usable backtest for this horizon — treat as descriptive only.")
    ic, p95 = h.get("mean_ic"), h.get("null_ic_p95")
    if ic is None or not np.isfinite(ic):
        return (0.1, "Backtest produced no measurable IC.")
    if not h.get("beats_null"):
        return (0.15, f"Backtest IC {ic:.3f} did not clear the random-chance bar "
                      f"({p95:.3f} if finite). These are activity rankings, not forecasts.")
    margin = ic - (p95 if p95 is not None and np.isfinite(p95) else 0.0)
    conf = float(np.clip(0.3 + margin * 6.0, 0.2, 0.8))
    return (conf, f"Backtest IC {ic:.3f} cleared the random-chance bar by {margin:.3f}. "
                  f"Signal is weak; size accordingly.")


def generate(con, panel: pd.DataFrame | None = None, *, top_n: int = TOP_N,
             store: bool = True) -> dict:
    if panel is None:
        from .features import build
        panel = build(con)
    if panel is None or panel.empty:
        return {"status": "no_data"}

    scores = factor_scores(panel)
    as_of_date = scores["date"].max()
    latest = scores[scores["date"] == as_of_date].copy()
    if latest.empty:
        return {"status": "no_current_slice"}

    out = {"as_of": str(pd.Timestamp(as_of_date).date()),
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "horizons": {}}

    for horizon, label in ((SHORT_HORIZON, "short_term"), (LONG_HORIZON, "long_term")):
        weights, fit_date = latest_weights(con, horizon)
        conf, conf_note = _confidence(con, horizon)

        w = np.array([weights.get(f, 0.0) for f in FACTORS], dtype=float)
        X = latest[FACTORS].to_numpy(dtype=float)
        contrib = X * w                               # per-factor signed contribution
        composite = contrib.sum(axis=1)

        latest[f"score_{horizon}"] = composite
        ranked = latest.assign(_score=composite).sort_values("_score", ascending=False)

        # Historical spread between top and bottom buckets, used to translate a
        # composite score into an expected-return figure with the right magnitude.
        scale = _expected_scale(con, horizon)

        picks = []
        n = len(ranked)
        for i, (_, row) in enumerate(ranked.head(top_n).iterrows()):
            idx = latest.index.get_loc(row.name)
            attrib = {f: float(contrib[idx, j]) for j, f in enumerate(FACTORS)}
            pct = 1.0 - (i / max(n - 1, 1))
            picks.append({
                "rank": i + 1, "ticker": row["ticker"],
                "score": float(row["_score"]), "score_pctile": float(pct),
                "expected_excess": float(row["_score"] * scale) if np.isfinite(scale) else None,
                "confidence": conf,
                "attribution": attrib,
                "rationale": _rationale(attrib),
            })

        out["horizons"][label] = {
            "horizon_days": horizon,
            "weights": {f: float(weights.get(f, 0.0)) for f in FACTORS},
            "weights_fitted_on": fit_date,
            "confidence": conf,
            "confidence_note": conf_note,
            "n_candidates": int(n),
            "picks": picks,
            "bottom_picks": _bottom(ranked, contrib, latest, top_n),
        }

        if store:
            _store(con, out["as_of"], horizon, picks)

    return out


def _bottom(ranked, contrib, latest, top_n) -> list[dict]:
    """The other end of the ranking. Useful as a sanity check: if the bottom of
    the list outperforms the top, the model is inverted or meaningless."""
    rows = []
    tail = ranked.tail(top_n).iloc[::-1]
    for i, (_, row) in enumerate(tail.iterrows()):
        idx = latest.index.get_loc(row.name)
        rows.append({"rank": i + 1, "ticker": row["ticker"], "score": float(row["_score"]),
                     "attribution": {f: float(contrib[idx, j]) for j, f in enumerate(FACTORS)}})
    return rows


def _expected_scale(con, horizon: int) -> float:
    """Map a unitless composite score onto a return scale, from realised fold spreads."""
    row = con.execute(
        "SELECT AVG(long_short) s, COUNT(*) n FROM backtest_results WHERE horizon_days=?",
        (horizon,)).fetchone()
    if not row or not row["n"] or row["s"] is None:
        return np.nan
    # long_short is the spread between top and bottom quintile; a composite of
    # +1 sd corresponds to roughly a quarter of that spread.
    return float(row["s"]) / 4.0


_LABELS = {
    "politician_flow": "net congressional buying",
    "politician_conviction": "breadth and size of congressional positions",
    "politician_freshness": "recency of disclosures",
    "lobbying": "federal lobbying spend",
    "earnings": "earnings surprise history",
    "events": "current-events regime",
    "momentum": "price momentum",
}


def _rationale(attrib: dict) -> str:
    items = sorted(attrib.items(), key=lambda kv: -abs(kv[1]))
    parts = []
    for f, v in items[:3]:
        if abs(v) < 1e-6:
            continue
        parts.append(f"{'+' if v > 0 else '−'} {_LABELS.get(f, f)}")
    return ", ".join(parts) if parts else "no dominant factor"


def _store(con, as_of: str, horizon: int, picks: list[dict]) -> None:
    con.executemany(
        "INSERT OR REPLACE INTO predictions (as_of,horizon_days,ticker,rank,score,"
        "score_pctile,expected_excess,confidence,attribution,rationale) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(as_of, horizon, p["ticker"], p["rank"], p["score"], p["score_pctile"],
          p["expected_excess"], p["confidence"], json.dumps(p["attribution"]),
          p["rationale"]) for p in picks])


def evaluate_past_predictions(con, horizon: int) -> pd.DataFrame:
    """Score previously-stored predictions against what actually happened.

    This is the honest scoreboard: it compares what the model said months ago to
    the realised outcome, with no refitting involved.
    """
    from ..config import SETTINGS
    preds = pd.read_sql_query(
        "SELECT as_of, ticker, rank, score FROM predictions WHERE horizon_days=?",
        con, params=[horizon])
    if preds.empty:
        return preds
    px = pd.read_sql_query("SELECT ticker, date, adj_close FROM prices", con)
    if px.empty:
        return pd.DataFrame()
    px["date"] = pd.to_datetime(px["date"])
    wide = px.pivot_table(index="date", columns="ticker", values="adj_close",
                          aggfunc="last").sort_index()
    bench = SETTINGS.benchmark
    if bench not in wide.columns:
        return pd.DataFrame()
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(full).ffill()

    rows = []
    for r in preds.itertuples(index=False):
        d0 = pd.Timestamp(r.as_of)
        d1 = d0 + pd.Timedelta(days=int(horizon * 1.45))     # calendar approx of trading days
        if d1 > wide.index[-1] or r.ticker not in wide.columns:
            continue
        def px_at(t, d):
            i = wide.index.searchsorted(d, side="left")
            return float(wide[t].iloc[i]) if i < len(wide) else np.nan
        p0, p1 = px_at(r.ticker, d0), px_at(r.ticker, d1)
        b0, b1 = px_at(bench, d0), px_at(bench, d1)
        if any(not np.isfinite(v) or v <= 0 for v in (p0, p1, b0, b1)):
            continue
        rows.append({"as_of": r.as_of, "ticker": r.ticker, "rank": r.rank,
                     "score": r.score,
                     "realised_excess": (p1 / p0 - 1) - (b1 / b0 - 1)})
    return pd.DataFrame(rows)
