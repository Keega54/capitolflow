"""Orchestration: one function per stage, plus `refresh()` which runs them in order.

Every stage is independently re-runnable and safe to interrupt: filings are keyed
by document id, transactions by a deterministic hash, prices by (ticker, date).
"""
from __future__ import annotations
import json, logging, os, traceback
from datetime import date, timedelta

from .config import SEC_TICKERS, SETTINGS
from . import db
from .sources import (aggregator, earnings, events, house, lobbying, members,
                      prices, senate)
from .sources.house import Budget
from .util.http import get_bytes, make_session

log = logging.getLogger(__name__)


def sync_reference(con) -> dict:
    """Rosters, committees, and the SEC ticker map the resolver depends on."""
    s = make_session()
    out = {}
    out["legislators"] = members.sync_legislators(con, s)
    out["committees"] = members.sync_committees(con, s)
    out["executive"] = members.seed_executive(con)
    try:
        blob = get_bytes(s, SEC_TICKERS, suffix=".json", max_age_s=7 * 86400,
                         headers={"User-Agent": SETTINGS.user_agent})
        SETTINGS.ensure_dirs()
        (SETTINGS.data_dir / "sec_tickers.json").write_bytes(blob)
        from .util.tickers import default_resolver
        default_resolver.cache_clear()
        out["sec_tickers"] = len(json.loads(blob))
    except Exception as e:
        log.warning("SEC ticker map unavailable: %s", e)
        out["sec_tickers"] = 0
    return out


def ingest_disclosures(con, *, years=None, incremental: bool = True) -> dict:
    this_year = date.today().year
    years = years or ([this_year, this_year - 1] if incremental
                      else list(range(SETTINGS.start_year, this_year + 1)))
    res = {"house": {}, "senate": {}, "aggregator": {}, "budget": None}
    # One allowance for the whole run. A cold-start backfill therefore spans
    # several scheduled runs and each one finishes inside the job time limit.
    budget = Budget(SETTINGS.max_new_filings_per_run)

    # Newest year first, so a partially-filled database is always current rather
    # than stuck in 2014 while the backfill grinds forward.
    for yr in sorted(years, reverse=True):
        if budget.exhausted:
            res["house"][yr] = {"skipped": "run budget exhausted"}
            continue
        rid = db.start_run(con, f"house:{yr}")
        try:
            f, t = house.ingest_year(con, yr, budget=budget)
            db.finish_run(con, rid, "ok", f, t)
            res["house"][yr] = {"filings": f, "transactions": t}
        except Exception as e:
            db.finish_run(con, rid, "error", note=traceback.format_exc())
            log.error("house %s failed: %s", yr, e)
            res["house"][yr] = {"error": str(e)}

    rid = db.start_run(con, "senate")
    try:
        start = (f"01/01/{min(years)}" if not incremental
                 else (date.today() - timedelta(days=120)).strftime("%m/%d/%Y"))
        f, t = senate.ingest(con, start_date=start,
                             limit=max(budget.remaining, 50))
        db.finish_run(con, rid, "ok", f, t)
        res["senate"] = {"filings": f, "transactions": t}
    except Exception as e:
        db.finish_run(con, rid, "error", note=traceback.format_exc())
        log.error("senate failed: %s", e)
        res["senate"] = {"error": str(e)}

    for name, fn in (("quiver", aggregator.ingest_quiver), ("fmp", aggregator.ingest_fmp)):
        try:
            f, t = fn(con)
            if f or t:
                res["aggregator"][name] = {"filings": f, "transactions": t}
        except Exception as e:
            log.warning("%s failed: %s", name, e)
    res["budget"] = {"total": budget.total, "used": budget.used,
                     "remaining": budget.remaining}
    try:
        res["reconciliation"] = aggregator.reconcile(con)
    except Exception as e:
        log.warning("reconcile failed: %s", e)
    return res


def ingest_lobbying(con, years=None) -> dict:
    rid = db.start_run(con, "lobbying")
    try:
        n = lobbying.ingest(con, years=years)
        db.finish_run(con, rid, "ok", n, 0)
        return {"filings": n}
    except Exception as e:
        db.finish_run(con, rid, "error", note=traceback.format_exc())
        log.error("lobbying failed: %s", e)
        return {"error": str(e)}


def sync_prices(con, max_tickers: int | None = None) -> dict:
    rid = db.start_run(con, "prices")
    try:
        n = prices.sync_prices(con, max_tickers=max_tickers)
        db.finish_run(con, rid, "ok", 0, n)
        return {"rows": n}
    except Exception as e:
        db.finish_run(con, rid, "error", note=traceback.format_exc())
        return {"error": str(e)}


def ingest_context(con) -> dict:
    """Earnings, current-events themes, and sector labels."""
    out = {}
    rid = db.start_run(con, "earnings")
    try:
        n = earnings.sync(con, max_tickers=int(os.environ.get("CAPITOLFLOW_MAX_EARNINGS", "300")))
        db.finish_run(con, rid, "ok", 0, n)
        out["earnings"] = n
    except Exception as e:
        db.finish_run(con, rid, "error", note=traceback.format_exc())
        out["earnings"] = {"error": str(e)}
    rid = db.start_run(con, "events")
    try:
        out["event_index"] = events.sync(con)
        out["sectors"] = events.sync_sectors(con)
        db.finish_run(con, rid, "ok", 0, out["event_index"])
    except Exception as e:
        db.finish_run(con, rid, "error", note=traceback.format_exc())
        out["events"] = {"error": str(e)}
    return out


def compute_analytics(con, *, event_study: bool = True) -> dict:
    from .analytics import accuracy, eventstudy, returns, timing
    out = {}
    r = returns.compute_trade_returns(con)
    out["trade_returns"] = returns.store_trade_returns(con, r)
    s = accuracy.compute_member_scores(con)
    out["member_scores"] = accuracy.store_member_scores(con, s)

    # Disclosure-lag decomposition, and the fitted decay the features depend on.
    try:
        t = timing.decompose(con)
        out["trade_timing"] = timing.store(con, t)
        out["timing_summary"] = timing.summary(t)
        # Stored under its own key so the dashboard never depends on the shape
        # of the last full-refresh report to find it.
        db.set_kv(con, "timing_summary", out["timing_summary"])
        decay = timing.fit_decay(con)
        hl = decay.get("half_life_days")
        if hl is not None and isinstance(hl, float) and hl == hl and hl > 0:
            db.set_kv(con, "signal_half_life_days", hl)
        db.set_kv(con, "signal_decay", decay)
        out["signal_half_life_days"] = hl
    except Exception as e:
        log.warning("timing analysis failed: %s", e)
    if event_study:
        try:
            con.execute("DELETE FROM event_studies WHERE scope='txn'")
            es = eventstudy.study_trades(con)
            out["event_studies"] = len(es)
        except Exception as e:
            log.warning("event study failed: %s", e)
            out["event_studies"] = 0
    return out


def run_backtest(con, *, n_splits: int = 6, n_null: int = 120) -> dict:
    """Fit factor weights and measure them against a shuffled-label null."""
    from .analytics import backtest, features
    rid = db.start_run(con, "backtest")
    try:
        panel = features.build(con)
        rep = backtest.run(con, panel=panel, n_splits=n_splits, n_null=n_null)
        db.finish_run(con, rid, "ok", 0, rep.get("n_rows", 0))
        return rep
    except Exception as e:
        db.finish_run(con, rid, "error", note=traceback.format_exc())
        log.error("backtest failed: %s", e)
        return {"status": "error", "error": str(e)}


def make_predictions(con, top_n: int = 10) -> dict:
    from .analytics import features, rank
    try:
        return rank.generate(con, panel=features.build(con), top_n=top_n)
    except Exception as e:
        log.error("prediction failed: %s", e)
        return {"status": "error", "error": str(e)}


def train_model(con, model_out: str | None = None) -> dict:
    from .analytics import model
    try:
        return model.train(con, model_out=model_out or str(SETTINGS.data_dir / "model.pkl"))
    except Exception as e:
        log.error("model training failed: %s", e)
        return {"status": "error", "error": str(e)}


def refresh(con, *, full: bool = False, with_model: bool = False) -> dict:
    """The single entry point a scheduler should call."""
    report = {"started_at": date.today().isoformat(), "full": full}
    report["reference"] = sync_reference(con)
    report["disclosures"] = ingest_disclosures(con, incremental=not full)
    report["lobbying"] = ingest_lobbying(con)
    report["prices"] = sync_prices(con)
    report["context"] = ingest_context(con)
    report["analytics"] = compute_analytics(con)
    if with_model:
        report["backtest"] = run_backtest(con)
        report["predictions"] = make_predictions(con)
    report["counts"] = health(con)
    db.set_kv(con, "last_refresh", report)
    return report


def health(con) -> dict:
    q = lambda s: con.execute(s).fetchone()[0]
    return {
        "members": q("SELECT COUNT(*) FROM members"),
        "filings": q("SELECT COUNT(*) FROM filings"),
        "filings_failed": q("SELECT COUNT(*) FROM filings WHERE parse_status='failed'"),
        "transactions": q("SELECT COUNT(*) FROM transactions"),
        "transactions_with_ticker": q("SELECT COUNT(*) FROM transactions WHERE ticker IS NOT NULL"),
        "transactions_unlinked_member": q("SELECT COUNT(*) FROM transactions WHERE member_id IS NULL"),
        "distinct_tickers": q("SELECT COUNT(DISTINCT ticker) FROM transactions WHERE ticker IS NOT NULL"),
        "price_rows": q("SELECT COUNT(*) FROM prices"),
        "lobbying_filings": q("SELECT COUNT(*) FROM lobbying_filings"),
        "earnings_rows": q("SELECT COUNT(*) FROM earnings"),
        "event_index_rows": q("SELECT COUNT(*) FROM event_index"),
        "predictions": q("SELECT COUNT(*) FROM predictions"),
        "latest_trade_date": q("SELECT MAX(transaction_date) FROM transactions"),
        "latest_filed_date": q("SELECT MAX(filed_date) FROM filings"),
    }
