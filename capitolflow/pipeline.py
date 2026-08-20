"""Orchestration: one function per stage, plus `refresh()` which runs them in order.

Every stage is independently re-runnable and safe to interrupt: filings are keyed
by document id, transactions by a deterministic hash, prices by (ticker, date).
"""
from __future__ import annotations
import json, logging, traceback
from datetime import date, timedelta

from .config import SEC_TICKERS, SETTINGS
from . import db
from .sources import aggregator, house, lobbying, members, prices, senate
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


def compute_analytics(con, *, event_study: bool = True) -> dict:
    from .analytics import accuracy, eventstudy, returns
    out = {}
    r = returns.compute_trade_returns(con)
    out["trade_returns"] = returns.store_trade_returns(con, r)
    s = accuracy.compute_member_scores(con)
    out["member_scores"] = accuracy.store_member_scores(con, s)
    if event_study:
        try:
            con.execute("DELETE FROM event_studies WHERE scope='txn'")
            es = eventstudy.study_trades(con)
            out["event_studies"] = len(es)
        except Exception as e:
            log.warning("event study failed: %s", e)
            out["event_studies"] = 0
    return out


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
    report["analytics"] = compute_analytics(con)
    if with_model:
        report["model"] = train_model(con)
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
        "latest_trade_date": q("SELECT MAX(transaction_date) FROM transactions"),
        "latest_filed_date": q("SELECT MAX(filed_date) FROM filings"),
    }
