"""Optional third-party feeds, used strictly as a cross-check and gap-filler.

Primary truth is always the filing itself. These feeds exist to (a) surface
filings our scrapers missed and (b) flag rows where a compiled source disagrees
with our parse, which is the cheapest possible parser regression test.
Every adapter degrades to a no-op when its key is absent.
"""
from __future__ import annotations
import json, logging
from datetime import date, timedelta

from ..config import SETTINGS
from ..db import txn_id, upsert, upsert_many
from ..util.dates import delay_days, iso
from ..util.http import get_json, make_session
from ..util.tickers import default_resolver
from .members import MemberMatcher

log = logging.getLogger(__name__)

FINNHUB_HOUSE = "https://finnhub.io/api/v1/stock/congressional-trading?symbol={sym}&from={f}&to={t}&token={k}"
QUIVER_CONGRESS = "https://api.quiverquant.com/beta/bulk/congresstrading"
FMP_SENATE = "https://financialmodelingprep.com/api/v4/senate-trading-rss-feed?page={p}&apikey={k}"
FMP_HOUSE = "https://financialmodelingprep.com/api/v4/senate-disclosure-rss-feed?page={p}&apikey={k}"


def _norm_row(r: dict) -> dict | None:
    """Normalize a vendor row into our transaction shape."""
    tdate = iso(r.get("transactionDate") or r.get("transaction_date") or r.get("TransactionDate")
                or r.get("date"))
    if not tdate:
        return None
    raw_type = str(r.get("type") or r.get("transactionType") or r.get("Transaction") or "").lower()
    if "purchase" in raw_type or raw_type.strip() in ("p", "buy"):
        ttype, direction = "buy", 1
    elif "sale" in raw_type or raw_type.strip() in ("s", "sell"):
        ttype = "sell_partial" if "partial" in raw_type else ("sell_full" if "full" in raw_type else "sell")
        direction = -1
    elif "exchange" in raw_type:
        ttype, direction = "exchange", 0
    else:
        ttype, direction = "other", 0
    from ..util.amounts import parse_amount
    lo, hi, est = parse_amount(r.get("amount") or r.get("range") or r.get("Range"))
    if lo is None:
        lo = r.get("amountFrom") or r.get("low")
        hi = r.get("amountTo") or r.get("high")
        from ..util.amounts import geo_mid
        est = geo_mid(lo, hi)
    return {
        "ticker": (r.get("symbol") or r.get("ticker") or r.get("Ticker") or "").upper() or None,
        "asset_name_raw": r.get("assetDescription") or r.get("asset_description")
                          or r.get("assetName") or r.get("Company") or r.get("symbol") or "",
        "transaction_date": tdate,
        "filed_date": iso(r.get("filingDate") or r.get("disclosureDate") or r.get("dateRecieved")
                          or r.get("Filed")),
        "owner": (r.get("owner") or "self").lower() or "self",
        "txn_type": ttype, "direction": direction,
        "amount_low": lo, "amount_high": hi, "amount_est": est,
        "member_name": r.get("name") or r.get("representative") or r.get("Representative")
                       or f"{r.get('firstName','')} {r.get('lastName','')}".strip(),
        "chamber": (r.get("chamber") or r.get("Chamber") or "").lower() or None,
        "raw": json.dumps(r)[:4000],
    }


def ingest_quiver(con, session=None) -> tuple[int, int]:
    if not SETTINGS.quiver_key:
        return (0, 0)
    s = session or make_session({"Authorization": f"Bearer {SETTINGS.quiver_key}"})
    try:
        rows = get_json(s, QUIVER_CONGRESS, max_age_s=3600)
    except Exception as e:
        log.warning("quiver fetch failed: %s", e)
        return (0, 0)
    return _store(con, rows, "quiver")


def ingest_fmp(con, session=None, pages: int = 5) -> tuple[int, int]:
    if not SETTINGS.fmp_key:
        return (0, 0)
    s = session or make_session()
    rows = []
    for tmpl in (FMP_SENATE, FMP_HOUSE):
        for p in range(pages):
            try:
                batch = get_json(s, tmpl.format(p=p, k=SETTINGS.fmp_key), max_age_s=3600)
            except Exception as e:
                log.warning("fmp page %s failed: %s", p, e)
                break
            if not batch:
                break
            rows += batch
    return _store(con, rows, "fmp")


def _store(con, rows, source: str) -> tuple[int, int]:
    matcher = MemberMatcher(con)
    resolver = default_resolver()
    filing_rows, txn_rows = {}, []
    for raw in rows or []:
        r = _norm_row(raw)
        if not r:
            continue
        member_id = matcher.match(r["member_name"], chamber=r.get("chamber"))
        fid = f"{source}:{member_id or r['member_name']}:{r['filed_date']}"
        filing_rows[fid] = {
            "filing_id": fid, "source": "aggregator", "doc_id": fid,
            "member_id": member_id, "filer_name_raw": r["member_name"],
            "filing_type": "ptr", "filing_year": int((r["filed_date"] or "0000")[:4]) or None,
            "filed_date": r["filed_date"], "url": None, "doc_format": "api",
            "parse_status": "ok", "parse_note": f"vendor:{source}", "fetched_at": None,
        }
        tic = r["ticker"]
        conf = 1.0 if tic else 0.0
        if not tic:
            tic, conf, _ = resolver.resolve(r["asset_name_raw"])
        _, _, atype = resolver.resolve(r["asset_name_raw"])
        txn_rows.append({
            "txn_id": txn_id(fid, r["asset_name_raw"], r["transaction_date"], r["txn_type"],
                             r["amount_low"], r["owner"]),
            "filing_id": fid, "member_id": member_id,
            "transaction_date": r["transaction_date"], "notification_date": None,
            "filed_date": r["filed_date"],
            "filing_delay_days": delay_days(r["transaction_date"], r["filed_date"]),
            "owner": r["owner"], "asset_name_raw": r["asset_name_raw"],
            "ticker": tic, "ticker_confidence": conf, "asset_type": atype,
            "txn_type": r["txn_type"], "direction": r["direction"],
            "amount_low": r["amount_low"], "amount_high": r["amount_high"],
            "amount_est": r["amount_est"], "comment": None, "cap_gains_over_200": None,
            "source": source, "raw": r["raw"],
        })
    upsert_many(con, "filings", list(filing_rows.values()), mode="IGNORE")
    n = upsert_many(con, "transactions", txn_rows, mode="IGNORE")
    return (len(filing_rows), n)


def reconcile(con) -> dict:
    """Compare vendor rows against our own parses on (member, ticker, trade date).

    Returns counts of agreement, vendor-only and ours-only rows. A spike in
    'vendor_only' means a scraper is silently dropping filings.
    """
    q = """
    WITH ours AS (
      SELECT member_id, ticker, transaction_date, direction FROM transactions
      WHERE source IN ('house','senate') AND ticker IS NOT NULL AND member_id IS NOT NULL
    ), theirs AS (
      SELECT member_id, ticker, transaction_date, direction FROM transactions
      WHERE source NOT IN ('house','senate') AND ticker IS NOT NULL AND member_id IS NOT NULL
    )
    SELECT
      (SELECT COUNT(*) FROM ours o JOIN theirs t USING (member_id, ticker, transaction_date)) AS agree,
      (SELECT COUNT(*) FROM theirs t LEFT JOIN ours o USING (member_id, ticker, transaction_date)
        WHERE o.ticker IS NULL) AS vendor_only,
      (SELECT COUNT(*) FROM ours o LEFT JOIN theirs t USING (member_id, ticker, transaction_date)
        WHERE t.ticker IS NULL) AS ours_only
    """
    return dict(con.execute(q).fetchone())
