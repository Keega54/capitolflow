"""US House of Representatives ingest.

The Clerk publishes one ZIP per calendar year containing an XML index of every
financial disclosure filed that year. The index gives DocID + filing type; the
actual transactions live in a per-filing PDF that has to be fetched and parsed.
Filing type 'P' is the Periodic Transaction Report — the 30/45-day trade filing
this project cares about.
"""
from __future__ import annotations
import io, json, logging, re, zipfile
from xml.etree import ElementTree as ET

from ..config import HOUSE_BULK_ZIP, HOUSE_PTR_PDF, SETTINGS
from ..db import txn_id, upsert, upsert_many
from ..parse.house_ptr import parse_ptr
from ..util.dates import delay_days, iso
from ..util.http import get_bytes, make_session
from ..util.tickers import default_resolver
from .members import MemberMatcher

log = logging.getLogger(__name__)


class Budget:
    """A run-level allowance of filings, shared across sources and years."""

    def __init__(self, total: int):
        self.total = max(int(total), 0)
        self.used = 0

    @property
    def remaining(self) -> int:
        return max(self.total - self.used, 0)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def spend(self, n: int = 1) -> None:
        self.used += n

PTR_TYPES = {"P"}
FILING_TYPE_MAP = {"P": "ptr", "O": "annual", "A": "amendment", "C": "candidate",
                   "T": "termination", "B": "blind_trust", "X": "extension", "D": "annual"}


def _text(el, tag) -> str | None:
    n = el.find(tag)
    return (n.text or "").strip() if n is not None and n.text else None


def fetch_year_index(session, year: int) -> list[dict]:
    """Download the Clerk's annual ZIP and return the filing index as dicts."""
    url = HOUSE_BULK_ZIP.format(year=year)
    # The current year's ZIP changes daily; older years are effectively frozen.
    from datetime import date
    max_age = 6 * 3600 if year >= date.today().year else None
    blob = get_bytes(session, url, suffix=".zip", max_age_s=max_age)
    z = zipfile.ZipFile(io.BytesIO(blob))
    xml_names = [n for n in z.namelist() if n.lower().endswith(".xml")]
    if not xml_names:
        raise RuntimeError(f"no XML index inside {url}: {z.namelist()}")
    root = ET.fromstring(z.read(xml_names[0]))
    out = []
    for m in root.iter("Member"):
        doc_id = _text(m, "DocID")
        if not doc_id:
            continue
        ft = (_text(m, "FilingType") or "").upper()
        first, last = _text(m, "First"), _text(m, "Last")
        suffix = _text(m, "Suffix")
        name = " ".join(x for x in [_text(m, "Prefix"), first, last, suffix] if x)
        sd = _text(m, "StateDst") or ""
        out.append({
            "doc_id": doc_id,
            "filing_type": FILING_TYPE_MAP.get(ft, ft.lower() or "unknown"),
            "filing_type_raw": ft,
            "filer_name_raw": name,
            "first_name": first, "last_name": last,
            "state": sd[:2] or None,
            "district": sd[2:] or None,
            "filing_year": int(_text(m, "Year") or year),
            "filed_date": iso(_text(m, "FilingDate")),
            "year": year,
        })
    return out


def ingest_year(con, year: int, *, session=None, ptr_only: bool = True,
                limit: int | None = None, ocr: bool = True,
                budget: "Budget | None" = None) -> tuple[int, int]:
    """Ingest one year of House filings. Returns (new_filings, new_transactions).

    `budget` is a run-level allowance shared across every year, so a multi-year
    backfill stops at a total the scheduler can finish rather than doing `limit`
    filings per year and running out of wall clock.
    """
    s = session or make_session()
    matcher = MemberMatcher(con)
    resolver = default_resolver()

    index = fetch_year_index(s, year)
    if ptr_only:
        index = [r for r in index if r["filing_type"] == "ptr"]

    have = {r["doc_id"] for r in con.execute(
        "SELECT doc_id FROM filings WHERE source='house' AND parse_status IN ('ok','partial')")}
    todo = [r for r in index if r["doc_id"] not in have]
    if limit:
        todo = todo[:limit]
    if budget is not None:
        todo = todo[:budget.remaining]
    log.info("house %s: %d PTRs in index, %d to fetch (budget left: %s)",
             year, len(index), len(todo), "n/a" if budget is None else budget.remaining)

    n_f = n_t = ocr_used = 0
    for rec in todo:
        filing_id = f"house:{rec['doc_id']}"
        member_id = matcher.match(rec["filer_name_raw"], chamber="house",
                                  state=rec.get("state"), district=rec.get("district"))
        url = HOUSE_PTR_PDF.format(year=rec["year"], doc_id=rec["doc_id"])
        status, note, mode, rows = "failed", "", None, []
        try:
            allow_ocr = ocr and SETTINGS.ocr_enabled and ocr_used < SETTINGS.max_ocr_per_run
            pdf = get_bytes(s, url, suffix=".pdf")
            rows, mode, meta = parse_ptr(pdf, ocr=allow_ocr)
            if mode == "pdf_ocr":
                ocr_used += 1
            if rows:
                status = "ok"
            else:
                status, note = "partial", "no transaction rows matched"
            if meta.get("filed_date") and not rec.get("filed_date"):
                rec["filed_date"] = meta["filed_date"]
        except Exception as e:
            note = f"{type(e).__name__}: {e}"[:500]
            log.warning("house %s %s: %s", year, rec["doc_id"], note)

        upsert(con, "filings", {
            "filing_id": filing_id, "source": "house", "doc_id": rec["doc_id"],
            "member_id": member_id, "filer_name_raw": rec["filer_name_raw"],
            "filing_type": rec["filing_type"], "filing_year": rec["filing_year"],
            "filed_date": rec["filed_date"], "url": url, "doc_format": mode,
            "parse_status": status, "parse_note": note,
            "fetched_at": None,
        })
        n_f += 1
        if budget is not None:
            budget.spend()
        n_t += _store_rows(con, rows, filing_id, member_id, rec["filed_date"], "house", resolver)
        if budget is not None and budget.exhausted:
            log.info("house %s: run budget exhausted, resuming next run", year)
            break
    return n_f, n_t


_DECLARED_TYPE = {
    "stock": "stock", "corporate bond": "bond", "municipal security": "bond",
    "non-public stock": "stock", "other securities": "other", "etf": "fund",
    "mutual fund": "fund", "exchange traded fund": "fund", "stock option": "option",
    "cryptocurrency": "crypto", "digital asset": "crypto", "variable annuity": "other",
}


def _map_declared_type(declared: str, fallback: str) -> str:
    return _DECLARED_TYPE.get((declared or "").strip().lower(), fallback)


def _store_rows(con, rows, filing_id, member_id, filed_date, source, resolver) -> int:
    out = []
    for r in rows:
        tic, conf, atype = resolver.resolve(r["asset_name_raw"])
        # A ticker the filer typed into a structured field beats anything inferred.
        if r.get("declared_ticker"):
            tic, conf = r["declared_ticker"], 1.0
        if r.get("declared_asset_type"):
            atype = _map_declared_type(r["declared_asset_type"], atype)
        out.append({
            "txn_id": txn_id(filing_id, r["asset_name_raw"], r["transaction_date"],
                             r["txn_type"], r["amount_low"], r["owner"]),
            "filing_id": filing_id, "member_id": member_id,
            "transaction_date": r["transaction_date"],
            "notification_date": r.get("notification_date"),
            "filed_date": filed_date,
            "filing_delay_days": delay_days(r["transaction_date"], filed_date),
            "owner": r["owner"], "asset_name_raw": r["asset_name_raw"],
            "ticker": tic, "ticker_confidence": conf, "asset_type": atype,
            "txn_type": r["txn_type"], "direction": r["direction"],
            "amount_low": r["amount_low"], "amount_high": r["amount_high"],
            "amount_est": r["amount_est"],
            "comment": r.get("parse_note"),
            "cap_gains_over_200": r.get("cap_gains_over_200"),
            "source": source,
            "raw": json.dumps(r.get("raw"))[:4000],
        })
    return upsert_many(con, "transactions", out, mode="IGNORE")
