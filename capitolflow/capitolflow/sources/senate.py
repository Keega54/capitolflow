"""US Senate ingest via the Electronic Financial Disclosure (eFD) search.

eFD gates every request behind a one-time "prohibition agreement" checkbox that
sets a session cookie, and its search endpoint is a DataTables-style POST that
needs the CSRF token from the landing page. Once through, report links come in
two shapes:
  /search/view/ptr/<uuid>/    electronic filing -> clean HTML table
  /search/view/paper/<uuid>/  scanned paper filing -> PDF, needs OCR
"""
from __future__ import annotations
import json, logging, re
from datetime import date, datetime

from ..config import (SENATE_BASE, SENATE_DATA, SENATE_HOME, SENATE_SEARCH, SETTINGS)
from ..db import txn_id, upsert, upsert_many
from ..parse.senate_ptr import parse_senate_html, parse_senate_pdf
from ..util.dates import delay_days, iso
from ..util.http import get_bytes, make_session, post, throttle
from ..util.tickers import default_resolver
from .house import _store_rows
from .members import MemberMatcher

log = logging.getLogger(__name__)

REPORT_LINK = re.compile(r'href="(/search/view/([a-z_]+)/([0-9a-f\-]+)/?)"', re.I)
CSRF = re.compile(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"')


def open_session():
    """Accept the eFD terms and return a session carrying the required cookies."""
    s = make_session({"Referer": SENATE_HOME})
    throttle()
    r = s.get(SENATE_HOME, timeout=SETTINGS.timeout_s)
    r.raise_for_status()
    m = CSRF.search(r.text)
    if not m:
        raise RuntimeError("eFD landing page has no CSRF token; layout changed")
    token = m.group(1)
    post(s, SENATE_HOME, data={"prohibition_agreement": "1", "csrfmiddlewaretoken": token},
         headers={"Referer": SENATE_HOME})
    throttle()
    r2 = s.get(SENATE_SEARCH, timeout=SETTINGS.timeout_s)
    m2 = CSRF.search(r2.text)
    s.headers["X-CSRFToken"] = (m2.group(1) if m2 else token)
    s.headers["Referer"] = SENATE_SEARCH
    return s, (m2.group(1) if m2 else token)


def search_reports(s, token, *, start_date: str, end_date: str | None = None,
                   report_types=("11",), page_size: int = 100, max_pages: int = 200):
    """Yield report rows. report_types '11' == Periodic Transaction Report."""
    end_date = end_date or date.today().strftime("%m/%d/%Y")
    offset = 0
    for _ in range(max_pages):
        payload = {
            "start": str(offset), "length": str(page_size),
            "report_types": f"[{','.join(report_types)}]",
            "filer_types": "[]", "submitted_start_date": start_date,
            "submitted_end_date": end_date, "candidate_state": "",
            "senator_state": "", "office_id": "", "first_name": "", "last_name": "",
            "csrfmiddlewaretoken": token,
        }
        r = post(s, SENATE_DATA, data=payload, headers={"Referer": SENATE_SEARCH})
        try:
            js = r.json()
        except Exception:
            log.error("eFD returned non-JSON (session likely expired)")
            return
        rows = js.get("data", [])
        if not rows:
            return
        for row in rows:
            first, last, office, link_html, filed = (list(row) + [""] * 5)[:5]
            m = REPORT_LINK.search(link_html or "")
            if not m:
                continue
            yield {
                "path": m.group(1), "kind": m.group(2), "uuid": m.group(3),
                "first_name": first, "last_name": last, "office": office,
                "filer_name_raw": " ".join(x for x in [first, last] if x).strip(),
                "filed_date": iso(filed),
                "url": SENATE_BASE + m.group(1),
            }
        offset += len(rows)
        if len(rows) < page_size:
            return


def ingest(con, *, start_date: str | None = None, session=None,
           limit: int | None = None, ocr: bool = True) -> tuple[int, int]:
    start_date = start_date or f"01/01/{SETTINGS.start_year}"
    if session is None:
        session, token = open_session()
    else:
        token = session.headers.get("X-CSRFToken", "")

    matcher = MemberMatcher(con)
    resolver = default_resolver()
    have = {r["doc_id"] for r in con.execute(
        "SELECT doc_id FROM filings WHERE source='senate' AND parse_status IN ('ok','partial')")}

    n_f = n_t = ocr_used = 0
    for rec in search_reports(session, token, start_date=start_date):
        if rec["uuid"] in have:
            continue
        if limit and n_f >= limit:
            break
        filing_id = f"senate:{rec['uuid']}"
        member_id = matcher.match(rec["filer_name_raw"], chamber="senate")
        status, note, mode, rows = "failed", "", None, []
        try:
            if rec["kind"] == "paper":
                pdf = get_bytes(session, rec["url"], suffix=".pdf")
                allow = ocr and SETTINGS.ocr_enabled and ocr_used < SETTINGS.max_ocr_per_run
                rows, mode, _ = parse_senate_pdf(pdf, ocr=allow)
                if mode == "pdf_ocr":
                    ocr_used += 1
            else:
                html = get_bytes(session, rec["url"], suffix=".html").decode("utf8", "replace")
                rows, mode = list(parse_senate_html(html)), "html"
            status = "ok" if rows else "partial"
            if not rows:
                note = "no transaction rows matched"
        except Exception as e:
            note = f"{type(e).__name__}: {e}"[:500]
            log.warning("senate %s: %s", rec["uuid"], note)

        upsert(con, "filings", {
            "filing_id": filing_id, "source": "senate", "doc_id": rec["uuid"],
            "member_id": member_id, "filer_name_raw": rec["filer_name_raw"],
            "filing_type": "ptr", "filing_year": int((rec["filed_date"] or "0000")[:4]) or None,
            "filed_date": rec["filed_date"], "url": rec["url"], "doc_format": mode,
            "parse_status": status, "parse_note": note, "fetched_at": None,
        })
        n_f += 1
        n_t += _store_rows(con, rows, filing_id, member_id, rec["filed_date"], "senate", resolver)
    return n_f, n_t
