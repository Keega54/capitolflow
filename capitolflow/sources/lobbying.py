"""Federal lobbying disclosures from the Senate/House LDA REST API.

Registrants file quarterly LD-2 reports naming the client, the dollars spent and
the specific issues and agencies lobbied. Joining that to trades needs the
client company mapped to a ticker, which is the same name-matching problem as
asset resolution, so it reuses the same resolver and keeps a confidence score.
"""
from __future__ import annotations
import logging, time
from datetime import date

from ..config import LDA_FILINGS, LDA_FILINGS_FALLBACK, SETTINGS
from ..db import upsert_many
from ..util.dates import iso
from ..util.http import get_json, make_session
from ..util.tickers import default_resolver, normalize_name

log = logging.getLogger(__name__)


def _session():
    h = {"Accept": "application/json"}
    if SETTINGS.lda_api_key:
        h["Authorization"] = f"Token {SETTINGS.lda_api_key}"
    return make_session(h)


def _iter_filings(s, base, params: dict, max_pages: int = 400):
    from urllib.parse import urlencode
    url = f"{base}?{urlencode(params)}"
    pages = 0
    while url and pages < max_pages:
        js = get_json(s, url, max_age_s=12 * 3600)
        for row in js.get("results", []):
            yield row
        url = js.get("next")
        pages += 1


def _amount(row) -> float | None:
    for k in ("income", "expenses"):
        v = row.get(k)
        if v not in (None, "", "0.00"):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def ingest(con, *, years=None, session=None, max_pages: int = 400) -> int:
    s = session or _session()
    years = years or [date.today().year, date.today().year - 1]
    resolver = default_resolver()

    filings, activities = [], []
    for yr in years:
        base = LDA_FILINGS
        params = {"filing_year": yr, "page_size": 100}
        try:
            it = _iter_filings(s, base, params, max_pages)
            first = next(it, None)
            rows = ([first] if first else [])
        except Exception as e:
            log.warning("LDA primary host failed (%s); trying fallback", e)
            base = LDA_FILINGS_FALLBACK
            it = _iter_filings(s, base, params, max_pages)
            rows = []
        for row in ([r for r in rows if r] + list(it)):
            client = (row.get("client") or {})
            cname = client.get("name") or ""
            tic, conf, _ = resolver.resolve(cname) if cname else (None, 0.0, "other")
            uid = row.get("filing_uuid") or row.get("id")
            if not uid:
                continue
            filings.append({
                "filing_uuid": uid,
                "filing_year": row.get("filing_year"),
                "filing_period": row.get("filing_period") or row.get("filing_period_display"),
                "filing_type": row.get("filing_type") or row.get("filing_type_display"),
                "dt_posted": iso((row.get("dt_posted") or "")[:10]),
                "period_start": None, "period_end": None,
                "registrant_name": (row.get("registrant") or {}).get("name"),
                "client_name": cname, "client_id": str(client.get("id") or ""),
                "amount": _amount(row),
                "ticker": tic, "ticker_confidence": conf,
            })
            for act in row.get("lobbying_activities", []) or []:
                ents = act.get("government_entities") or []
                names = [e.get("name") for e in ents] or [None]
                for nm in names:
                    activities.append({
                        "filing_uuid": uid,
                        "issue_code": act.get("general_issue_code")
                                      or act.get("general_issue_code_display"),
                        "description": (act.get("description") or "")[:1000],
                        "entity": nm,
                    })
    n = upsert_many(con, "lobbying_filings", filings, mode="REPLACE")
    upsert_many(con, "lobbying_activities", activities, mode="IGNORE")
    _fill_periods(con)
    return n


_PERIOD_MONTHS = {
    "first quarter": ("01-01", "03-31"), "q1": ("01-01", "03-31"),
    "second quarter": ("04-01", "06-30"), "q2": ("04-01", "06-30"),
    "third quarter": ("07-01", "09-30"), "q3": ("07-01", "09-30"),
    "fourth quarter": ("10-01", "12-31"), "q4": ("10-01", "12-31"),
    "mid-year": ("01-01", "06-30"), "year-end": ("07-01", "12-31"),
}


def _fill_periods(con) -> None:
    for r in con.execute("SELECT filing_uuid, filing_year, filing_period FROM lobbying_filings "
                         "WHERE period_start IS NULL AND filing_year IS NOT NULL").fetchall():
        p = (r["filing_period"] or "").strip().lower()
        key = next((k for k in _PERIOD_MONTHS if k in p), None)
        if not key:
            continue
        a, b = _PERIOD_MONTHS[key]
        con.execute("UPDATE lobbying_filings SET period_start=?, period_end=? WHERE filing_uuid=?",
                    (f"{r['filing_year']}-{a}", f"{r['filing_year']}-{b}", r["filing_uuid"]))
