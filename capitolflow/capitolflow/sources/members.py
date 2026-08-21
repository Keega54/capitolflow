"""Roster of filers: Congress (from unitedstates/congress-legislators) plus the
executive branch (cabinet + White House officials, who file OGE Form 278e).

Name matching from filings back to a roster entry is the single biggest source
of silent error in this kind of project, so aliases are stored explicitly and
ambiguous matches are left unresolved rather than guessed.
"""
from __future__ import annotations
import json, logging, re
from datetime import date

from ..config import (COMMITTEES_CURRENT, COMMITTEE_MEMBERSHIP, LEGISLATORS_CURRENT,
                      LEGISLATORS_HISTORICAL, SETTINGS)
from ..db import upsert, upsert_many
from ..util.http import get_json, make_session

log = logging.getLogger(__name__)

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|md|phd|esq)\.?$", re.I)
_HONORIFIC = re.compile(r"^(hon|mr|mrs|ms|dr|rep|sen|representative|senator|the honorable)\.?\s+", re.I)


def norm_person(name: str) -> str:
    """Normalize a filer name to 'last first' for alias lookup."""
    if not name:
        return ""
    n = str(name).strip()
    n = _HONORIFIC.sub("", n)
    n = re.sub(r"\s+", " ", n)
    if "," in n:                                  # "Pelosi, Nancy" form
        last, _, rest = n.partition(",")
        n = f"{rest.strip()} {last.strip()}"
    n = _SUFFIX.sub("", n).strip()
    n = re.sub(r"[^\w\s'-]", "", n).lower()
    parts = [p for p in n.split() if p]
    if len(parts) < 2:
        return " ".join(parts)
    return f"{parts[-1]} {parts[0]}"


def _member_row(p: dict) -> dict | None:
    idd = p.get("id", {})
    name = p.get("name", {})
    terms = p.get("terms") or []
    if not terms:
        return None
    last_term = terms[-1]
    bg = idd.get("bioguide")
    if not bg:
        return None
    chamber = "house" if last_term.get("type") == "rep" else "senate"
    end = last_term.get("end")
    return {
        "member_id": bg,
        "bioguide_id": bg,
        "full_name": name.get("official_full") or f"{name.get('first','')} {name.get('last','')}".strip(),
        "first_name": name.get("first"),
        "last_name": name.get("last"),
        "chamber": chamber,
        "party": last_term.get("party"),
        "state": last_term.get("state"),
        "district": str(last_term.get("district")) if last_term.get("district") is not None else None,
        "role_title": "Rep." if chamber == "house" else "Sen.",
        "term_start": terms[0].get("start"),
        "term_end": end,
        "active": 1 if (end and end >= date.today().isoformat()) else 0,
    }


def _aliases_for(p: dict, member_id: str) -> list[dict]:
    name = p.get("name", {})
    first, last = name.get("first", ""), name.get("last", "")
    nick = name.get("nickname")
    official = name.get("official_full")
    cands = {f"{first} {last}", f"{last}, {first}"}
    if nick:
        cands |= {f"{nick} {last}", f"{last}, {nick}"}
    if official:
        cands.add(official)
    if name.get("middle"):
        cands.add(f"{first} {name['middle']} {last}")
    out, seen = [], set()
    for c in cands:
        k = norm_person(c)
        if k and k not in seen:
            seen.add(k)
            out.append({"alias_norm": k, "member_id": member_id, "source": "congress-legislators"})
    return out


def sync_legislators(con, session=None, include_historical: bool = True) -> int:
    s = session or make_session()
    n = 0
    urls = [LEGISLATORS_CURRENT] + ([LEGISLATORS_HISTORICAL] if include_historical else [])
    for url in urls:
        try:
            people = get_json(s, url, max_age_s=86400)
        except Exception as e:                                    # network optional
            log.warning("legislator sync failed for %s: %s", url, e)
            continue
        rows, aliases = [], []
        for p in people:
            r = _member_row(p)
            if not r:
                continue
            # Skip anyone whose last term ended before our analysis window.
            if r["term_end"] and r["term_end"] < f"{SETTINGS.start_year - 2}-01-01":
                continue
            rows.append(r)
            aliases += _aliases_for(p, r["member_id"])
        n += upsert_many(con, "members", rows, mode="REPLACE")
        upsert_many(con, "member_aliases", aliases, mode="REPLACE")
    return n


def sync_committees(con, session=None) -> int:
    s = session or make_session()
    try:
        comms = get_json(s, COMMITTEES_CURRENT, max_age_s=86400)
        memb = get_json(s, COMMITTEE_MEMBERSHIP, max_age_s=86400)
    except Exception as e:
        log.warning("committee sync failed: %s", e)
        return 0
    crows = []
    for c in comms:
        crows.append({"committee_id": c.get("thomas_id"), "name": c.get("name"),
                      "chamber": (c.get("type") or "").lower()})
        for sub in c.get("subcommittees", []) or []:
            crows.append({"committee_id": f"{c.get('thomas_id')}{sub.get('thomas_id')}",
                          "name": f"{c.get('name')} — {sub.get('name')}",
                          "chamber": (c.get("type") or "").lower()})
    upsert_many(con, "committees", [r for r in crows if r["committee_id"]], mode="REPLACE")

    mrows = []
    known = {r["committee_id"] for r in crows if r["committee_id"]}
    for cid, people in memb.items():
        if cid not in known:
            continue
        for i, p in enumerate(people):
            bg = p.get("bioguide")
            if bg:
                mrows.append({"member_id": bg, "committee_id": cid,
                              "rank": p.get("rank", i + 1), "title": p.get("title")})
    # Foreign keys: only keep memberships for members we actually have.
    have = {r["member_id"] for r in con.execute("SELECT member_id FROM members")}
    mrows = [r for r in mrows if r["member_id"] in have]
    return upsert_many(con, "committee_memberships", mrows, mode="REPLACE")


# --------------------------------------------------------------- executive
def seed_executive(con, officials: list[dict] | None = None) -> int:
    """Seed cabinet / executive filers.

    There is no machine-readable roster of OGE 278e filers, so this takes an
    explicit list (data/executive_officials.json) rather than scraping a name
    list that would silently go stale. Each entry: {name, role_title, term_start}.
    """
    path = SETTINGS.data_dir / "executive_officials.json"
    if officials is None:
        if not path.exists():
            return 0
        officials = json.loads(path.read_text())
    rows, aliases = [], []
    for o in officials:
        name = o["name"]
        mid = "exec:" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        parts = name.split()
        rows.append({
            "member_id": mid, "bioguide_id": None, "full_name": name,
            "first_name": parts[0] if parts else None,
            "last_name": parts[-1] if len(parts) > 1 else None,
            "chamber": "executive", "party": o.get("party"), "state": None, "district": None,
            "role_title": o.get("role_title"), "term_start": o.get("term_start"),
            "term_end": o.get("term_end"), "active": 0 if o.get("term_end") else 1,
        })
        aliases.append({"alias_norm": norm_person(name), "member_id": mid, "source": "executive-seed"})
    upsert_many(con, "members", rows, mode="REPLACE")
    upsert_many(con, "member_aliases", aliases, mode="REPLACE")
    return len(rows)


# --------------------------------------------------------------- matching
class MemberMatcher:
    def __init__(self, con):
        self.alias = {r["alias_norm"]: r["member_id"]
                      for r in con.execute("SELECT alias_norm, member_id FROM member_aliases")}
        self.by_last: dict[str, list[dict]] = {}
        for r in con.execute("SELECT member_id, last_name, first_name, chamber, state, district FROM members"):
            ln = (r["last_name"] or "").lower()
            if ln:
                self.by_last.setdefault(ln, []).append(dict(r))

    def match(self, raw_name: str, *, chamber: str | None = None,
              state: str | None = None, district: str | None = None) -> str | None:
        key = norm_person(raw_name)
        if key in self.alias:
            return self.alias[key]
        parts = key.split()
        if not parts:
            return None
        last, first = parts[0], (parts[1] if len(parts) > 1 else "")
        cands = self.by_last.get(last, [])
        if chamber:
            cands = [c for c in cands if c["chamber"] == chamber] or cands
        if state:
            st = [c for c in cands if (c["state"] or "").upper() == state.upper()]
            cands = st or cands
        if district:
            d = [c for c in cands if str(c["district"] or "") == str(district)]
            cands = d or cands
        if len(cands) == 1:
            return cands[0]["member_id"]
        if first:
            fc = [c for c in cands if (c["first_name"] or "").lower().startswith(first[:3])]
            if len(fc) == 1:
                return fc[0]["member_id"]
        return None                     # ambiguous: leave unlinked rather than guess
