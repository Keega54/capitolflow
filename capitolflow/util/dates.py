from __future__ import annotations
import re
from datetime import date, datetime, timedelta

_FORMATS = ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%b %d, %Y", "%B %d, %Y",
            "%d %b %Y", "%m-%d-%Y", "%Y/%m/%d")
_CLEAN = re.compile(r"[^\w/\-,: ]+")


def parse_date(s) -> date | None:
    if s is None:
        return None
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, date):
        return s
    t = _CLEAN.sub("", str(s)).strip()
    if not t:
        return None
    t = t.split()[0] if re.match(r"^\d", t) and " " in t and "," not in t else t
    for f in _FORMATS:
        try:
            d = datetime.strptime(t, f).date()
            # Two-digit years: 20xx unless that lands absurdly in the future.
            if d.year < 1900:
                d = d.replace(year=d.year + 2000)
            return d
        except ValueError:
            continue
    return None


def iso(d) -> str | None:
    d = parse_date(d)
    return d.isoformat() if d else None


def delay_days(txn_date, filed) -> int | None:
    a, b = parse_date(txn_date), parse_date(filed)
    if not a or not b:
        return None
    return (b - a).days


def add_days(d, n: int) -> date | None:
    d = parse_date(d)
    return (d + timedelta(days=n)) if d else None


def quarter_of(d) -> str | None:
    d = parse_date(d)
    return f"{d.year}Q{(d.month - 1) // 3 + 1}" if d else None
