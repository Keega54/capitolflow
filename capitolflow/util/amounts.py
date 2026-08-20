"""Parse the STOCK Act disclosure amount ranges into low/high/point estimates.

Members disclose a bracket, never an exact figure. A naive arithmetic midpoint
badly overstates typical trade size because the brackets are wide and the
underlying distribution is roughly log-uniform, so we use the geometric mean as
the point estimate and keep low/high around for honest error bars.
"""
from __future__ import annotations
import math, re
from ..config import AMOUNT_BRACKETS

_MONEY = re.compile(r"\$?\s*([\d,]+(?:\.\d+)?)")
_OVER = re.compile(r"(over|greater than|more than|\+)\s*\$?\s*([\d,]+)", re.I)
_SPELLED = {
    "1,001 - 15,000": (1001, 15000),
    "15,001 - 50,000": (15001, 50000),
    "50,001 - 100,000": (50001, 100000),
    "100,001 - 250,000": (100001, 250000),
    "250,001 - 500,000": (250001, 500000),
    "500,001 - 1,000,000": (500001, 1000000),
    "1,000,001 - 5,000,000": (1000001, 5000000),
    "5,000,001 - 25,000,000": (5000001, 25000000),
    "25,000,001 - 50,000,000": (25000001, 50000000),
}


def _num(s: str) -> float:
    return float(s.replace(",", "").replace("$", "").strip())


def parse_amount(text: str | None) -> tuple[float | None, float | None, float | None]:
    """Return (low, high, geometric_point_estimate). None on failure."""
    if not text:
        return (None, None, None)
    t = " ".join(str(text).split()).replace("–", "-").replace("—", "-").replace("‐", "-")
    key = t.replace("$", "").strip()
    if key in _SPELLED:
        lo, hi = _SPELLED[key]
        return (float(lo), float(hi), geo_mid(lo, hi))

    m = _OVER.search(t)
    if m:
        lo = _num(m.group(2))
        hi = lo * 2.0                       # open-ended top bracket; assume one more octave
        return (lo, hi, geo_mid(lo, hi))

    nums = [_num(x) for x in _MONEY.findall(t)]
    nums = [n for n in nums if n >= 1]
    if len(nums) >= 2:
        lo, hi = min(nums[0], nums[1]), max(nums[0], nums[1])
        return (lo, hi, geo_mid(lo, hi))
    if len(nums) == 1:
        n = nums[0]
        lo, hi = snap_bracket(n)
        return (lo, hi, geo_mid(lo, hi))
    return (None, None, None)


def geo_mid(lo: float | None, hi: float | None) -> float | None:
    if lo is None or hi is None:
        return None
    lo = max(float(lo), 1.0)
    hi = max(float(hi), lo)
    return math.sqrt(lo * hi)


def snap_bracket(n: float) -> tuple[float, float]:
    for lo, hi in AMOUNT_BRACKETS:
        if lo <= n <= hi:
            return (lo, hi)
    return (n, n)
