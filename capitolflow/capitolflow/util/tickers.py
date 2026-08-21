"""Resolve free-text asset descriptions from filings into tickers.

Filings write things like:
    "Apple Inc. (AAPL) [ST]"
    "Microsoft Corporation - Common Stock"
    "NVIDIA Corp $170 Call 01/16/2026"
    "US Treasury Bill 912797..."
Strategy, in confidence order:
  1. explicit ticker in parentheses / brackets  -> 0.99
  2. exact normalized company-name match        -> 0.90
  3. distinctive token overlap against SEC names-> 0.60-0.85
  4. give up (ticker=None) rather than guess    -> 0.0
Never fabricate a ticker: an unresolved row still counts in volume stats, and a
wrong ticker silently corrupts every return calculation downstream.
"""
from __future__ import annotations
import json, re, unicodedata
from functools import lru_cache
from pathlib import Path

# Bracketed asset-type codes used on House PTRs.
ASSET_CODES = {
    "ST": "stock", "SO": "option", "OP": "option", "ETF": "fund", "MF": "fund",
    "EF": "fund", "CS": "stock", "PS": "stock", "CT": "crypto", "GS": "bond",
    "CO": "bond", "AB": "bond", "MU": "bond", "TB": "bond", "OT": "other",
    "RP": "real_estate", "RE": "real_estate", "IH": "other", "OL": "other",
    "PE": "other", "HN": "other", "FA": "other", "BA": "cash", "5C": "other",
    "5F": "other", "5P": "other",
}

_TICKER_PAREN = re.compile(r"[\(\[]\s*([A-Z][A-Z\.\-]{0,6})\s*[\)\]]")
_TICKER_TRAIL = re.compile(r"\bticker\s*[:\-]?\s*([A-Z][A-Z\.\-]{0,6})\b", re.I)
_CODE_BRACKET = re.compile(r"\[([A-Z0-9]{2,3})\]")
_OPTION_HINT = re.compile(r"\b(call|put)\b|\bexpir", re.I)
_BOND_HINT = re.compile(r"\b(bond|note|treasury|municipal|debenture|t-bill)\b", re.I)
_CRYPTO_HINT = re.compile(r"\b(bitcoin|ethereum|crypto|coinbase wallet|solana)\b", re.I)
_FUND_HINT = re.compile(r"\b(fund|etf|index|trust|portfolio|shares of)\b", re.I)

# Contract/quantity boilerplate that must be stripped before name matching.
_NOISE = [
    re.compile(r"\$\s*[\d,]+(?:\.\d+)?", re.I),                 # strike / dollar figures
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),            # expiry dates
    re.compile(r"\b(call|put|option[s]?|expir\w*|strike|contract[s]?)\b", re.I),
    re.compile(r"\b(common|ordinary|preferred|class\s+[a-c])\s+(stock|shares?)\b", re.I),
    re.compile(r"\b(purchased|sold|partial|full|exchange)\b", re.I),
    re.compile(r"\b\d+(\.\d+)?\s*(shares?|units?|contracts?)\b", re.I),
    re.compile(r"\b(19|20)\d{2}\b"),                              # bare years
    re.compile(r"\b(nyse|nasdaq|amex|otc)\b", re.I),
]


def strip_noise(text: str) -> str:
    t = text
    for rx in _NOISE:
        t = rx.sub(" ", t)
    return " ".join(t.split())

_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "plc", "ltd",
    "limited", "lp", "llc", "sa", "nv", "ag", "holdings", "holding", "group",
    "the", "class", "common", "stock", "shares", "ordinary", "adr", "ads", "new",
    "cl", "a", "b", "c", "and", "&",
}
_STOP_FOR_MATCH = _SUFFIXES | {"international", "technologies", "technology",
                               "systems", "solutions", "industries", "enterprises"}


def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [t for t in s.split() if t and t not in _SUFFIXES]
    return " ".join(toks)


def _key_tokens(s: str) -> frozenset:
    return frozenset(t for t in normalize_name(s).split() if t not in _STOP_FOR_MATCH and len(t) > 2)


def classify_asset(text: str, code: str | None = None) -> str:
    if code and code.upper() in ASSET_CODES:
        return ASSET_CODES[code.upper()]
    t = text or ""
    if _OPTION_HINT.search(t):
        return "option"
    if _CRYPTO_HINT.search(t):
        return "crypto"
    if _BOND_HINT.search(t):
        return "bond"
    if _FUND_HINT.search(t):
        return "fund"
    return "stock"


class TickerResolver:
    """Name/ticker index built from SEC company_tickers.json plus overrides."""

    def __init__(self, mapping: dict | None = None):
        # ticker -> name, and normalized name -> ticker
        self.by_ticker: dict[str, str] = {}
        self.by_name: dict[str, str] = {}
        self.token_index: dict[str, set[str]] = {}
        if mapping:
            self.load(mapping)

    def load(self, mapping: dict) -> None:
        for tic, name in mapping.items():
            tic = tic.upper().strip()
            if not tic:
                continue
            self.by_ticker.setdefault(tic, name)
            n = normalize_name(name)
            if n:
                self.by_name.setdefault(n, tic)
                for tok in _key_tokens(name):
                    self.token_index.setdefault(tok, set()).add(tic)

    @classmethod
    def from_sec_json(cls, blob) -> "TickerResolver":
        if isinstance(blob, (bytes, str)):
            blob = json.loads(blob)
        mapping = {}
        rows = blob.values() if isinstance(blob, dict) else blob
        for row in rows:
            t = str(row.get("ticker", "")).upper()
            n = row.get("title") or row.get("name") or ""
            if t:
                mapping[t] = n
        return cls(mapping)

    @classmethod
    def from_cache(cls, path: Path) -> "TickerResolver":
        if Path(path).exists():
            return cls.from_sec_json(Path(path).read_bytes())
        return cls({})

    # -- resolution ---------------------------------------------------------
    def resolve(self, raw: str) -> tuple[str | None, float, str]:
        """Return (ticker, confidence, asset_type)."""
        if not raw:
            return (None, 0.0, "other")
        text = " ".join(str(raw).split())

        code = None
        mc = _CODE_BRACKET.search(text)
        if mc and mc.group(1).upper() in ASSET_CODES:
            code = mc.group(1)
        asset_type = classify_asset(text, code)

        # 1. explicit symbol
        for m in list(_TICKER_PAREN.finditer(text)) + list(_TICKER_TRAIL.finditer(text)):
            cand = m.group(1).upper().strip(".-")
            if cand in ASSET_CODES or len(cand) < 1:
                continue
            if not self.by_ticker or cand in self.by_ticker:
                return (cand, 0.99, asset_type)
            # unknown symbol but syntactically a ticker: still useful, lower confidence
            if re.fullmatch(r"[A-Z][A-Z\.\-]{0,5}", cand):
                return (cand, 0.70, asset_type)

        if asset_type in ("bond", "cash", "real_estate"):
            return (None, 0.0, asset_type)

        # 2. exact normalized name
        head = strip_noise(re.split(r"[\(\[\-–—,]", text)[0])
        n = normalize_name(head)
        if n and n in self.by_name:
            return (self.by_name[n], 0.90, asset_type)

        # 3. distinctive token overlap
        toks = _key_tokens(head)
        if toks and self.token_index:
            scores: dict[str, float] = {}
            for tok in toks:
                bucket = self.token_index.get(tok)
                if not bucket or len(bucket) > 40:   # ignore generic tokens
                    continue
                w = 1.0 / len(bucket) ** 0.5
                for tic in bucket:
                    scores[tic] = scores.get(tic, 0.0) + w
            if scores:
                best, sc = max(scores.items(), key=lambda kv: kv[1])
                cand_toks = _key_tokens(self.by_ticker.get(best, ""))
                if cand_toks:
                    inter = toks & cand_toks
                    jac = len(inter) / len(toks | cand_toks)
                    # containment: every distinctive token of the candidate appears
                    cont = len(inter) / len(cand_toks)
                    if jac >= 0.6 or (cont >= 1.0 and len(inter) >= 1 and sc >= 0.5):
                        return (best, min(0.85, 0.55 + max(jac, cont * 0.8) * 0.35), asset_type)
        return (None, 0.0, asset_type)


@lru_cache(maxsize=1)
def default_resolver() -> TickerResolver:
    from ..config import SETTINGS
    return TickerResolver.from_cache(SETTINGS.data_dir / "sec_tickers.json")
