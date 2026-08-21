"""Parse Senate electronic PTR pages (HTML tables) and paper PTRs (scanned PDFs).

Electronic filings render a clean table:
  # | Transaction Date | Owner | Ticker | Asset Name | Asset Type | Type | Amount | Comment
which is far easier than the House's PDFs — when a senator files electronically.
Paper filers fall back to the same OCR path the House uses.
"""
from __future__ import annotations
import re
from typing import Iterator

from ..util.amounts import parse_amount
from ..util.dates import iso

_TYPE_MAP = {
    "purchase": ("buy", 1), "buy": ("buy", 1),
    "sale": ("sell", -1), "sale (full)": ("sell_full", -1),
    "sale (partial)": ("sell_partial", -1), "sale (partial sale)": ("sell_partial", -1),
    "exchange": ("exchange", 0), "receive": ("receive", 0),
}
_OWNER_MAP = {"self": "self", "spouse": "spouse", "joint": "joint",
              "child": "dependent", "dependent child": "dependent", "--": "self", "": "self"}

_HDR_ALIASES = {
    "transaction date": "transaction_date", "date": "transaction_date",
    "owner": "owner", "ticker": "ticker", "asset name": "asset_name_raw",
    "asset": "asset_name_raw", "asset type": "asset_type_raw",
    "type": "txn_type_raw", "transaction type": "txn_type_raw",
    "amount": "amount", "comment": "comment", "comments": "comment",
}


def _norm(s: str) -> str:
    return " ".join(str(s or "").split()).strip()


def parse_senate_html(html: str) -> Iterator[dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    for table in soup.find_all("table"):
        head = table.find("thead")
        if not head:
            continue
        headers = [_norm(th.get_text()).lower() for th in head.find_all("th")]
        cols = [_HDR_ALIASES.get(h, None) for h in headers]
        if "transaction_date" not in cols or "amount" not in cols:
            continue
        body = table.find("tbody") or table
        for tr in body.find_all("tr"):
            cells = [_norm(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
            if len(cells) < 4:
                continue
            rec = {}
            for key, val in zip(cols, cells):
                if key:
                    rec[key] = val
            row = _to_row(rec)
            if row:
                yield row


def _to_row(rec: dict) -> dict | None:
    tdate = iso(rec.get("transaction_date"))
    if not tdate:
        return None
    raw_type = _norm(rec.get("txn_type_raw")).lower()
    ttype, direction = _TYPE_MAP.get(raw_type, ("other", 0))
    if ttype == "other" and raw_type.startswith("sale"):
        ttype, direction = "sell", -1
    if ttype == "other" and raw_type.startswith("purchase"):
        ttype, direction = "buy", 1
    lo, hi, est = parse_amount(rec.get("amount"))
    ticker = _norm(rec.get("ticker")).upper()
    if ticker in ("--", "-", "N/A", "NONE"):
        ticker = ""
    asset = _norm(rec.get("asset_name_raw"))
    if ticker and ticker not in asset:
        asset = f"{asset} ({ticker})"
    return {
        "owner": _OWNER_MAP.get(_norm(rec.get("owner")).lower(), "self"),
        "asset_name_raw": asset or ticker,
        "declared_ticker": ticker or None,
        "declared_asset_type": _norm(rec.get("asset_type_raw")) or None,
        "txn_type": ttype, "direction": direction,
        "transaction_date": tdate, "notification_date": None,
        "amount_low": lo, "amount_high": hi, "amount_est": est,
        "comment": (lambda c: None if c in ("", "--", "-", "N/A") else c)(_norm(rec.get("comment"))),
        "raw": " | ".join(f"{k}={v}" for k, v in rec.items()),
    }


def parse_senate_pdf(pdf_bytes: bytes, *, ocr: bool = True):
    """Paper senate filings use the same layout family as House PTRs."""
    from .house_ptr import parse_ptr
    return parse_ptr(pdf_bytes, ocr=ocr)
