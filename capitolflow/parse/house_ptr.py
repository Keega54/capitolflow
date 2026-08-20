"""Parse a House Periodic Transaction Report (PTR) PDF into transaction rows.

House PTRs come in two flavours:
  * digitally generated PDFs with a real text layer (most since ~2018)
  * scans of a paper form, which need OCR
Both render the same table:
  ID | Owner | Asset | Transaction Type | Date | Notification Date | Amount | Cap Gains > $200?

The text layer's column order is stable but whitespace is not, so we anchor on
the two dates and the dollar range and treat everything to their left as the
asset description. That survives line-wrapping, which a column-position parser
does not.
"""
from __future__ import annotations
import io, logging, re
from typing import Iterator

from ..util.amounts import parse_amount
from ..util.dates import iso

log = logging.getLogger(__name__)

DATE_RX = r"\d{1,2}/\d{1,2}/\d{2,4}"
AMOUNT_RX = (r"Over\s*\$[\d,]+"
             r"|\$[\d,]+(?:\.\d{2})?\s*[-\u2010\u2013\u2014]\s*\$?[\d,]+(?:\.\d{2})?"
             r"|\$[\d,]{4,}(?:\.\d{2})?"
             r"|None")

# Longest-first alternation matters: "S (partial)" must be tried before "S".
TYPE_RX = r"S\s*\(\s*partial\s*\)|S\s*\(\s*full\s*\)|Purchase|Sale|P|S|E|X|R"

# The stable anchor on every PTR row is:  TYPE  txn_date  [notification_date]  $amount
TAIL_RX = re.compile(
    r"\b(?P<type>" + TYPE_RX + r")\s+"
    r"(?P<txn_date>" + DATE_RX + r")\s+"
    r"(?:(?P<notif_date>" + DATE_RX + r")\s+)?"
    r"(?P<amount>" + AMOUNT_RX + r")"
    r"(?P<rest>.*)$",
    re.I)

TYPE_MAP = {
    "p": ("buy", 1), "purchase": ("buy", 1),
    "s": ("sell", -1), "sale": ("sell", -1),
    "s(partial)": ("sell_partial", -1), "s(full)": ("sell_full", -1),
    "e": ("exchange", 0), "x": ("exchange", 0), "r": ("receive", 0),
}

_OWNER_MAP = {"sp": "spouse", "dc": "dependent", "jt": "joint",
              "self": "self", "spouse": "spouse", "joint": "joint", "dependent": "dependent"}
_OWNER_LEAD = re.compile(r"^\s*(SP|DC|JT|SELF)\b\s*", re.I)

# The House PTR transaction-type column only ever contains P, S, S (partial),
# S (full) or E. Tesseract routinely misreads the single letters, so in OCR mode
# an out-of-vocabulary letter is snapped to its visually nearest valid value and
# the substitution is recorded on the row.
VALID_TYPES = {"p", "s", "s(partial)", "s(full)", "e", "purchase", "sale"}
OCR_TYPE_FIX = {"r": "p", "f": "p", "b": "p", "d": "p", "5": "s", "$": "s", "8": "s", "c": "e"}

# Header / boilerplate lines that must never be stitched into an asset name.
_JUNK_PATTERNS = [
    r"^ID\b", r"^Owner\b", r"^Asset\b", r"^Transaction\b", r"^Type\b", r"^Date\b",
    r"^Notification\b", r"^Amount\b", r"^Cap\.?\s*Gains", r"^\$200", r"^FILING STATUS",
    r"^SUBHOLDING", r"^Filing\s*ID", r"^Page\s*\d", r"^\*\s*For the complete",
    r"^Periodic Transaction Report", r"^Clerk of the House", r"^Name\s*:", r"^Status\s*:",
    r"^State/District\s*:", r"^Digitally Signed", r"^Initial Public Offering",
    r"^Description\s*:", r"^Location\s*:", r"^IPO\s*:", r"^F\s*S\s*:", r"^COMMENTS",
    r"^\(?Full Year\)?$", r"^SP\s+DC\s+JT", r"^U\.?S\.? House of Representatives",
]
_JUNK_LINE = re.compile("|".join(_JUNK_PATTERNS), re.I)
# Fragments that survive OCR of the column headers and glue onto asset names.
_HEADER_FRAG = re.compile(
    r"\b(Cap\.?\s*Gains\s*>?|\$200\?|Notification\s+Date|Transaction\s+Type|"
    r"ID\s+Owner|Asset\s+Transaction|Type\s+Date)\b", re.I)

_FILING_ID = re.compile(r"Filing\s*ID\s*#?\s*(\d+)", re.I)
_FILED_DATE = re.compile(r"(?:filing date|date received|signed on|digitally signed[^,]*,)\s*:?\s*("
                         + DATE_RX + ")", re.I)


def extract_text(pdf_bytes: bytes, *, ocr: bool = True, ocr_dpi: int = 300) -> tuple[str, str]:
    """Return (text, mode) where mode is 'pdf_text' or 'pdf_ocr'."""
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = []
            for pg in pdf.pages:
                pages.append(pg.extract_text(x_tolerance=1.5, y_tolerance=3) or "")
            text = "\n".join(pages)
    except Exception as e:
        log.debug("pdfplumber failed: %s", e)

    if _has_usable_text(text):
        return text, "pdf_text"

    if not ocr:
        return text, "pdf_text"

    try:
        import pytesseract
        from pdf2image import convert_from_bytes
        imgs = convert_from_bytes(pdf_bytes, dpi=ocr_dpi)
        out = []
        for im in imgs:
            out.append(pytesseract.image_to_string(im, config="--psm 6"))
        return "\n".join(out), "pdf_ocr"
    except Exception as e:
        log.warning("OCR failed: %s", e)
        return text, "pdf_text"


def _has_usable_text(text: str) -> bool:
    if not text or len(text.strip()) < 120:
        return False
    # A real PTR text layer contains dates and a dollar range.
    return bool(re.search(DATE_RX, text)) and bool(re.search(r"\$[\d,]{3,}", text))


def _stitch(lines: list[str]) -> list[str]:
    """Join wrapped asset descriptions onto the line that carries the dates.

    PTR asset names routinely wrap across two or three lines while the type,
    dates and amount always land on the last of them, so we accumulate until a
    line completes the anchor pattern and flush there.
    """
    out: list[str] = []
    buf: list[str] = []
    for ln in lines:
        ln = " ".join(str(ln).split())
        if not ln:
            continue
        if _JUNK_LINE.search(ln):
            buf = []                      # a header resets the buffer, never joins it
            continue
        buf.append(ln)
        candidate = " ".join(buf)
        if TAIL_RX.search(candidate):
            out.append(candidate)
            buf = []
        elif len(buf) > 3 or len(candidate) > 400:
            buf = buf[-1:]                # bound the lookback so junk cannot snowball
    return out


def _clean_asset(prefix: str) -> tuple[str, str]:
    """Split the pre-anchor text into (owner, asset_name)."""
    t = _HEADER_FRAG.sub(" ", prefix)
    t = re.sub(r"^\s*\d{1,3}[\.\)]\s+", "", t)          # leading row number
    owner = "self"
    m = _OWNER_LEAD.search(t)
    if m:
        owner = _OWNER_MAP.get(m.group(1).lower(), "self")
        t = t[m.end():]
    else:                                                   # owner code may follow junk
        m2 = re.search(r"\b(SP|DC|JT)\b(?=\s+[A-Z0-9])", t)
        if m2 and m2.start() < 12:
            owner = _OWNER_MAP.get(m2.group(1).lower(), "self")
            t = t[m2.end():]
    t = re.sub(r"^[\s\-\u2013\u2014:]+", "", t)
    return owner, " ".join(t.split())


def parse_ptr_text(text: str, *, ocr: bool = False) -> Iterator[dict]:
    for line in _stitch(text.splitlines()):
        m = TAIL_RX.search(line)
        if not m:
            continue
        owner, asset = _clean_asset(line[:m.start()])
        if len(asset) < 3:
            continue
        raw_type = re.sub(r"\s+", "", m.group("type")).lower()
        type_note = None
        if ocr and raw_type not in VALID_TYPES and raw_type in OCR_TYPE_FIX:
            type_note = f"ocr_type_fix:{raw_type}->{OCR_TYPE_FIX[raw_type]}"
            raw_type = OCR_TYPE_FIX[raw_type]
        ttype, direction = TYPE_MAP.get(raw_type, ("other", 0))
        lo, hi, est = parse_amount(m.group("amount"))
        rest = (m.group("rest") or "").strip()
        yield {
            "owner": owner,
            "asset_name_raw": asset,
            "txn_type": ttype,
            "direction": direction,
            "transaction_date": iso(m.group("txn_date")),
            "notification_date": iso(m.group("notif_date")) if m.group("notif_date") else None,
            "amount_low": lo, "amount_high": hi, "amount_est": est,
            "cap_gains_over_200": 1 if re.search(r"\bY(es)?\b", rest, re.I) else
                                  (0 if re.search(r"\bN(o)?\b", rest, re.I) else None),
            "raw": line,
            "parse_note": type_note,
        }


def parse_ptr(pdf_bytes: bytes, *, ocr: bool = True) -> tuple[list[dict], str, dict]:
    text, mode = extract_text(pdf_bytes, ocr=ocr)
    rows = list(parse_ptr_text(text, ocr=(mode == "pdf_ocr")))
    meta = {}
    m = _FILING_ID.search(text)
    if m:
        meta["doc_id"] = m.group(1)
    m = _FILED_DATE.search(text)
    if m:
        meta["filed_date"] = iso(m.group(1))
    return rows, mode, meta
