"""Parser and utility tests. These are the layer most likely to break silently
when a filing template changes, so they assert on exact values."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
FIX = Path(__file__).parent / "fixtures"

from capitolflow.parse.house_ptr import parse_ptr, parse_ptr_text
from capitolflow.parse.senate_ptr import parse_senate_html
from capitolflow.util.amounts import parse_amount, geo_mid
from capitolflow.util.dates import parse_date, delay_days, quarter_of
from capitolflow.util.tickers import TickerResolver
from capitolflow.sources.members import norm_person


# ---------------------------------------------------------------- amounts
@pytest.mark.parametrize("text,lo,hi", [
    ("$1,001 - $15,000", 1001, 15000),
    ("$15,001–$50,000", 15001, 50000),
    ("1,000,001 - 5,000,000", 1000001, 5000000),
    ("$50,001 -$100,000", 50001, 100000),
    ("Over $50,000,000", 50000000, 100000000),
])
def test_parse_amount(text, lo, hi):
    a, b, est = parse_amount(text)
    assert (a, b) == (lo, hi)
    assert lo <= est <= hi


def test_amount_unparseable_is_none():
    assert parse_amount(None) == (None, None, None)
    assert parse_amount("n/a") == (None, None, None)


def test_geometric_midpoint_is_below_arithmetic():
    """The point estimate must not overstate a wide bracket."""
    lo, hi = 1000001, 5000000
    assert geo_mid(lo, hi) < (lo + hi) / 2


# ---------------------------------------------------------------- dates
def test_date_formats_agree():
    for s in ("03/14/2026", "2026-03-14", "Mar 14, 2026", "March 14, 2026"):
        assert parse_date(s).isoformat() == "2026-03-14"


def test_delay_and_quarter():
    assert delay_days("01/02/2026", "03/14/2026") == 71
    assert quarter_of("2026-03-14") == "2026Q1"
    assert delay_days(None, "03/14/2026") is None


# ---------------------------------------------------------------- tickers
@pytest.fixture
def resolver():
    return TickerResolver.from_sec_json({
        "0": {"ticker": "AAPL", "title": "Apple Inc."},
        "1": {"ticker": "MSFT", "title": "MICROSOFT CORP"},
        "2": {"ticker": "NVDA", "title": "NVIDIA CORP"},
        "3": {"ticker": "LMT", "title": "LOCKHEED MARTIN CORP"},
        "4": {"ticker": "TSLA", "title": "Tesla, Inc."},
    })


@pytest.mark.parametrize("raw,expected", [
    ("Apple Inc. (AAPL) [ST]", "AAPL"),
    ("MICROSOFT CORP - Common Stock", "MSFT"),
    ("NVIDIA Corp $170 Call 01/16/2026", "NVDA"),      # option -> underlying
    ("Tesla Inc Common Stock", "TSLA"),
    ("Lockheed Martin Corp 250 shares purchased", "LMT"),
])
def test_ticker_resolution(resolver, raw, expected):
    tic, conf, _ = resolver.resolve(raw)
    assert tic == expected and conf >= 0.7


def test_resolver_refuses_to_guess(resolver):
    """An unknown private holding must resolve to None, never a nearby company."""
    assert resolver.resolve("Acme Widget Holdings LLC")[0] is None
    assert resolver.resolve("US Treasury Bill 912797GK4 [GS]")[0] is None


def test_asset_type_classification(resolver):
    assert resolver.resolve("US Treasury Bill 912797GK4 [GS]")[2] == "bond"
    assert resolver.resolve("NVIDIA Corp $170 Call 01/16/2026")[2] == "option"
    assert resolver.resolve("Bitcoin [CT]")[2] == "crypto"


# ---------------------------------------------------------------- names
@pytest.mark.parametrize("raw", ["Hon. Nancy Pelosi", "Pelosi, Nancy", "Nancy P. Pelosi",
                                 "Rep. Nancy Pelosi"])
def test_name_normalization(raw):
    assert norm_person(raw) == "pelosi nancy"


# ---------------------------------------------------------------- House PTR
def test_house_ptr_text_parse():
    rows = list(parse_ptr_text((FIX / "house_ptr_sample.txt").read_text()))
    assert len(rows) == 7, "every transaction row must be recovered"
    by_asset = {r["asset_name_raw"].split(" (")[0]: r for r in rows}
    apple = by_asset["Apple Inc."]
    assert apple["owner"] == "spouse"
    assert apple["txn_type"] == "buy"
    assert apple["transaction_date"] == "2026-01-14"
    assert apple["amount_low"] == 15001
    msft = by_asset["Microsoft Corporation"]
    assert msft["txn_type"] == "sell_partial", "S (partial) must not collapse to a plain sale"
    lmt = by_asset["Lockheed Martin Corporation"]
    assert lmt["txn_type"] == "sell", "an asset name wrapped across lines must still parse"


def test_house_ptr_header_never_leaks_into_asset_name():
    rows = list(parse_ptr_text((FIX / "house_ptr_sample.txt").read_text()))
    for r in rows:
        assert "$200" not in r["asset_name_raw"]
        assert "Notification" not in r["asset_name_raw"]
        assert not r["asset_name_raw"].startswith("Type")


@pytest.mark.skipif(not (FIX / "house_ptr_sample.pdf").exists(), reason="fixture PDF absent")
def test_house_ptr_pdf_text_layer():
    rows, mode, meta = parse_ptr((FIX / "house_ptr_sample.pdf").read_bytes())
    assert mode == "pdf_text" and len(rows) == 7
    assert meta["doc_id"] == "20026451"


@pytest.mark.skipif(not (FIX / "house_ptr_scanned.pdf").exists(), reason="fixture PDF absent")
def test_house_ptr_ocr_fallback():
    """A scan with no text layer must still yield every row via OCR."""
    rows, mode, _ = parse_ptr((FIX / "house_ptr_scanned.pdf").read_bytes())
    assert mode == "pdf_ocr"
    assert len(rows) == 7
    assert all(r["transaction_date"] for r in rows)
    # OCR misreads single-letter type codes; the corrector must snap them back.
    assert all(r["txn_type"] != "other" for r in rows)


# ---------------------------------------------------------------- Senate PTR
def test_senate_html_parse():
    rows = list(parse_senate_html((FIX / "senate_ptr_sample.html").read_text()))
    assert len(rows) == 4
    assert rows[0]["declared_ticker"] == "AAPL" and rows[0]["owner"] == "spouse"
    assert rows[1]["txn_type"] == "sell_full"
    assert rows[1]["comment"] == "Sold to fund tax payment"
    assert rows[2]["declared_ticker"] is None       # '--' is not a ticker
    assert rows[3]["owner"] == "dependent"
