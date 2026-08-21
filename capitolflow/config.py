"""Central configuration. Everything overridable by env var so CI needs no file edits."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CAPITOLFLOW_DATA", ROOT / "data"))
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = Path(os.environ.get("CAPITOLFLOW_DB", DATA_DIR / "capitolflow.db"))

# Disclosure ranges defined by the STOCK Act / EIGA reporting brackets.
AMOUNT_BRACKETS = [
    (1.0, 1000.0),
    (1001.0, 15000.0),
    (15001.0, 50000.0),
    (50001.0, 100000.0),
    (100001.0, 250000.0),
    (250001.0, 500000.0),
    (500001.0, 1000000.0),
    (1000001.0, 5000000.0),
    (5000001.0, 25000000.0),
    (25000001.0, 50000000.0),
    (50000001.0, 100000000.0),
]

BENCHMARK = os.environ.get("CAPITOLFLOW_BENCHMARK", "SPY")
RETURN_HORIZONS = (30, 90, 180, 365)


@dataclass
class Settings:
    db_path: Path = DB_PATH
    data_dir: Path = DATA_DIR
    cache_dir: Path = CACHE_DIR

    # How far back to ingest on a cold start.
    start_year: int = int(os.environ.get("CAPITOLFLOW_START_YEAR", "2014"))

    # Politeness. These are public servers; do not hammer them.
    request_delay_s: float = float(os.environ.get("CAPITOLFLOW_DELAY", "1.0"))
    user_agent: str = os.environ.get(
        "CAPITOLFLOW_UA",
        "CapitolFlow/1.0 (public-disclosure research; contact: set CAPITOLFLOW_UA)",
    )
    timeout_s: int = 60
    connect_timeout_s: int = 15
    max_retries: int = 4

    # OCR is slow; cap per-run work so a scheduled job finishes.
    ocr_enabled: bool = os.environ.get("CAPITOLFLOW_OCR", "1") != "0"
    max_ocr_per_run: int = int(os.environ.get("CAPITOLFLOW_MAX_OCR", "150"))
    max_new_filings_per_run: int = int(os.environ.get("CAPITOLFLOW_MAX_FILINGS", "400"))

    # Credentials (all optional; each source degrades gracefully without them).
    lda_api_key: str | None = os.environ.get("LDA_API_KEY") or None
    finnhub_key: str | None = os.environ.get("FINNHUB_API_KEY") or None
    quiver_key: str | None = os.environ.get("QUIVER_API_KEY") or None
    fmp_key: str | None = os.environ.get("FMP_API_KEY") or None

    # Yahoo first: this project's main home is a cloud scheduler, and Stooq
    # refuses datacenter IPs and enforces a daily cap. Stooq remains a key-free
    # fallback and is usually the better choice when running on a laptop.
    price_provider: str = os.environ.get("CAPITOLFLOW_PRICES", "yahoo")  # yahoo | stooq | fmp
    max_price_tickers: int = int(os.environ.get("CAPITOLFLOW_MAX_PRICE_TICKERS", "150"))
    horizons: tuple = field(default_factory=lambda: RETURN_HORIZONS)
    benchmark: str = BENCHMARK

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()

# --- endpoints -------------------------------------------------------------
HOUSE_BULK_ZIP = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.ZIP"
HOUSE_PTR_PDF = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"
HOUSE_FD_PDF = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}/{doc_id}.pdf"

SENATE_HOME = "https://efdsearch.senate.gov/search/home/"
SENATE_SEARCH = "https://efdsearch.senate.gov/search/"
SENATE_DATA = "https://efdsearch.senate.gov/search/report/data/"
SENATE_BASE = "https://efdsearch.senate.gov"

LDA_FILINGS = "https://lda.gov/api/v1/filings/"
LDA_FILINGS_FALLBACK = "https://lda.senate.gov/api/v1/filings/"

LEGISLATORS_CURRENT = "https://unitedstates.github.io/congress-legislators/legislators-current.json"
LEGISLATORS_HISTORICAL = "https://unitedstates.github.io/congress-legislators/legislators-historical.json"
COMMITTEE_MEMBERSHIP = "https://unitedstates.github.io/congress-legislators/committee-membership-current.json"
COMMITTEES_CURRENT = "https://unitedstates.github.io/congress-legislators/committees-current.json"

SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
STOOQ_CSV = "https://stooq.com/q/d/l/?s={sym}&i=d"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
