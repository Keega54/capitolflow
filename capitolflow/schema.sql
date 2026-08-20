-- CapitolFlow schema. SQLite. Everything is keyed on TRADE DATE, not disclosure date.
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ---------------------------------------------------------------- people
CREATE TABLE IF NOT EXISTS members (
    member_id      TEXT PRIMARY KEY,          -- bioguide id when known, else slug
    bioguide_id    TEXT,
    full_name      TEXT NOT NULL,
    first_name     TEXT,
    last_name      TEXT,
    chamber        TEXT,                      -- house | senate | executive
    party          TEXT,
    state          TEXT,
    district       TEXT,
    role_title     TEXT,                      -- e.g. 'Rep.', 'Sen.', 'Secretary of X'
    term_start     TEXT,
    term_end       TEXT,
    active         INTEGER DEFAULT 1,
    updated_at     TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_members_last ON members(last_name);
CREATE INDEX IF NOT EXISTS idx_members_chamber ON members(chamber);

CREATE TABLE IF NOT EXISTS member_aliases (
    alias_norm  TEXT PRIMARY KEY,             -- normalized name string seen in a filing
    member_id   TEXT NOT NULL REFERENCES members(member_id),
    source      TEXT
);

CREATE TABLE IF NOT EXISTS committees (
    committee_id TEXT PRIMARY KEY,
    name         TEXT,
    chamber      TEXT
);

CREATE TABLE IF NOT EXISTS committee_memberships (
    member_id    TEXT NOT NULL REFERENCES members(member_id),
    committee_id TEXT NOT NULL REFERENCES committees(committee_id),
    rank         INTEGER,
    title        TEXT,
    PRIMARY KEY (member_id, committee_id)
);

-- ---------------------------------------------------------------- filings
CREATE TABLE IF NOT EXISTS filings (
    filing_id      TEXT PRIMARY KEY,          -- source-prefixed, e.g. 'house:20026123'
    source         TEXT NOT NULL,             -- house | senate | executive | aggregator
    doc_id         TEXT,
    member_id      TEXT REFERENCES members(member_id),
    filer_name_raw TEXT,
    filing_type    TEXT,                      -- ptr | annual | amendment | blind_trust
    filing_year    INTEGER,
    filed_date     TEXT,                      -- ISO date the disclosure hit the public record
    url            TEXT,
    doc_format     TEXT,                      -- html | pdf_text | pdf_ocr
    parse_status   TEXT DEFAULT 'pending',    -- pending | ok | partial | failed
    parse_note     TEXT,
    content_hash   TEXT,
    fetched_at     TEXT,
    UNIQUE (source, doc_id)
);
CREATE INDEX IF NOT EXISTS idx_filings_member ON filings(member_id);
CREATE INDEX IF NOT EXISTS idx_filings_filed ON filings(filed_date);
CREATE INDEX IF NOT EXISTS idx_filings_status ON filings(parse_status);

-- ---------------------------------------------------------------- trades
CREATE TABLE IF NOT EXISTS transactions (
    txn_id            TEXT PRIMARY KEY,       -- deterministic hash, so re-parsing is idempotent
    filing_id         TEXT NOT NULL REFERENCES filings(filing_id),
    member_id         TEXT REFERENCES members(member_id),
    transaction_date  TEXT,                   -- THE date that matters
    notification_date TEXT,
    filed_date        TEXT,
    filing_delay_days INTEGER,                -- filed_date - transaction_date
    owner             TEXT,                   -- self | spouse | joint | dependent
    asset_name_raw    TEXT,
    ticker            TEXT,
    ticker_confidence REAL,                   -- 0..1, how sure the resolver is
    asset_type        TEXT,                   -- stock | option | bond | fund | crypto | other
    txn_type          TEXT,                   -- buy | sell | sell_partial | sell_full | exchange | receive
    direction         INTEGER,                -- +1 buy, -1 sell, 0 neutral
    amount_low        REAL,
    amount_high       REAL,
    amount_est        REAL,                   -- geometric midpoint of the disclosed range
    comment           TEXT,
    cap_gains_over_200 INTEGER,
    source            TEXT,
    raw               TEXT,                   -- original row json, for auditing
    UNIQUE (filing_id, asset_name_raw, transaction_date, txn_type, amount_low, owner)
);
CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_txn_ticker ON transactions(ticker);
CREATE INDEX IF NOT EXISTS idx_txn_member ON transactions(member_id);
CREATE INDEX IF NOT EXISTS idx_txn_ticker_date ON transactions(ticker, transaction_date);

-- ---------------------------------------------------------------- prices
CREATE TABLE IF NOT EXISTS prices (
    ticker  TEXT NOT NULL,
    date    TEXT NOT NULL,
    open    REAL, high REAL, low REAL, close REAL, adj_close REAL, volume REAL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

CREATE TABLE IF NOT EXISTS securities (
    ticker      TEXT PRIMARY KEY,
    name        TEXT,
    cik         TEXT,
    exchange    TEXT,
    sector      TEXT,
    industry    TEXT
);

-- ---------------------------------------------------------------- returns / scores
CREATE TABLE IF NOT EXISTS trade_returns (
    txn_id        TEXT NOT NULL REFERENCES transactions(txn_id),
    horizon_days  INTEGER NOT NULL,
    asset_return  REAL,
    bench_return  REAL,
    excess_return REAL,                       -- signed by trade direction
    computed_at   TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (txn_id, horizon_days)
);

CREATE TABLE IF NOT EXISTS member_scores (
    member_id        TEXT NOT NULL REFERENCES members(member_id),
    horizon_days     INTEGER NOT NULL,
    n_trades         INTEGER,
    n_scored         INTEGER,
    hit_rate         REAL,                    -- share of trades with positive excess return
    mean_excess      REAL,
    dollar_wtd_excess REAL,
    shrunk_excess    REAL,                    -- empirical-Bayes shrunk toward the population mean
    weight           REAL,                    -- 0..~2 multiplier used in weighted aggregates
    t_stat           REAL,
    computed_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (member_id, horizon_days)
);

-- ---------------------------------------------------------------- lobbying
CREATE TABLE IF NOT EXISTS lobbying_filings (
    filing_uuid     TEXT PRIMARY KEY,
    filing_year     INTEGER,
    filing_period   TEXT,
    filing_type     TEXT,
    dt_posted       TEXT,
    period_start    TEXT,
    period_end      TEXT,
    registrant_name TEXT,
    client_name     TEXT,
    client_id       TEXT,
    amount          REAL,
    ticker          TEXT,                     -- resolved client -> issuer
    ticker_confidence REAL
);
CREATE INDEX IF NOT EXISTS idx_lob_ticker ON lobbying_filings(ticker);
CREATE INDEX IF NOT EXISTS idx_lob_period ON lobbying_filings(period_start);

CREATE TABLE IF NOT EXISTS lobbying_activities (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_uuid   TEXT REFERENCES lobbying_filings(filing_uuid),
    issue_code    TEXT,
    description   TEXT,
    entity        TEXT                        -- government entity lobbied
);
CREATE INDEX IF NOT EXISTS idx_lobact_filing ON lobbying_activities(filing_uuid);
CREATE INDEX IF NOT EXISTS idx_lobact_code ON lobbying_activities(issue_code);

-- ---------------------------------------------------------------- event study
CREATE TABLE IF NOT EXISTS event_studies (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    scope         TEXT,                       -- 'txn' | 'ticker_cluster' | 'policy_event'
    key           TEXT,
    event_date    TEXT,
    window_start  INTEGER,
    window_end    INTEGER,
    alpha         REAL, beta REAL, r2 REAL,
    car           REAL,
    car_tstat     REAL,
    n_obs         INTEGER,
    computed_at   TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_event_key ON event_studies(scope, key);

CREATE TABLE IF NOT EXISTS policy_events (
    event_id    TEXT PRIMARY KEY,
    event_date  TEXT,
    category    TEXT,                         -- vote | hearing | executive_order | tariff | rule
    title       TEXT,
    tickers     TEXT,                         -- json array of affected tickers
    source_url  TEXT
);

-- ---------------------------------------------------------------- ops
CREATE TABLE IF NOT EXISTS ingest_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT,
    started_at   TEXT,
    finished_at  TEXT,
    status       TEXT,
    n_new_filings INTEGER DEFAULT 0,
    n_new_txns   INTEGER DEFAULT 0,
    note         TEXT
);

CREATE TABLE IF NOT EXISTS kv (
    k TEXT PRIMARY KEY,
    v TEXT
);

-- ================================================================ v2: prediction layer
CREATE TABLE IF NOT EXISTS earnings (
    ticker         TEXT NOT NULL,
    report_date    TEXT NOT NULL,          -- date the figures were released
    fiscal_period  TEXT,
    eps_actual     REAL,
    eps_estimate   REAL,
    surprise_pct   REAL,
    revenue_actual REAL,
    revenue_estimate REAL,
    rev_surprise_pct REAL,
    time_of_day    TEXT,                   -- bmo | amc | unknown
    source         TEXT,
    PRIMARY KEY (ticker, report_date)
);
CREATE INDEX IF NOT EXISTS idx_earn_date ON earnings(report_date);

CREATE TABLE IF NOT EXISTS event_index (
    theme    TEXT NOT NULL,                -- conflict | tariffs | energy | health | ai | monetary
    date     TEXT NOT NULL,
    intensity REAL,                        -- raw volume measure from the provider
    z_score  REAL,                         -- standardized against a trailing window
    source   TEXT,
    PRIMARY KEY (theme, date)
);
CREATE INDEX IF NOT EXISTS idx_event_date ON event_index(date);

CREATE TABLE IF NOT EXISTS ticker_sectors (
    ticker   TEXT PRIMARY KEY,
    sector   TEXT,
    industry TEXT,
    source   TEXT
);

-- Fitted factor weights, one row per factor per walk-forward refit.
CREATE TABLE IF NOT EXISTS factor_weights (
    fit_id       TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    as_of        TEXT NOT NULL,
    factor       TEXT NOT NULL,
    weight       REAL,
    raw_ic       REAL,                     -- standalone IC of this factor in the fit window
    stability    REAL,                     -- sign agreement across bootstrap resamples
    PRIMARY KEY (fit_id, horizon_days, factor)
);
CREATE INDEX IF NOT EXISTS idx_fw_asof ON factor_weights(as_of);

CREATE TABLE IF NOT EXISTS backtest_results (
    run_id        TEXT NOT NULL,
    horizon_days  INTEGER NOT NULL,
    fold          INTEGER,
    test_start    TEXT,
    test_end      TEXT,
    n_obs         INTEGER,
    ic            REAL,
    top_decile_ret REAL,
    bottom_decile_ret REAL,
    long_short    REAL,
    hit_rate      REAL,
    null_ic_mean  REAL,                    -- label-shuffled control
    null_ic_p95   REAL,
    PRIMARY KEY (run_id, horizon_days, fold)
);

CREATE TABLE IF NOT EXISTS predictions (
    as_of         TEXT NOT NULL,
    horizon_days  INTEGER NOT NULL,
    ticker        TEXT NOT NULL,
    rank          INTEGER,
    score         REAL,
    score_pctile  REAL,
    expected_excess REAL,
    confidence    REAL,
    attribution   TEXT,                    -- json: factor -> signed contribution
    rationale     TEXT,
    PRIMARY KEY (as_of, horizon_days, ticker)
);
CREATE INDEX IF NOT EXISTS idx_pred_asof ON predictions(as_of, horizon_days, rank);

-- Per-trade decomposition of return around the disclosure date.
CREATE TABLE IF NOT EXISTS trade_timing (
    txn_id             TEXT PRIMARY KEY REFERENCES transactions(txn_id),
    horizon_days       INTEGER,
    pre_disclosure_excess  REAL,           -- trade date -> disclosure date (nobody could trade this)
    post_disclosure_excess REAL,           -- disclosure date -> disclosure + horizon
    total_excess       REAL,
    capturable_share   REAL,               -- post / total, when total is nonzero
    lag_days           INTEGER
);
