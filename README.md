# CapitolFlow

Tracks stock trading by members of the U.S. House, Senate, and executive-branch
officials — and analyzes it **on the date the trade happened**, not the date it
was disclosed. Under the STOCK Act a member can trade today and tell the public
up to 45 days from now, so any analysis keyed on filing dates is measuring the
wrong thing.

It pulls from primary sources, resolves messy asset descriptions into tickers,
scores each person's timing against a market benchmark, overlays federal
lobbying spend, and publishes a dashboard that updates itself.

---

## What you get

**Ingest** — House Clerk annual disclosure archives (XML index + per-filing PDFs,
with OCR for the scanned ones), Senate eFD (HTML tables for electronic filings,
OCR for paper), Senate/House lobbying disclosures, and optional commercial feeds
used purely as a cross-check.

**Analysis**

| Layer | What it answers |
|---|---|
| Trade aggregates | Which companies the most *distinct people* traded, and where the dollars went |
| Disclosure delay | How long the public actually waited, and who files late |
| Timing accuracy | Per-person excess return vs. benchmark at 30/90/180/365 days from the trade date |
| Cluster detection | Windows where an unusual number of people traded the same stock the same way |
| Lobbying overlay | Company lobbying spend against trading interest, with committee-assignment matching |
| Event study | Market-model cumulative abnormal returns around each trade date |
| Model | Gradient-boosted forward-return model with purged walk-forward validation |

**Output** — a self-contained dashboard (light/dark, keyboard- and
screen-reader-friendly, every chart backed by a sortable table) plus a JSON API.

---

## Quick start

```bash
git clone <your-repo> && cd capitolflow
pip install -r requirements.txt

# OCR toolchain (needed for scanned House filings)
sudo apt-get install tesseract-ocr poppler-utils      # Debian/Ubuntu
brew install tesseract poppler                        # macOS

python -m capitolflow refresh        # fetch, parse, score — the whole pipeline
python -m capitolflow serve          # http://127.0.0.1:8000
```

**Want to see it before you wait on a download?**

```bash
python -m capitolflow --db demo.db synthetic   # fabricated data, known ground truth
python -m capitolflow --db demo.db export --out site
python -m capitolflow --db demo.db serve
```

The first real `refresh` is slow — it downloads and OCRs years of PDFs. Later
runs are incremental and take minutes.

**The backfill is resumable by design.** `CAPITOLFLOW_MAX_FILINGS` is a budget for
the *whole run*, shared across every year, and years are processed newest-first.
So `refresh --full` does not try to swallow a decade of PDFs in one sitting — it
takes a bite, stops cleanly, and the next scheduled run picks up where it left
off. Your dashboard is current from day one and history fills in behind it. Raise
the budget if you want it to move faster and your job time limit allows.

---

## Making it update itself

`.github/workflows/refresh.yml` runs the pipeline daily on GitHub Actions and
publishes the dashboard to GitHub Pages. There is no server to maintain and it
costs nothing on a public repository.

1. Push this repo to GitHub.
2. **Settings → Pages → Source: GitHub Actions.**
3. **Actions → refresh → Run workflow** for the first run (kick it off with
   *full backfill* checked if you want history).

The SQLite database is carried between runs as a cache artifact, so each run only
fetches what is new. It is also uploaded as a downloadable artifact after every
run, so you always have a copy of the data.

Optional API keys go in **Settings → Secrets and variables → Actions**:
`LDA_API_KEY` (lobbying — free, from lda.gov), plus `FINNHUB_API_KEY`,
`QUIVER_API_KEY`, `FMP_API_KEY` if you want commercial cross-check feeds. Every
one is optional; the pipeline degrades gracefully without them.

---

## Commands

```
capitolflow reference    # member roster, committees, SEC ticker map
capitolflow ingest       # new disclosures  (--full to backfill)
capitolflow lobbying     # lobbying filings
capitolflow prices       # price history for every traded ticker
capitolflow analytics    # returns, accuracy scores, event studies
capitolflow model        # train the forward-return model
capitolflow model --leakage-check   # size the disclosure-lag information gap
capitolflow refresh      # all of the above, in order
capitolflow export       # write the static dashboard to ./site
capitolflow serve        # run the API + dashboard locally
capitolflow health       # row counts and data freshness
```

Configuration is entirely environment variables — see `capitolflow/config.py`.
The ones worth knowing: `CAPITOLFLOW_UA` (identify yourself to the government
servers), `CAPITOLFLOW_DELAY` (seconds between requests, default 1.0),
`CAPITOLFLOW_START_YEAR`, `CAPITOLFLOW_BENCHMARK`.

---

## Decisions that shape the numbers

**Amounts are brackets, not figures.** Filers disclose `$15,001 – $50,000`, never
an exact number. The point estimate is the **geometric** midpoint, not the
arithmetic one — these brackets span whole orders of magnitude and the underlying
distribution is closer to log-uniform, so an arithmetic midpoint systematically
overstates typical trade size. `amount_low` and `amount_high` are always kept.

**Accuracy is shrunk toward the average.** A member with four lucky trades will
show a huge raw mean excess return, and ranking on that is ranking noise. Scores
use an empirical-Bayes shrink whose strength is set by how much of the observed
spread is real skill versus sampling error. Someone with five trades gets pulled
most of the way to the population mean; someone with two hundred barely moves.

**The resolver refuses to guess.** An asset description that does not clearly
match a company resolves to `NULL` with confidence 0, not to the nearest-looking
ticker. A wrong ticker silently corrupts every return downstream, whereas an
unresolved row still counts toward volume and is visibly missing.

**Backtests use disclosure timing.** The model's features can only see trades
that had *already been publicly filed* as of the feature date. Using trade dates
would give the model information no investor had, which produces a beautiful
backtest and a worthless one. `model --leakage-check` trains both ways and reports
the gap — which is itself a measurement of how much the disclosure lag is worth.

**Validation is purged and embargoed.** Forward returns overlap, so random K-fold
leaks the future. Splits are expanding-window with a gap covering the full return
horizon.

**Honest expectations for the model.** On any real sample this is a low-signal
problem with a few thousand usable observations. Treat a positive information
coefficient as a hypothesis worth investigating, not a trading strategy. The test
suite deliberately confirms the model finds *no* signal in random-walk synthetic
prices, which is the behavior you want.

---

## Data model

```
members ──< transactions >── filings
   │            │
   │            ├──< trade_returns      (per-trade excess return by horizon)
   │            └──< event_studies      (market-model CAR around trade dates)
   ├──< member_scores                   (shrunk accuracy + weight)
   └──< committee_memberships
lobbying_filings ──< lobbying_activities
prices, securities, policy_events, ingest_runs
```

`transactions.transaction_date` is the analytical key throughout.
`filing_delay_days` is the gap between it and public disclosure.

---

## Tests

```bash
pip install -r requirements-dev.txt
python tests/build_fixtures.py       # regenerate the PDF fixtures
python -m pytest -q
```

The analytics tests run against a synthetic dataset that plants three
archetypes — consistently well-timed, consistently badly-timed, and random — and
assert that the scoring recovers that ordering, that shrinkage strengthens as
sample size falls, and that the event study separates the groups. If a refactor
breaks the math, they fail.

The parser tests assert exact values on a House PTR fixture in both its
text-layer and scanned-image forms, so a template change or an OCR regression
surfaces immediately.

---

## Adding executive-branch filers

Cabinet officials file OGE Form 278e, and there is no machine-readable roster of
them. Create `data/executive_officials.json` and `capitolflow reference` will
pick it up:

```json
[{"name": "Jane Doe", "role_title": "Secretary of the Treasury", "term_start": "2025-01-28"}]
```

Their filings are not automatically fetched — OGE's disclosure portal has no bulk
API. The roster entries exist so manually-added filings link to a real person.

---

## Scope and limits

- Members disclose their own trade dates. Nothing here can verify them.
- Some House PTRs are scans; OCR is good but not perfect. `filings.doc_format`
  records how each was read and `parse_status` flags failures — check
  `capitolflow health` after a refresh.
- Blind trusts, spousal accounts, and dependents' accounts are disclosed
  unevenly. The `owner` field is preserved so you can filter.
- Options are resolved to their underlying ticker, which discards leverage and
  understates the directional bet.
- Correlations between lobbying and trading are co-occurrence. They are not
  evidence of causation and definitely not evidence of wrongdoing.

Be polite to the source servers: they are public infrastructure, the default
1-second delay is there for a reason, and `CAPITOLFLOW_UA` should say who you are.

## License

MIT. Government disclosure data is public domain; commercial feed data follows
the vendor's terms.
