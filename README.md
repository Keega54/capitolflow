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
| **Disclosure timing** | Splits each trade's edge at the filing date: how much happened before the public could act, how much was left |
| **Ranked picks** | Top 10 short-term (~1 month) and top 10 long-term (~6 months), with per-factor attribution |
| **Track record** | Cumulative % return vs benchmark and hit rate, with simulated and live records kept strictly apart |
| **Core universe** | The ~50 most politically-traded names, refreshed daily and guaranteed fresh prices |
| **Fitted weights** | Seven factors weighted by walk-forward regression, tested against a shuffled-label null |
| Model | Gradient-boosted forward-return model with purged walk-forward validation |

**Output** — a self-contained dashboard (light/dark, keyboard- and
screen-reader-friendly, every chart backed by a sortable table) plus a JSON API.

---

## Read this before you read a ranking

The single most important number this project produces is not a stock pick. It is
**how much of a politician's edge survives the disclosure lag.**

A member trades on day 0. The filing appears up to 45 days later. Whatever the
stock did in between is unreachable — nobody outside the filer could have acted
on it. So every trade's excess return is split in two:

```
total edge  =  before disclosure (unreachable)  +  after disclosure (capturable)
```

The dashboard's first panel shows that split, and the ranked picks are only worth
reading if the "after" number is meaningfully positive. If the edge turns out to
sit entirely before the filing goes public — which is exactly what you would
expect if the information advantage is real and the market absorbs it fast — then
the honest conclusion is that following disclosures is close to worthless, and
this project is built to tell you that rather than hide it.

Every ranking ships with a confidence banner driven by that test. When the model
fails to beat random chance, the banner says **"No demonstrated signal"** and
tells you to read the list as *where political activity is concentrated*, not as
a forecast. That is a feature.

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

## How the picks are made

Seven factors, each collapsed to one cross-sectionally standardized score:

| Factor | What it measures |
|---|---|
| `politician_flow` | net congressional buying vs selling |
| `politician_conviction` | how many distinct members, how much money |
| `politician_freshness` | how recent the disclosures are, aged by a **fitted** decay half-life |
| `lobbying` | federal lobbying spend and its year-over-year change |
| `earnings` | surprise history, time since last report, whether the next is imminent |
| `events` | current-events intensity **interacted with the stock's sector** |
| `momentum` | price trend and volatility, so the model can't relabel drift as skill |

The composite is a weighted sum. Weights come from ridge regression inside a
walk-forward window, then get shrunk toward equal weighting, scaled down by
bootstrap sign-stability, and clipped so no single factor can dominate. Seven
fitted parameters against thousands of observations is a deliberately small
budget — the temptation in this kind of project is to fit hundreds.

**On "how does a war affect stocks":** an event theme's intensity is identical for
every stock on a given day, so on its own it is useless for choosing *between*
stocks — it gets annihilated the moment you standardize across the market. What
carries information is the interaction: *is there a conflict escalating, AND does
this company sell to defense ministries.* Each theme is therefore projected onto
each stock's sector exposure. The direction and size of the effect are never
asserted; the backtest fits them from history. The themes tracked are conflict,
tariffs, energy, health, AI/semis, and monetary policy — deliberately few, because
every extra theme is another chance to find a spurious correlation.

## How overfitting is guarded against

This is where most projects like this quietly fail, so the controls are explicit:

1. **Purged, embargoed walk-forward splits.** Forward returns overlap, so any
   training row whose target window reaches into the test period is dropped.
   Random K-fold on overlapping returns leaks the future and is the single most
   common way this kind of study fools itself.
2. **A shuffled-label null.** The identical pipeline is re-run many times with
   targets randomly permuted, producing the distribution of performance that pure
   luck yields *on this data*. A real result must beat that distribution, not
   zero. Comparing to zero is not good enough — with a fitted model and
   overlapping returns, luck produces a positive score surprisingly often.
3. **Block bootstrap by company, not by row.** Rows for one ticker across
   overlapping windows are massively dependent. An ordinary row bootstrap treats
   ~200 genuinely independent observations as thousands and reports 100%
   stability for factors that are pure noise. Resampling whole companies is the
   difference between a stability number that means something and one that always
   says 1.00.
4. **Deflation for multiple testing**, and reporting the number of independent
   companies alongside the row count, so a big row count is never mistaken for
   statistical power.

The test suite plants known signals in synthetic data *and* a deliberate placebo
factor, then asserts the harness loads on the real ones and starves the placebo.

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

### When does it update?

| What | When | Why |
|---|---|---|
| Core universe recomputed | every day | a newly popular name enters the same run it qualifies on |
| Prices, disclosures, earnings, events | every day, ~11:17 UTC (~7am ET) | market data has to be current for anything else to mean anything |
| Rankings regenerated | every day | new data, new scores |
| Live picks scored | every day | cheap: only fills in outcomes whose horizon has elapsed |
| Factor weights refitted | Mondays + manual runs | see below |
| Backtested history rebuilt | Mondays + manual runs | expensive walk-forward simulation |
| Dashboard republished | every day | right after everything above |

**Why weights aren't refit daily.** The relationships being estimated move on the
scale of months. Refitting every day would mostly re-estimate noise, and you'd
watch weights jitter around and mistake that for the model learning something.
Weekly refitting with daily re-scoring gets the responsiveness without the
thrash. If you disagree, change the `if:` condition on the refit step — it's one
line.

### Reading the track record

The dashboard shows two records and never merges them:

- **Backtested** — picks reconstructed for past dates using only data available
  then. Available immediately, and useful, but still a simulation: it knows which
  companies exist today, and it never had to place a real order.
- **Live** — picks written to the database *before* the outcome was knowable, then
  scored once the horizon elapsed. This is the only real evidence, and it is
  **empty on a new install** until enough time passes. That gap is the honest
  state of things.

Neither includes trading costs, slippage, or taxes.

One trap worth knowing about, because it bit this project during development: if
you rebalance monthly but hold for six months, consecutive periods overlap by
five months, and chaining those returns compounds the same market move six times.
It showed **+350,000%** before the fix. Cumulative returns are now chained only
over periods you could actually have held back to back, and there's a test
asserting no curve exceeds a sanity ceiling.

Weights are refitted weekly rather than daily on purpose: the underlying
relationships move slowly, and refitting every day invites reading noise as
change.

You can always force a run: **Actions → refresh → Run workflow**.

### Do I need a separate website?

No. GitHub Pages *is* the free website, and it's already wired up — the workflow
publishes to it on every run. Your dashboard lives at
`https://<your-username>.github.io/capitolflow/` and updates itself with no
server, no hosting bill, and nothing running on your laptop.

If you later want a custom domain, add a `CNAME` file to the `site/` output and
point your domain's DNS at GitHub Pages; nothing else about the setup changes.

### If the dashboard comes up empty

The most likely cause is prices. Everything — returns, member scores, features,
the backtest, the rankings — is defined relative to price history, so an empty
`prices` table empties the entire site while the disclosure data sits there
perfectly intact.

The run now **fails loudly** rather than publishing a broken site: a `health
--strict` gate stops the workflow with a red X and prints exactly what is wrong.

Price providers are the usual culprit, because they treat cloud IPs differently
from laptops:

| Provider | Key | Notes |
|---|---|---|
| **yahoo** | none | the default here — generally works from GitHub Actions |
| **stooq** | none | fine locally, but **blocks datacenter IPs** and caps daily requests |
| **fmp** | yes | most reliable if you have a key; set `FMP_API_KEY` |

All three are tried in order, and a provider that starts refusing is retired for
the rest of the run instead of being asked hundreds more times. Switch the
preferred one with `CAPITOLFLOW_PRICES=stooq|yahoo|fmp`.

The benchmark and the core universe are fetched **first**, so even a run cut
short by rate limits leaves a usable database behind.

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
capitolflow analytics    # returns, accuracy scores, timing split, event studies
capitolflow context      # earnings, current-events themes, sector labels
capitolflow timing       # how much edge survives the disclosure lag
capitolflow backtest     # fit factor weights, test against a shuffled-label null
capitolflow predict      # produce the top-10 short and long term lists
capitolflow universe     # recompute the ~50-name core universe
capitolflow scoreboard   # score live picks whose horizon has elapsed
capitolflow scoreboard --simulate  # rebuild the walk-forward backtested history
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

**Honest expectations.** This is a low-signal problem. Published academic work on
congressional trading finds effects that are small, unstable across time periods,
and concentrated in exactly the window you cannot trade. A positive information
coefficient here is a hypothesis worth investigating, not a strategy. The test
suite deliberately confirms the model finds *no* signal in random-walk synthetic
prices, which is the behaviour you want from an honest harness.

**The scoreboard is the real test.** `capitolflow predict --scoreboard` compares
predictions the model made months ago against what actually happened, with no
refitting involved. Every backtest in the world can be talked into looking good;
a forward record cannot. Give it a few months before you believe anything.

**This is research tooling, not investment advice.** Nothing here accounts for
transaction costs, slippage, taxes, position sizing, or your risk tolerance, and
none of it is a recommendation to buy or sell anything.

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
